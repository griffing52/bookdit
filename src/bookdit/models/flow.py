"""Rectified-flow order-book backbone.

``BookDiT`` is the Wan2.2 video-diffusion attention block adapted to limit-order-book
frames: RMSNorm with per-head qk-norm, 1D RoPE, 6-way timestep modulation and
bidirectional self-attention (the text cross-attention of the original block is dropped).
The architecture follows Wan2.2 (https://github.com/Wan-Video/Wan2.2, Apache-2.0); the
blocks are reimplemented here, not copied.

Objective (rectified-flow / conditional-OT, t in [0,1], t=1 == data):
    x0 ~ N(0, I) ;  x_t = (1-t) x0 + t x1 ;  v* = x1 - x0 ;  loss = MSE(net(x_t,t), v*)

One token per book frame, so the token budget is ``seq_len``.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


# =============================================================================
# Shared flow-matching base
# =============================================================================
class _FlowMatchingBase(nn.Module):
    """Mixin owning the rectified-flow objective.

    Subclasses must set ``self.final_norm`` (LayerNorm/RMSNorm over ``hidden``) and
    ``self.velocity_head`` (Linear hidden->input_dim, zero-init), and implement
    ``forward_features(x, t) -> (B, T, hidden)`` (the trunk over noisy input ``x`` at
    continuous timestep ``t``).
    """

    @staticmethod
    def _squeeze(x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:  # (B, 1, T, D) -> (B, T, D)
            x = x.squeeze(1)
        return x

    def forward(self, x: torch.Tensor) -> dict:
        x = self._squeeze(x).float()
        b = x.shape[0]
        t = torch.rand(b, device=x.device, dtype=x.dtype)
        x0 = torch.randn_like(x)                       # noise
        tt = t.view(b, 1, 1)
        x_t = (1.0 - tt) * x0 + tt * x                 # t=1 -> data, t=0 -> noise
        v_target = x - x0                              # constant-velocity OT path
        feats = self.forward_features(x_t, t)
        v_pred = self.velocity_head(self.final_norm(feats))
        loss = F.mse_loss(v_pred, v_target)
        return {"loss": loss, "v_pred": v_pred}


# =============================================================================
# Continuous timestep embedding (shared)
# =============================================================================
def _timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
    """Sinusoidal embedding for continuous t in [0,1], scaled to ~[0,1000] (Wan2.1 conv)."""
    t_scaled = t.float() * 1000.0
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device).float() / half)
    args = t_scaled[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


# =============================================================================
# 1D RoPE (shared, LLaMA-style)
# =============================================================================
def _rope_cache(seq_len: int, head_dim: int, device, base: float = 10000.0):
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    pos = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(pos, inv_freq)                 # (T, head_dim/2)
    return freqs.cos(), freqs.sin()


def _apply_rope(q, k, cos, sin):
    # q,k: (B, H, T, Dh); cos,sin: (T, Dh/2)
    cos = cos[None, None]
    sin = sin[None, None]

    def rot(x):
        xe, xo = x[..., 0::2], x[..., 1::2]
        return torch.stack([xe * cos - xo * sin, xe * sin + xo * cos], dim=-1).flatten(-2)

    return rot(q), rot(k)


# =============================================================================
# Wan2.2-style attention block -- BIDIRECTIONAL
# =============================================================================
class _WanRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        n = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return n.type_as(x) * self.weight


class _WanSelfAttention(nn.Module):
    """Wan self-attention: per-head qk-RMSNorm + RoPE, full (bidirectional) attention."""

    def __init__(self, dim, num_heads, eps=1e-6):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = _WanRMSNorm(self.head_dim, eps=eps)
        self.norm_k = _WanRMSNorm(self.head_dim, eps=eps)

    def forward(self, x):
        b, l, _ = x.shape
        n, hd = self.num_heads, self.head_dim
        q = self.norm_q(self.q(x).view(b, l, n, hd)).transpose(1, 2)
        k = self.norm_k(self.k(x).view(b, l, n, hd)).transpose(1, 2)
        v = self.v(x).view(b, l, n, hd).transpose(1, 2)
        cos, sin = _rope_cache(l, hd, x.device)
        q, k = _apply_rope(q, k, cos, sin)
        out = F.scaled_dot_product_attention(q, k, v)   # bidirectional
        return self.o(out.transpose(1, 2).reshape(b, l, n * hd))


class _WanBlock(nn.Module):
    """WanAttentionBlock minus the text cross-attention; 6-way AdaLN-style modulation."""

    def __init__(self, dim, num_heads, mlp_ratio, eps=1e-6):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.attn = _WanSelfAttention(dim, num_heads, eps)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        h = int(dim * mlp_ratio)
        self.ffn = nn.Sequential(nn.Linear(dim, h), nn.GELU(approximate="tanh"), nn.Linear(h, dim))
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(self, x, e):
        # e: (B, 6, dim) shared timestep signal; self.modulation: per-block bias.
        # Each chunk is (B, 1, dim) -> broadcasts over the L (token) axis of x.
        m = (self.modulation + e).chunk(6, dim=1)
        x = x + m[2] * self.attn(self.norm1(x) * (1 + m[1]) + m[0])
        x = x + m[5] * self.ffn(self.norm2(x) * (1 + m[4]) + m[3])
        return x


class BookDiT(_FlowMatchingBase):
    """Bidirectional Wan2.2-style DiT used as the BookDiT velocity field.

    The network has no dropout layers. Extra keyword arguments (e.g. a ``dropout`` passed by a
    generic config) are accepted and ignored, so every caller can build it with one signature.
    """

    def __init__(self, *, input_dim: int, seq_len: int, embed_dim: int = 128, depth: int = 4,
                 num_heads: int = 4, mlp_ratio: float = 4.0, freq_dim: int = 256, **_ignored):
        super().__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.freq_dim = freq_dim
        self.input_proj = nn.Linear(input_dim, embed_dim)
        # time embedding -> shared 6-way modulation signal (Wan2.2 time_projection)
        self.time_embedding = nn.Sequential(nn.Linear(freq_dim, embed_dim), nn.SiLU(),
                                             nn.Linear(embed_dim, embed_dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(embed_dim, 6 * embed_dim))
        self.blocks = nn.ModuleList([_WanBlock(embed_dim, num_heads, mlp_ratio) for _ in range(depth)])
        self.final_norm = nn.LayerNorm(embed_dim)
        self.velocity_head = nn.Linear(embed_dim, input_dim)
        nn.init.zeros_(self.velocity_head.weight)
        nn.init.zeros_(self.velocity_head.bias)

    def forward_features(self, x, t):
        tok = self.input_proj(x)
        e = self.time_embedding(_timestep_embedding(t, self.freq_dim).to(tok.dtype))
        e = self.time_projection(e).view(-1, 6, tok.shape[-1])   # (B, 6, dim)
        for blk in self.blocks:
            tok = blk(tok, e)
        return tok
