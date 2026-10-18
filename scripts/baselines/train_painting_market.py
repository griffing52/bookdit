#!/usr/bin/env python
"""Reproduce Backhouse et al. (2025), "Painting the market".

This is a Level-2 conditional DDPM baseline, not BookDiT code.  A 256-state
window (156 observed + 100 generated) is represented as a two-channel image
(price, signed size), with the 20 book levels expanded to a square 256x256
image.  The denoiser receives [noisy image, clean masked history, mask].

The paper's source code and weights were not public when this reproduction
was built, so it reproduces the architecture and training protocol described
in arXiv:2509.05107v1 rather than claiming source-level identity.  It reads
the corpus manifest (scripts/data/build_corpus.py) lazily: the 1.3B-frame
corpus must never be stitched into RAM.

The defaults are the run reported in the paper: one pass over the GOOG
training split (235 trading days) at effective batch 32 (batch 1 x 32
accumulation), i.e. 11,622 optimizer steps, base width 128.  Pass
`--ticker all` to train on every ticker in the manifest instead.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset, get_worker_info


HISTORY = 156
FUTURE = 100
WIDTH = HISTORY + FUTURE


def _load_day(path: str) -> torch.Tensor:
    obj = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(obj, dict):
        return obj["books"].float()
    # Manifest rows can also point at overlapping (N,100,40), stride-20 window caches.
    if obj.ndim != 3 or obj.shape[1] != 100:
        raise ValueError(f"unsupported tensor shape {tuple(obj.shape)} in {path}")
    return torch.cat((obj[:, :20].reshape(-1, 40), obj[-1, 20:]), dim=0).float()


def normalize_book(
    book: torch.Tensor,
    norm_history: int = HISTORY,
    clip_quantile: float | None = 0.95,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    """Causally normalize a book image and return invertible scale values."""
    if book.ndim != 2 or book.shape[1] != 40:
        raise ValueError(f"expected (T,40), got {tuple(book.shape)}")
    if len(book) not in (HISTORY, WIDTH):
        raise ValueError(f"expected T={HISTORY} or {WIDTH}, got {len(book)}")
    ask_p = book[:, 0::4]
    ask_v = book[:, 1::4]
    bid_p = book[:, 2::4]
    bid_v = book[:, 3::4]
    price = torch.cat((bid_p.flip(1), ask_p), 1).T
    size = torch.cat((bid_v.flip(1), -ask_v), 1).T
    mid0 = (ask_p[0, 0] + bid_p[0, 0]) / 2
    price = price - mid0
    n = min(int(norm_history), price.shape[1])
    pstd = price[:, :n].std().clamp_min(1e-6)
    smean = size[:, :n].mean()
    sstd = size[:, :n].std().clamp_min(1e-6)
    price = price / pstd
    size = (size - smean) / sstd
    if clip_quantile is not None:
        q = float(clip_quantile)
        if not 0.0 < q <= 1.0:
            raise ValueError(f"clip_quantile must lie in (0,1], got {q}")
        pclip = torch.quantile(price[:, :n].abs().flatten(), q).clamp_min(1e-6)
        sclip = torch.quantile(size[:, :n].abs().flatten(), q).clamp_min(1e-6)
        price = price.clamp(-pclip, pclip)
        size = size.clamp(-sclip, sclip)
    else:
        pclip = torch.tensor(float("inf"), device=book.device)
        sclip = torch.tensor(float("inf"), device=book.device)
    return torch.stack((price, size)), (mid0, pstd, smean, sstd, pclip, sclip)


def book_image(book: torch.Tensor, norm_history: int = HISTORY,
               clip_quantile: float | None = 0.95) -> torch.Tensor:
    """LOBSTER book40 (T,40) -> normalized (2,256,256) paper layout."""
    x, _ = normalize_book(book, norm_history, clip_quantile)
    x = x[None]
    return F.interpolate(x, size=(256, WIDTH), mode="nearest").squeeze(0)


class LazyWindows(IterableDataset):
    def __init__(self, manifest: str, split: str, seed: int, ticker: str | None,
                 norm_history: int, clip_quantile: float | None):
        rows = [r for r in csv.DictReader(open(manifest))
                if r.get("split") == split
                and (ticker is None or r.get("ticker") == ticker)
                and Path(os.path.expandvars(r["tensor_path"])).exists()]
        if not rows:
            raise RuntimeError(f"no readable {split} rows for ticker={ticker!r} in {manifest}")
        # `${LOBSTER_CACHE}`-style placeholders (as in the shipped manifest) resolve from the env
        self.paths = [os.path.expandvars(r["tensor_path"]) for r in rows]
        self.seed = seed
        self.ticker = ticker
        self.norm_history = norm_history
        self.clip_quantile = clip_quantile

    def __iter__(self):
        wi = get_worker_info()
        rank, workers = (wi.id, wi.num_workers) if wi else (0, 1)
        rng = random.Random(self.seed + rank)
        paths = self.paths[rank::workers]
        while True:
            rng.shuffle(paths)
            for path in paths:
                day = _load_day(path)
                if len(day) < WIDTH:
                    continue
                # The paper advances by the complete history length so examples
                # contain fresh data. A randomized epoch offset avoids always
                # selecting the same modulo-WIDTH grid.
                offset = rng.randrange(min(WIDTH, len(day) - WIDTH + 1))
                for s in range(offset, len(day) - WIDTH + 1, WIDTH):
                    yield book_image(day[s:s + WIDTH], self.norm_history,
                                     self.clip_quantile)


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    f = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / max(half - 1, 1))
    z = t.float()[:, None] * f[None]
    return torch.cat((z.sin(), z.cos()), 1)


class ResBlock(nn.Module):
    def __init__(self, cin: int, cout: int, tdim: int):
        super().__init__()
        self.n1, self.c1 = nn.GroupNorm(min(32, cin), cin), nn.Conv2d(cin, cout, 3, padding=1)
        self.n2, self.c2 = nn.GroupNorm(min(32, cout), cout), nn.Conv2d(cout, cout, 3, padding=1)
        self.temb = nn.Linear(tdim, 2 * cout)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, te):
        h = self.c1(F.silu(self.n1(x)))
        scale, shift = self.temb(F.silu(te)).chunk(2, 1)
        h = self.n2(h) * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        return self.skip(x) + self.c2(F.silu(h))


class Attention(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.norm = nn.GroupNorm(min(32, c), c)
        self.qkv = nn.Conv1d(c, 3 * c, 1)
        self.out = nn.Conv1d(c, c, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        z = self.norm(x).reshape(b, c, h * w)
        q, k, v = self.qkv(z).chunk(3, 1)
        a = torch.softmax(torch.bmm(q.transpose(1, 2), k) / math.sqrt(c), -1)
        z = self.out(torch.bmm(v, a.transpose(1, 2))).reshape(b, c, h, w)
        return x + z


class PaintingUNet(nn.Module):
    """Six-resolution UNet; paper setting is base=128, mults=1,1,2,2,4,4."""
    def __init__(self, base=128, mults=(1, 1, 2, 2, 4, 4)):
        super().__init__()
        cs, td = [base * m for m in mults], base * 4
        self.tnet = nn.Sequential(nn.Linear(base, td), nn.SiLU(), nn.Linear(td, td))
        self.base = base
        self.inp = nn.Conv2d(5, cs[0], 3, padding=1)
        self.down, self.ds = nn.ModuleList(), nn.ModuleList()
        c = cs[0]
        for i, co in enumerate(cs):
            self.down.append(nn.ModuleList((ResBlock(c, co, td), ResBlock(co, co, td),
                                            Attention(co) if i == 4 else nn.Identity())))
            c = co
            if i < len(cs) - 1:
                self.ds.append(nn.Conv2d(c, c, 3, stride=2, padding=1))
        self.mid = nn.ModuleList((ResBlock(c, c, td), Attention(c), ResBlock(c, c, td)))
        self.up, self.us = nn.ModuleList(), nn.ModuleList()
        for i in reversed(range(len(cs))):
            self.up.append(nn.ModuleList((ResBlock(c + cs[i], cs[i], td), ResBlock(cs[i], cs[i], td),
                                          Attention(cs[i]) if i == 4 else nn.Identity())))
            c = cs[i]
            if i:
                self.us.append(nn.Conv2d(c, cs[i - 1], 3, padding=1))
                c = cs[i - 1]
        self.out = nn.Sequential(nn.GroupNorm(min(32, c), c), nn.SiLU(), nn.Conv2d(c, 2, 3, padding=1))

    def forward(self, x, t):
        te = self.tnet(timestep_embedding(t, self.base))
        h, skips = self.inp(x), []
        for i, (a, b, attn) in enumerate(self.down):
            h = attn(b(a(h, te), te)); skips.append(h)
            if i < len(self.ds): h = self.ds[i](h)
        h = self.mid[2](self.mid[1](self.mid[0](h, te)), te)
        for j, (a, b, attn) in enumerate(self.up):
            h = attn(b(a(torch.cat((h, skips.pop()), 1), te), te))
            if j < len(self.us): h = F.interpolate(h, scale_factor=2, mode="nearest"); h = self.us[j](h)
        return self.out(h)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", default="runs/crossasset/corpus_manifest_2024.csv")
    p.add_argument("--out", default="runs/baselines/painting_market/checkpoint.pt")
    p.add_argument("--steps", type=int, default=11_622,
                   help="optimizer steps; the default is one pass over the GOOG training split "
                        "at effective batch 32 (the paper's run)")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--accum", type=int, default=32)
    p.add_argument("--base", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--diffusion-steps", type=int, default=1000)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-every", type=int, default=1000)
    p.add_argument("--ticker", default="GOOG",
                   help="train on this ticker's rows only; 'all' uses every ticker")
    p.add_argument("--norm-history", type=int, default=HISTORY)
    p.add_argument("--clip-quantile", type=float, default=0.95)
    p.add_argument("--resume", default=None)
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    if a.ticker.lower() == "all":
        a.ticker = None
    if a.smoke:
        a.steps, a.base, a.workers, a.save_every = 2, 16, 0, 1
    clip_quantile = None if a.clip_quantile <= 0 else a.clip_quantile
    torch.manual_seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PaintingUNet(a.base).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
    start_step = 0
    if a.resume:
        resume = torch.load(a.resume, map_location="cpu", weights_only=False)
        expected = {"base": a.base, "diffusion_steps": a.diffusion_steps,
                    "history": HISTORY, "future": FUTURE, "ticker": a.ticker}
        for key, value in expected.items():
            if resume.get(key) != value:
                raise ValueError(f"resume mismatch for {key}: {resume.get(key)!r} != {value!r}")
        expected_norm = {"history": a.norm_history, "clip_quantile": clip_quantile}
        if resume.get("normalization") != expected_norm:
            raise ValueError("resume normalization mismatch")
        model.load_state_dict(resume["model_state_dict"])
        opt.load_state_dict(resume["optimizer_state_dict"])
        start_step = int(resume["step"])
    beta = torch.linspace(1e-4, 0.02, a.diffusion_steps, device=dev)
    abar = torch.cumprod(1 - beta, 0)
    dataset = LazyWindows(a.manifest, "train", a.seed, a.ticker,
                          a.norm_history, clip_quantile)
    loader = iter(DataLoader(dataset, batch_size=a.batch_size,
                             num_workers=a.workers, pin_memory=True, persistent_workers=a.workers > 0))
    mask = torch.zeros(1, 1, 256, WIDTH, device=dev); mask[..., HISTORY:] = 1
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    nparams = sum(x.numel() for x in model.parameters())
    print(f"device={dev} params={nparams:,} base={a.base} batch={a.batch_size} accum={a.accum} "
          f"ticker={a.ticker} train_days={len(dataset.paths)} norm_history={a.norm_history} "
          f"clip_quantile={clip_quantile} start_step={start_step} target_step={a.steps}",
          flush=True)
    t0 = time.time()
    for step in range(start_step + 1, a.steps + 1):
        opt.zero_grad(set_to_none=True); total = 0.0
        for _ in range(a.accum if not a.smoke else 1):
            clean = next(loader).to(dev, non_blocking=True)
            t = torch.randint(a.diffusion_steps, (len(clean),), device=dev)
            eps = torch.randn_like(clean)
            at = abar[t, None, None, None]
            noisy = at.sqrt() * clean + (1 - at).sqrt() * eps
            hist = clean * (1 - mask)
            pred = model(torch.cat((noisy, hist, mask.expand(len(clean), -1, -1, -1)), 1), t)
            loss = ((pred - eps).square() * mask).sum() / (mask.sum() * len(clean) * 2)
            (loss / (a.accum if not a.smoke else 1)).backward(); total += loss.item()
        opt.step()
        if step == 1 or step % 10 == 0:
            mem = torch.cuda.max_memory_allocated() / 2**30 if dev.type == "cuda" else 0
            print(f"step={step} loss={total/(a.accum if not a.smoke else 1):.6f} "
                  f"sec/step={(time.time()-t0)/step:.2f} peak_GiB={mem:.2f}", flush=True)
        if step % a.save_every == 0 or step == a.steps:
            torch.save({"model_state_dict": model.state_dict(),
                        "optimizer_state_dict": opt.state_dict(), "base": a.base,
                        "diffusion_steps": a.diffusion_steps, "history": HISTORY, "future": FUTURE,
                        "step": step, "seed": a.seed, "manifest": a.manifest,
                        "paper": "arXiv:2509.05107v1", "params": nparams,
                        "ticker": a.ticker,
                        "normalization": {"history": a.norm_history,
                                          "clip_quantile": clip_quantile},
                        "effective_batch": a.batch_size * a.accum,
                        "train_days": len(dataset.paths)}, out)
            print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
