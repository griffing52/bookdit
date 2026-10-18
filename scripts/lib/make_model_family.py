#!/usr/bin/env python
"""Model-size family with constant architectural ratios.

Every member has head_dim=32, mlp_ratio=4 and depth = embed_dim/32, so changing
``embed_dim`` walks an iso-shape line and any change in loss is attributable to
capacity rather than shape.  Exposes ``FAMILY`` (name -> shape dict) and
``build_family_model``, which instantiates one member at a given input shape.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


def _member(embed_dim: int, *, head_dim: int = 32) -> dict:
    """One iso-shape family member derived from its width."""
    return {
        "embed_dim": embed_dim,
        "depth": max(1, embed_dim // 32),
        "num_heads": max(1, embed_dim // head_dim),
    }


# Width-anchored family. "mini" (embed=128, depth=4; 953,769 parameters at the 41-channel,
# 500-frame shape) is the paper's BookDiT; "pico" (embed=32, depth=1) is the quickstart size.
FAMILY: dict[str, dict] = {
    "pico": _member(32),
    "nano": _member(48),
    "micro": _member(64),
    "milli": _member(96),
    "mini": _member(128),
    "midi": _member(160),
    "small": _member(192),
    "base": _member(256),
    "large": _member(384),
    "xlarge": _member(512),
}


def build_family_model(model_name, sample_x, member):
    """Instantiate ``model_name`` at family size ``member``, shaped by ``sample_x``."""
    from bookdit.models import build_model

    seq_len, input_dim = sample_x.shape[-2], sample_x.shape[-1]
    return build_model(
        model_name,
        input_dim=input_dim,
        seq_len=seq_len,
        embed_dim=member["embed_dim"],
        depth=member["depth"],
        num_heads=member["num_heads"],
        mlp_ratio=4,
    )
