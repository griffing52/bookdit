#!/usr/bin/env python
"""Dump the metadata of a BookDiT checkpoint without building a model.

A checkpoint carries its training metadata alongside the weights -- architecture,
representation name, context length and the per-channel normalization -- so the scorers
rebuild a model from the file alone, with no config to keep in sync. This script reads a
checkpoint and prints that metadata; it writes nothing. No weights ship with this repository:
pass the toy checkpoint from ``train_toy.py`` or one you trained with
``scripts/lib/train_bookdit.py``.

    python quickstart/inspect_checkpoint.py runs/quickstart/toy_smoke/bookdit_pico.pt
    python quickstart/inspect_checkpoint.py <ckpt> --state-dict      # + every tensor shape

The file is opened with ``torch.load(..., weights_only=True)``, which refuses arbitrary pickled
objects; every checkpoint this repository writes holds only tensors and plain Python values.
Paths are printed relative to the repository root when they sit inside it, so the output is
machine-independent.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

QUICKSTART = Path(__file__).resolve().parent
REPO_ROOT = QUICKSTART.parent

# Every key a BookDiT checkpoint carries. The generation scripts read model_state_dict, rep,
# size, arch, mean, std and seq_len to rebuild the model; the rest record how it was trained.
CONTRACT = {"model_state_dict", "rep", "size", "arch", "mean", "std",
            "seq_len", "flop_budget", "seed", "n_enc"}


def _display(path: Path) -> str:
    """Repo-relative when possible -- keeps the output free of machine-specific paths."""
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def _human(n_bytes: int) -> str:
    return f"{n_bytes / 1024:.0f} KB" if n_bytes < 1024 * 1024 else f"{n_bytes / 1024 / 1024:.1f} MB"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("ckpt", type=Path,
                    help="checkpoint to inspect, e.g. runs/quickstart/toy_smoke/bookdit_pico.pt")
    ap.add_argument("--state-dict", action="store_true",
                    help="also list every model_state_dict tensor and its shape")
    args = ap.parse_args()

    ckpt = args.ckpt
    if not ckpt.exists():
        raise SystemExit(f"no such checkpoint: {_display(ckpt)}")

    ck = torch.load(ckpt, map_location="cpu", weights_only=True)
    if not isinstance(ck, dict):
        raise SystemExit(f"expected a dict checkpoint, got {type(ck).__name__}")

    print(f"{_display(ckpt)}  ({_human(ckpt.stat().st_size)})")
    print(f"keys: {sorted(ck)}")
    print()

    for key in sorted(ck):
        if key == "model_state_dict":
            continue
        val = ck[key]
        if isinstance(val, torch.Tensor):
            print(f"  {key}: tensor{tuple(val.shape)}  dtype={val.dtype}")
        else:
            print(f"  {key}: {val!r}")

    sd = ck.get("model_state_dict")
    if isinstance(sd, dict):
        n_params = sum(v.numel() for v in sd.values() if isinstance(v, torch.Tensor))
        print(f"\n  model_state_dict: {len(sd)} tensors, {n_params:,} parameters total")
        # n_enc is the parameter count recorded at training time, so a mismatch here is a broken file.
        n_enc = ck.get("n_enc")
        if isinstance(n_enc, int) and n_enc != n_params:
            print(f"  WARNING: n_enc={n_enc:,} disagrees with the state dict ({n_params:,})")
        if args.state_dict:
            for name, tensor in sd.items():
                shape = tuple(tensor.shape) if isinstance(tensor, torch.Tensor) else "?"
                print(f"    {name}: {shape}")

    missing = CONTRACT - set(ck)
    if missing:
        print(f"\n  NOTE: missing key(s) {sorted(missing)} that a BookDiT checkpoint carries.")


if __name__ == "__main__":
    main()
