#!/usr/bin/env python
"""Train a pico-size BookDiT on the shipped synthetic toy data, in the paper's representation.

A smoke test that runs on CPU in seconds. It takes the same path as the paper's trainer
(``scripts/lib/train_bookdit.py``), shrunk to toy size:

* each LOBSTER-format day pair under ``--data-root`` (default ``quickstart/toy_data/``) is read
  as its 40 orderbook columns plus the message times, and mapped to the 41-channel
  ``midspread_dt`` chart (``scripts/lib/lob_representation.py``: log spread, mid increment, log
  ladder gaps, absolute sizes, log1p(Δt));
* every channel is standardized with the mean/std of the training day(s);
* the model is the BookDiT backbone at the ``pico`` size of ``scripts/lib/make_model_family.py``
  (embed 32, depth 1, one head: 31,209 parameters at 41 channels), trained with the
  flow-matching loss by :class:`bookdit.training.Trainer`.

What is smaller than the paper: windows of ``--window`` frames (default 100, stride 10) rather
than 500, two synthetic days rather than the corpus, and a couple of epochs. The toy days are
random walks on a regular ladder (see ``make_toy_lobster.py``), so only the fact that the run
completes is informative.

It writes ``config.yaml``, ``metrics.yaml``, a training-curve plot and
``checkpoints/{best,last}.pt`` (trainer state) into ``<out-dir>/<run-name>/``, plus
``bookdit_pico.pt``, a checkpoint with the keys of a train_bookdit.py one, which
``quickstart/inspect_checkpoint.py`` can read.

Usage::

    python quickstart/train_toy.py
    python quickstart/train_toy.py --epochs 1 --out-dir /tmp/qs --run-name my_smoke
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

QUICKSTART = Path(__file__).resolve().parent
REPO_ROOT = QUICKSTART.parent
sys.path.insert(0, str(REPO_ROOT / "scripts" / "lib"))

from bookdit.data import discover_lobster_pairs, split_lobster_pairs  # noqa: E402
from bookdit.training import Trainer  # noqa: E402
from bookdit.utils.reproducibility import set_seed  # noqa: E402
from lob_representation import book_to_midspread_dt  # noqa: E402
from make_model_family import FAMILY, build_family_model  # noqa: E402

DEFAULT_DATA = QUICKSTART / "toy_data"
REP, SIZE, ARCH = "midspread_dt", "pico", "bookdit"


def day_chart(pair) -> np.ndarray:
    """One LOBSTER day pair -> (T, 41) float32 ``midspread_dt`` chart."""
    if pair.message_path is None:
        raise SystemExit(f"{pair.book_path.name}: no message file; the chart's Δt channel needs "
                         "the message times")
    book = pd.read_csv(pair.book_path, header=None).to_numpy(np.float64)
    times = pd.read_csv(pair.message_path, header=None, usecols=[0])[0].to_numpy(np.float64)
    if len(times) != len(book):
        raise SystemExit(f"{pair.prefix}: {len(book)} book rows but {len(times)} messages")
    return book_to_midspread_dt(book[:, :40], times)


def windows(days: list[np.ndarray], window: int, stride: int) -> torch.Tensor:
    """Every ``window``-frame slice at ``stride`` across ``days`` -> (N, window, C) float32."""
    out = [d[s:s + window] for d in days for s in range(0, len(d) - window + 1, stride)]
    if not out:
        raise SystemExit(f"no day has {window} frames; lower --window")
    return torch.from_numpy(np.stack(out).astype(np.float32))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA,
                    help="directory of LOBSTER-format day pairs (default: quickstart/toy_data)")
    ap.add_argument("--out-dir", default="runs/quickstart", help="parent directory for the run dir")
    ap.add_argument("--run-name", default="toy_smoke", help="run dir name under --out-dir")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--window", type=int, default=100, help="frames per window (paper: 500)")
    ap.add_argument("--stride", type=int, default=10, help="frames between window starts")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-4, help="the paper trainer's learning rate")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu", help="'cpu' (default) or 'cuda'")
    args = ap.parse_args()

    if not args.data_root.is_dir():
        raise SystemExit(f"no such data root: {args.data_root}")

    # The last pair (by date) is held out for validation, so at least two days are needed.
    train_pairs, val_pairs = split_lobster_pairs(discover_lobster_pairs(args.data_root))
    train_days = [day_chart(p) for p in train_pairs]
    val_days = [day_chart(p) for p in val_pairs]

    # Per-channel statistics over the training frames, as build_rep_norm_stats.py computes them
    # for the corpus. The toy ladder is perfectly regular, so its gap channels are constant;
    # those keep std 1 and are only centred, instead of being divided by a vanishing std.
    frames = np.concatenate(train_days).astype(np.float64)
    mean = frames.mean(0)
    std = frames.std(0)
    std = np.where(std > 1e-6, std, 1.0)
    mean, std = mean.astype(np.float32), std.astype(np.float32)
    train_x = windows([(d - mean) / std for d in train_days], args.window, args.stride)
    val_x = windows([(d - mean) / std for d in val_days], args.window, args.stride)
    n_ch = train_x.shape[-1]

    set_seed(args.seed)
    model = build_family_model(ARCH, torch.zeros(1, args.window, n_ch), FAMILY[SIZE])
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{REP}: {n_ch} channels | train {len(train_x)} windows, val {len(val_x)} windows "
          f"of {args.window} frames | BookDiT {SIZE}: {n_params:,} parameters")

    trainer = Trainer(model=model, lr=args.lr, device=args.device, output_dir=args.out_dir,
                      run_name=args.run_name, seed=args.seed)
    trainer.logger.save_config({
        "data_root": str(args.data_root), "rep": REP, "size": SIZE, "arch": ARCH,
        "window": args.window, "stride": args.stride, "batch_size": args.batch_size,
        "lr": args.lr, "epochs": args.epochs, "seed": args.seed, "device": args.device,
    })
    gen = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_x, batch_size=args.batch_size, shuffle=True, generator=gen)
    val_loader = DataLoader(val_x, batch_size=args.batch_size, shuffle=False)
    summary = trainer.fit(train_loader, val_loader=val_loader, epochs=args.epochs)

    # The keys a train_bookdit.py checkpoint carries for the inspector and scorers.
    frames_seen = args.epochs * len(train_x) * args.window
    ckpt = trainer.run_dir / "bookdit_pico.pt"
    torch.save({"model_state_dict": {k: v.cpu() for k, v in trainer.model.state_dict().items()},
                "rep": REP, "size": SIZE,
                "arch": ARCH, "mean": torch.from_numpy(mean), "std": torch.from_numpy(std),
                "seq_len": args.window, "n_enc": n_params, "seed": args.seed,
                "flop_budget": 6.0 * n_params * frames_seen, "steps": trainer.global_step}, ckpt)
    print(f"Summary: {summary}")
    print(f"wrote {ckpt}")


if __name__ == "__main__":
    main()
