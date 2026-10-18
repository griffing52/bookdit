#!/usr/bin/env python
"""Exact streaming channel mean/std for variable-depth LOB representations over a manifest.

The corpus does not fit in RAM, so the statistics are accumulated in one streaming pass.  Sums are
exact (float64, per-day partials combined by addition), so the result is identical to an in-memory
computation over the same days -- no subsampling.

Because `book_to_midspread_dt_logvol_trades` is literally
`concat(book_to_midspread_dt_logvol(book, t), log1p(trades))`, the first 41 entries of a
`midspread_dt_logvol_trades` result are exactly the statistics of `midspread_dt_logvol` (log
sizes), so one pass serves both of those reps. They are NOT the statistics of `midspread_dt`,
which keeps sizes in absolute shares and needs its own pass.

Each shard writes partial sums, which the trainer rejects; combine the shards first. The combined
path below is the one `train_bookdit.py --norm-stats` reads by default.

Usage
    python scripts/lib/build_rep_norm_stats.py --shard 0 --nshards 12 \
        --out runs/crossasset/stats_p0.npz
    python scripts/lib/build_rep_norm_stats.py --combine 'runs/crossasset/stats_p*.npz' \
        --out runs/crossasset/norm_midspread_dt.npz
"""
from __future__ import annotations

import argparse
import csv
import glob
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lob_representation import (book_to_midspread_dt,
                                book_to_midspread_dt_logvol,                   # noqa: E402
                                book_to_midspread_dt_logvol_trades,
                                book_to_chart_dt, CHART_ABLATIONS)

REPS = (("midspread_dt", "midspread_dt_logvol", "midspread_dt_logvol_trades")
        + tuple(CHART_ABLATIONS))


def day_rep(contig: Path, rep: str = "midspread_dt") -> np.ndarray | None:
    """One contig day in ``rep`` space, or None when required auxiliary data is absent."""
    d = torch.load(contig, map_location="cpu", weights_only=True)
    book = d["books"].float().numpy()
    tday = d["times"].double().numpy()
    if book.ndim != 2 or book.shape[1] % 4:
        raise ValueError(f"{contig.name}: books must have shape (T, 4*levels), got {book.shape}")
    if len(tday) != len(book):
        raise ValueError(f"{contig.name}: time/book frame mismatch: {len(tday)} vs {len(book)}")
    if rep == "midspread_dt":            # absolute sizes (the paper's model's rep)
        return book_to_midspread_dt(book, tday)
    if rep in CHART_ABLATIONS:           # chart ablation ladder; chart_full == midspread_dt
        return book_to_chart_dt(book, tday, CHART_ABLATIONS[rep])
    if rep == "midspread_dt_logvol":
        return book_to_midspread_dt_logvol(book, tday)
    if rep != "midspread_dt_logvol_trades":
        raise ValueError(f"unknown representation: {rep}")
    if "trades" not in d:
        return None
    trades = d["trades"].float().numpy()
    if trades.shape != (len(book), 4):
        raise ValueError(f"{contig.name}: trades must have shape ({len(book)}, 4), "
                         f"got {trades.shape}")
    return book_to_midspread_dt_logvol_trades(book, tday, trades)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default="runs/crossasset/corpus_manifest_2024.csv")
    ap.add_argument("--split", default="train")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--combine", nargs="*", default=None,
                    help="glob(s) of partial npz files to sum into --out")
    ap.add_argument("--rep", default="midspread_dt", choices=REPS)
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    if args.combine:
        files = sorted({f for g in args.combine for f in glob.glob(g)})
        n = 0.0
        s = ss = None
        nd = 0
        partial_reps = set()
        for f in files:
            z = np.load(f)
            if "rep" in z:
                partial_reps.add(str(z["rep"].item()))
            zs, zss = z["s"], z["ss"]
            if zs.ndim != 1 or zss.shape != zs.shape:
                raise SystemExit(f"invalid partial {f}: s/ss shape {zs.shape}/{zss.shape}")
            if s is None:
                s, ss = np.zeros_like(zs, dtype=np.float64), np.zeros_like(zss, dtype=np.float64)
            if zs.shape != s.shape:
                raise SystemExit(f"partial dimension mismatch: {f} has C={len(zs)}, expected {len(s)}")
            n += float(z["n"]); s += zs; ss += zss; nd += int(z["n_days"])
        if not files or s is None or n <= 0:
            raise SystemExit("no nonempty partial statistics to combine")
        if len(partial_reps) > 1:
            raise SystemExit(f"partial representation mismatch: {sorted(partial_reps)}")
        mean = s / n
        std = np.maximum(np.sqrt(np.maximum(ss / n - mean ** 2, 0.0)), 1e-6)
        meta = {"rep": next(iter(partial_reps))} if partial_reps else {}
        np.savez(out, mean=mean.astype(np.float32), std=std.astype(np.float32),
                 n=np.float64(n), n_days=np.int64(nd), n_parts=np.int64(len(files)), **meta)
        print(f"combined {len(files)} partials: {nd} days, {n/1e9:.4f} B frames -> {out}")
        print("mean[:5]", mean[:5], "\nstd[:5]", std[:5])
        print(f"channels={len(mean)}", "\ntail mean", mean[-6:], "\ntail std", std[-6:])
        return 0

    rows = [r for r in csv.DictReader(open(args.manifest)) if r.get("split") == args.split]
    jobs = [r for i, r in enumerate(rows) if i % args.nshards == args.shard]
    n = 0.0
    s = ss = None
    nd = 0
    t0 = time.time()
    for r in jobs:
        x = day_rep(Path(r["tensor_path"]), args.rep)
        if x is None:
            print(f"NODATA[{args.rep}] {Path(r['tensor_path']).name}", flush=True)
            continue
        xd = x.astype(np.float64)
        if s is None:
            s = np.zeros(xd.shape[-1], np.float64)
            ss = np.zeros(xd.shape[-1], np.float64)
        if xd.ndim != 2 or xd.shape[-1] != len(s):
            raise SystemExit(f"representation dimension mismatch for {r['tensor_path']}: "
                             f"got {xd.shape}, expected (*,{len(s)})")
        n += len(xd); s += xd.sum(0); ss += (xd ** 2).sum(0); nd += 1
        if nd % 50 == 0:
            print(f"[shard {args.shard}] {nd} days, {n/1e6:.0f}M frames, "
                  f"{time.time()-t0:.0f}s", flush=True)
    if s is None or n <= 0:
        raise SystemExit(f"no loadable {args.rep} days in shard {args.shard}")
    np.savez(out, n=np.float64(n), s=s, ss=ss, n_days=np.int64(nd), rep=args.rep)
    print(f"SHARD_DONE {args.shard} days={nd} frames={n:.0f} secs={time.time()-t0:.0f} -> {out}",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
