#!/usr/bin/env python
"""Structural validity of the native generated book frames, per chart-ablation arm.

The chart's guarantees are properties of the decoded coordinate, so they are read off the frames
the model emits (`gen_bookdit_on_lobbench_windows.py --dump-frames`) rather than off a replay of
IDM-decoded messages.

An empty level has price 0, which a naive ladder test reads as a violation.  Rates are therefore
reported twice: over all rows, and over rows whose ten levels are populated on both sides.

Units.  `locked`, `crossed`, `nonmono` and `neg_size` are ROW rates: the fraction of frames with at
least one violating level.  `pop_neg_size` is 0 by construction, since the populated mask already
requires positive sizes.  Sizes are read before the decode-time clip, so two finer size measures
are also written over all rows: `all_neg_size_cells`, the fraction of individual size cells that
are negative, and `all_neg_size_vol`, the negative cells' share of total absolute generated size.

Usage:
    chart_validity.py --root <dir with frames_<arm>/*.npz> --arms chart_full chart_absmid ...
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# The dump carries six arrays per window: cond_book/cond_time (the 250-frame real conditioning
# prefix), real_book/real_time (the 50-frame real continuation) and gen_book/gen_time (the 50
# frames the model produced).  Structural validity is read off gen_book; the key is required and a
# missing key raises.
def load_frames(d: Path, key: str = "gen_book", limit: int = 0):
    fs = sorted(d.glob("*.npz"))
    if limit:
        fs = fs[:limit]
    out = []
    for f in fs:
        z = np.load(f)
        if key not in z:
            raise SystemExit(f"{f}: no array {key!r}; has {list(z)}")
        a = np.asarray(z[key], dtype=np.float64)
        if a.ndim != 2 or a.shape[-1] % 4:
            raise SystemExit(f"{f}: {key} has shape {a.shape}, expected (T, 4*levels)")
        out.append(a)
    return out


def rates(books: list[np.ndarray]) -> dict:
    a = np.concatenate(books, axis=0)
    L = a.shape[-1] // 4
    ask, bid = a[:, 0::4], a[:, 2::4]
    asz, bsz = a[:, 1::4], a[:, 3::4]
    full = (ask > 0).all(1) & (bid > 0).all(1) & (asz > 0).all(1) & (bsz > 0).all(1)
    def r(mask):
        if mask.sum() == 0:
            return dict(n=0, locked=np.nan, crossed=np.nan, nonmono=np.nan, neg_size=np.nan)
        A, B, SA, SB = ask[mask], bid[mask], asz[mask], bsz[mask]
        return dict(n=int(mask.sum()),
                    locked=float(np.mean(A[:, 0] == B[:, 0])),
                    crossed=float(np.mean(B[:, 0] > A[:, 0])),
                    nonmono=float(np.mean(~(np.all(np.diff(A, axis=1) > 0, axis=1)
                                            & np.all(np.diff(B, axis=1) < 0, axis=1)))),
                    neg_size=float(np.mean((SA < 0).any(1) | (SB < 0).any(1))))
    all_rows, pop = r(np.ones(len(a), bool)), r(full)
    sizes = np.concatenate([asz, bsz], axis=1)
    tot = np.abs(sizes).sum()
    return {"levels": L, "n_windows": len(books), "n_rows": len(a),
            "populated_rate": float(full.mean()),
            **{f"all_{k}": v for k, v in all_rows.items() if k != "n"},
            "all_neg_size_cells": float(np.mean(sizes < 0)),
            "all_neg_size_vol": float(-sizes[sizes < 0].sum() / tot) if tot > 0 else np.nan,
            **{f"pop_{k}": v for k, v in pop.items()}}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--limit", type=int, default=0, help="windows per arm (0 = all)")
    ap.add_argument("--key", default="gen_book", choices=["gen_book", "real_book", "cond_book"],
                    help="which array of the dump to score; `real_book` gives the reference row")
    ap.add_argument("--out", default="runs/lobbench/chart_ablation/validity.csv")
    a = ap.parse_args()
    rows = []
    for arm in a.arms:
        d = Path(a.root) / f"frames_{arm}"
        if not d.is_dir():
            print(f"  SKIP {arm}: no {d}")
            continue
        books = load_frames(d, a.key, a.limit)
        if not books:
            print(f"  SKIP {arm}: no frames in {d}")
            continue
        rows.append({"arm": arm, "key": a.key, **rates(books)})
        print(f"  {arm}: {rows[-1]['n_windows']} windows, {rows[-1]['n_rows']} rows", flush=True)
    df = pd.DataFrame(rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    cols = ["arm", "n_rows", "populated_rate", "pop_locked", "pop_crossed", "pop_nonmono",
            "all_crossed", "all_nonmono", "all_neg_size_cells", "all_neg_size_vol"]
    print(df[[c for c in cols if c in df]].to_string(index=False))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
