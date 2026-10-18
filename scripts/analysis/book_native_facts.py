#!/usr/bin/env python
"""Clock-free book-state statistics: BookDiT, message models and Painting on one footing.

These statistics are computed on the order book itself, so book-state and message models are
compared without an inverse-dynamics decode on either side.  A message model's books come from
replaying its own messages through the matching engine, which is the same replay LOB-Bench
performs.

WHY ONLY *CLOCK-FREE* STATISTICS.  The arms do not share an event clock: over the identical
500-message window the on-disk row counts are real 377, BookDiT 276, LOB-S5 495, Painting 1,151.
Anything defined per row or per event therefore measures a different amount of market time in each
arm -- that alone accounts for most of the apparent volatility gap between BookDiT and LOB-S5
(495/276 = 1.79x).  The five statistics below are functionals of the MARGINAL DISTRIBUTION OF BOOK
STATES and do not reference consecutive rows, time, or row ordering at all, so no clock enters:

  1. spread                    ask_1 - bid_1, in ticks
  2. depth profile             mean share of total volume at each level (20 numbers, sums to 1)
  3. inter-level gaps          ask_k - ask_{k-1} and bid_{k-1} - bid_k, in ticks
  4. populated levels          how many of the ten levels carry a price and a size on both sides
  5. touch imbalance           (bid_sz_1 - ask_sz_1) / (bid_sz_1 + ask_sz_1)

Statistics that reference the clock -- return kurtosis, realised variance, autocorrelations,
variance ratios -- are in the time-gridded `book_time_facts.py` instead.

SAMPLE-SIZE MATCHING.  Wasserstein and total-variation between empirical distributions are biased
upward at small n, and the arms differ by 4x in row count.  Every arm is therefore subsampled to a
common row count (the minimum across arms) with a fixed seed before any distance is computed, so
no part of a gap is the estimator reacting to sample size.  The unmatched values are reported too.

GRID INVARIANCE.  Because these are marginals and not dynamics, the sampling grid does not matter,
so one real reference serves every arm regardless of its native rate.  `--npz-arms` reads
`gen_book` out of a `--dump-frames` dump so BookDiT can be scored with no decode at all.

Usage:
    book_native_facts.py --arms real=<dir> bookdit=<dir> lobs5=<dir> painting=<dir> \
        --npz-arms bookdit_native=<frames dir>
"""
from __future__ import annotations
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

TICK = 100.0            # LOBSTER prices are 1/10000 dollars; GOOG's tick is 100 of those units
WIN = re.compile(r"(GOOG_\d{4}-\d{2}-\d{2}_orderbook_real_id_\d+)")
# LOBSTER marks an empty level with +/-9999999999; the ask sentinel is positive, so a bare
# `px > 0` check would admit it as a price.  PX_MAX = 1e8 (= $10,000 in LOBSTER units) separates
# real prices from every sentinel convention seen here.
PX_MAX = 1e8


def _px_ok(px: np.ndarray) -> np.ndarray:
    """True where a price column holds a real quote rather than an empty-level sentinel."""
    return (px > 0) & (px < PX_MAX)


def files(root: Path) -> dict:
    out = {}
    for p in root.glob("*orderbook*.csv"):
        m = WIN.search(p.name)
        if m:
            out[m.group(1)] = p
    return out


def load(paths: list[Path]) -> np.ndarray:
    out = []
    for p in paths:
        a = pd.read_csv(p, header=None).values.astype(np.float64)
        if a.ndim == 2 and a.shape[1] >= 40:
            out.append(a[:, :40])
    return np.concatenate(out, axis=0) if out else np.zeros((0, 40))


def w1(x: np.ndarray, y: np.ndarray) -> float:
    """Wasserstein-1 between two empirical samples (equal-weight, via sorted quantiles)."""
    x, y = np.sort(x[np.isfinite(x)]), np.sort(y[np.isfinite(y)])
    if x.size == 0 or y.size == 0:
        return float("nan")
    q = np.linspace(0, 1, 2001)
    return float(np.mean(np.abs(np.quantile(x, q) - np.quantile(y, q))))


def fields(a: np.ndarray) -> dict:
    """The five clock-free quantities, each with its own validity mask."""
    ask_px, ask_sz, bid_px, bid_sz = a[:, 0::4], a[:, 1::4], a[:, 2::4], a[:, 3::4]
    ask_ok, bid_ok = _px_ok(ask_px), _px_ok(bid_px)
    touch = ask_ok[:, 0] & bid_ok[:, 0]
    full = ask_ok.all(1) & bid_ok.all(1) & (ask_sz > 0).all(1) & (bid_sz > 0).all(1)
    spread = (ask_px[touch, 0] - bid_px[touch, 0]) / TICK
    # Gaps use a PER-PAIR mask (both adjacent levels priced), not the all-ten-levels mask: an arm
    # that never fills all ten -- Painting fills 4.7 on average -- would otherwise return an empty
    # sample and a blank cell, which reads as "not measured" for the arm this most indicts.
    ga = np.where(ask_ok[:, 1:] & ask_ok[:, :-1],
                  (ask_px[:, 1:] - ask_px[:, :-1]) / TICK, np.nan).ravel()
    gb = np.where(bid_ok[:, :-1] & bid_ok[:, 1:],
                  (bid_px[:, :-1] - bid_px[:, 1:]) / TICK, np.nan).ravel()
    g = np.concatenate([ga, gb])
    gaps = g[np.isfinite(g)]
    gap_pairs = float(np.isfinite(g).mean())          # share of adjacent pairs that are measurable
    tot = bid_sz[touch, 0] + ask_sz[touch, 0]
    ok = tot > 0
    imb = (bid_sz[touch, 0][ok] - ask_sz[touch, 0][ok]) / tot[ok]
    # depth profile: share of the row's total volume at each level, averaged over rows.  Per-row
    # normalisation first, so a generator with the wrong volume SCALE is not penalised twice here
    # (LOB-Bench's volume statistics already carry the scale).
    vol = np.concatenate([ask_sz, bid_sz], axis=1)
    s = vol.sum(axis=1)
    prof = (vol[s > 0] / s[s > 0, None]).mean(axis=0)
    n_pop = (ask_ok & bid_ok & (ask_sz > 0) & (bid_sz > 0)).sum(axis=1)
    return {"spread": spread, "gaps": gaps, "imb": imb, "profile": prof,
            "gap_pairs": gap_pairs, "n_pop": n_pop.astype(float), "full_rate": float(full.mean()),
            "touch_rate": float(touch.mean()), "n_rows": len(a)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", nargs="+", required=True, metavar="NAME=DIR")
    ap.add_argument("--npz-arms", nargs="*", default=[], metavar="NAME=DIR",
                    help="arms stored as --dump-frames .npz files; scored from `gen_book`, i.e. "
                         "the native model output with no message decode in the path")
    ap.add_argument("--ref", default="real", help="arm every distance is measured against")
    ap.add_argument("--seed", type=int, default=20260805)
    ap.add_argument("--out", default="runs/lobbench/book_native")
    a = ap.parse_args()

    roots = {}
    for spec in a.arms:
        name, _, path = spec.partition("=")
        if not path:
            raise SystemExit(f"--arms entries must be name=path, got {spec!r}")
        roots[name] = Path(path)
        if not roots[name].is_dir():
            raise SystemExit(f"arm {name!r}: not a directory: {path}")
    if a.ref not in roots:
        raise SystemExit(f"reference arm {a.ref!r} not among {sorted(roots)}")

    npz = {}
    for spec in a.npz_arms:
        name, _, path = spec.partition("=")
        if not path or not Path(path).is_dir():
            raise SystemExit(f"--npz-arms entry {spec!r}: not a directory")
        npz[name] = Path(path)

    fmap = {k: files(v) for k, v in roots.items()}
    ids = sorted(set.intersection(*(set(v) for v in fmap.values())))
    print(f"[arms] {', '.join(roots)} | {len(ids)} common windows", flush=True)
    if not ids:
        raise SystemExit("no window id is present in every arm")

    raw = {k: load([fmap[k][i] for i in ids]) for k in roots}
    for k, d in npz.items():
        fs = sorted(d.glob("*.npz"))
        raw[k] = np.concatenate([np.load(f)["gen_book"] for f in fs]).astype(np.float64)[:, :40]
        print(f"    {k:16s} {len(raw[k]):7d} rows from {len(fs)} npz windows (NATIVE, no decode)")
    for k, v in raw.items():
        print(f"    {k:16s} {len(v):7d} rows ({len(v)/len(ids):6.1f} per window)")
    # SAMPLE-SIZE MATCHING: equalise row counts before any distance is computed.
    n = min(len(v) for v in raw.values())
    rng = np.random.default_rng(a.seed)
    sub = {k: v[np.sort(rng.choice(len(v), n, replace=False))] for k, v in raw.items()}
    print(f"    matched sample: {n} rows per arm")

    rows = []
    for tag, src in (("matched", sub), ("unmatched", raw)):
        F = {k: fields(v) for k, v in src.items()}
        R = F[a.ref]
        for k in raw:
            f = F[k]
            rows.append({
                "arm": k, "sample": tag, "n_rows": f["n_rows"],
                "spread_w1": w1(f["spread"], R["spread"]),
                "spread_med": float(np.median(f["spread"])) if f["spread"].size else np.nan,
                "gap_w1": w1(f["gaps"], R["gaps"]),
                "gap_med": float(np.median(f["gaps"])) if f["gaps"].size else np.nan,
                "gap_pairs": f["gap_pairs"],
                "imb_w1": w1(f["imb"], R["imb"]),
                "profile_l1": float(np.abs(f["profile"] - R["profile"]).sum() / 2),
                "levels_mean": float(f["n_pop"].mean()),
                "full_rate": f["full_rate"], "touch_rate": f["touch_rate"],
            })
    df = pd.DataFrame(rows)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "book_native_facts.csv", index=False)
    # the depth profiles themselves, for the figure
    prof = pd.DataFrame({k: fields(sub[k])["profile"] for k in raw})
    prof.index = [f"ask_L{i+1}" for i in range(10)] + [f"bid_L{i+1}" for i in range(10)]
    prof.to_csv(out / "depth_profile.csv")
    (out / "protocol.json").write_text(json.dumps(
        {"arms": {k: str(v) for k, v in roots.items()},
         "npz_arms": {k: str(v) for k, v in npz.items()}, "ref": a.ref, "n_windows": len(ids),
         "matched_rows_per_arm": int(n), "tick": TICK, "seed": a.seed,
         "note": "clock-free marginal book-state statistics; no IDM, no benchmark"}, indent=2))

    show = ["arm", "n_rows", "spread_w1", "spread_med", "gap_w1", "gap_med", "gap_pairs",
            "imb_w1", "profile_l1", "levels_mean", "full_rate"]
    for tag in ("matched", "unmatched"):
        print(f"\n=== {tag} sample (distances vs {a.ref}) ===")
        print(df[df["sample"] == tag][show].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
