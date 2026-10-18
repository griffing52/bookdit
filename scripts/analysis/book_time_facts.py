#!/usr/bin/env python
"""Time-gridded book dynamics: kurtosis, variance ratio, signature plot, autocorrelations.

Companion to `book_native_facts.py`, which carries the statistics that need no clock. Everything
here references consecutive states, and the arms do not share an event clock, so each arm's book
is resampled onto a grid before anything is computed.

Two grids, both reported:

  `wall`  fixed real-time step from each window's start, so an arm is charged for generating the
          wrong amount of elapsed time.
  `norm`  a fixed number of points spread across each window's own span, which divides out the
          duration error and measures the shape of the dynamics.

`span_ratio` (generated duration / real duration, per window) is reported separately, so the clock
error is stated once in seconds rather than smeared across the other rows.

Resampling is last-observation-carried-forward, which is what an order book is: a step function
between events. Grid points before an arm's first message are dropped.

Painting the Market is excluded: its message timestamps are set by the evaluation harness rather
than by the model, so any time-referencing statistic would measure the interpolation. Its
clock-free statistics in `book_native_facts.py` are unaffected.

Usage:
    book_time_facts.py --arms real=<dir> bookdit=<dir> lobs5_released=<dir> --grid norm
"""
from __future__ import annotations
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

WIN = re.compile(r"(GOOG_\d{4}-\d{2}-\d{2})_orderbook_(real_id_\d+)")
QS = (1, 2, 5, 10, 20)
LAGS = (1, 2, 5, 10, 20)


def pairs(root: Path) -> dict:
    """window id -> (orderbook path, message path); both are required and row-aligned."""
    out = {}
    for p in root.glob("*orderbook*.csv"):
        m = WIN.search(p.name)
        if not m:
            continue
        msg = p.with_name(p.name.replace("orderbook", "message"))
        if msg.exists():
            out[f"{m.group(1)}_{m.group(2)}"] = (p, msg)
    return out


def mid_series(ob: Path, msg: Path):
    b = pd.read_csv(ob, header=None).values.astype(np.float64)[:, :40]
    t = pd.read_csv(msg, header=None).values[:, 0].astype(np.float64)
    n = min(len(b), len(t))
    b, t = b[:n], t[:n]
    ok = (b[:, 0] > 0) & (b[:, 2] > 0)
    return t[ok], 0.5 * (b[ok, 0] + b[ok, 2])


def regrid(t: np.ndarray, y: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Last observation carried forward; NaN before the first observation."""
    idx = np.searchsorted(t, grid, side="right") - 1
    out = np.where(idx >= 0, y[np.clip(idx, 0, len(y) - 1)], np.nan)
    return out


def stats_from_returns(R: list[np.ndarray]) -> dict:
    """R = one log-return series per window, all on the same grid step."""
    out = {}
    for q in QS:
        agg = [r.reshape(-1, q).sum(axis=1) for r in R if len(r) >= q and len(r) // q >= 2
               for r in [r[:len(r) // q * q]]]
        v = np.concatenate(agg) if agg else np.zeros(0)
        v = v[np.isfinite(v)]
        if v.size < 8 or v.std() == 0:
            out[f"kurt_q{q}"], out[f"rv_q{q}"] = np.nan, np.nan
            continue
        out[f"kurt_q{q}"] = float(np.mean((v - v.mean()) ** 4) / v.std() ** 4 - 3)
        out[f"rv_q{q}"] = float(np.mean(v ** 2) / q)          # signature plot: RV per unit step
    v1 = np.concatenate([r for r in R if len(r)])
    v1 = v1[np.isfinite(v1)]
    base = np.var(v1) if v1.size else np.nan
    for q in QS[1:]:
        num = out.get(f"rv_q{q}")
        out[f"vr_q{q}"] = float(num / base) if (num and np.isfinite(base) and base > 0) else np.nan
    for lag in LAGS:
        a, b = [], []
        for r in R:
            r = r[np.isfinite(r)]
            if len(r) <= lag + 2:
                continue
            x = r - r.mean()
            d = x @ x
            if d > 0:
                a.append(x[:-lag] @ x[lag:] / d)
            z = np.abs(r) - np.abs(r).mean()
            dz = z @ z
            if dz > 0:
                b.append(z[:-lag] @ z[lag:] / dz)
        out[f"racf_l{lag}"] = float(np.mean(a)) if a else np.nan
        out[f"absacf_l{lag}"] = float(np.mean(b)) if b else np.nan
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", nargs="+", required=True, metavar="NAME=DIR")
    ap.add_argument("--ref", default="real")
    ap.add_argument("--grid", choices=["norm", "wall", "both"], default="both")
    ap.add_argument("--points", type=int, default=64, help="grid points per window (norm)")
    ap.add_argument("--step", type=float, default=0.05, help="grid step in seconds (wall)")
    ap.add_argument("--out", default="runs/lobbench/book_time")
    a = ap.parse_args()

    roots = {}
    for spec in a.arms:
        name, _, path = spec.partition("=")
        if not path or not Path(path).is_dir():
            raise SystemExit(f"--arms entry {spec!r}: not a directory")
        roots[name] = Path(path)
    if a.ref not in roots:
        raise SystemExit(f"reference {a.ref!r} not among {sorted(roots)}")

    pmap = {k: pairs(v) for k, v in roots.items()}
    ids = sorted(set.intersection(*(set(v) for v in pmap.values())))
    print(f"[arms] {', '.join(roots)} | {len(ids)} common windows", flush=True)

    series = {k: {} for k in roots}
    for k in roots:
        for i in ids:
            t, m = mid_series(*pmap[k][i])
            if len(t) >= 8:
                series[k][i] = (t, m)
    usable = sorted(set.intersection(*(set(v) for v in series.values())))
    print(f"    {len(usable)} windows usable in every arm", flush=True)

    # ---- the clock error itself, stated once
    span = {k: np.array([series[k][i][0][-1] - series[k][i][0][0] for i in usable]) for k in roots}
    ref_span = span[a.ref]
    clock = []
    for k in roots:
        ok = ref_span > 0
        n_ev = np.array([len(series[k][i][0]) for i in usable])
        clock.append({"arm": k,
                      "median_span_s": float(np.median(span[k])),
                      "median_span_ratio": float(np.median(span[k][ok] / ref_span[ok])),
                      "median_events": float(np.median(n_ev)),
                      "events_per_s": float(np.median(n_ev[ok] / np.maximum(span[k][ok], 1e-9)))})
    cdf = pd.DataFrame(clock)
    print("\n=== event clock (no grid involved) ===")
    print(cdf.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    rows = []
    grids = ["norm", "wall"] if a.grid == "both" else [a.grid]
    for g in grids:
        R = {k: [] for k in roots}
        for i in usable:
            for k in roots:
                t, m = series[k][i]
                if g == "norm":
                    grid = np.linspace(t[0], t[-1], a.points)
                else:
                    t0 = series[a.ref][i][0][0]
                    t1 = series[a.ref][i][0][-1]
                    grid = np.arange(t0, t1 + 1e-12, a.step)
                if len(grid) < 8:
                    continue
                y = regrid(t, m, grid)
                y = y[np.isfinite(y) & (y > 0)]
                if len(y) >= 8:
                    R[k].append(np.diff(np.log(y)))
        for k in roots:
            rows.append({"arm": k, "grid": g, "n_windows": len(R[k]),
                         "pts_per_window": float(np.mean([len(r) + 1 for r in R[k]])) if R[k] else np.nan,
                         **stats_from_returns(R[k])})
    df = pd.DataFrame(rows)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "book_time_facts.csv", index=False)
    cdf.to_csv(out / "event_clock.csv", index=False)
    (out / "protocol.json").write_text(json.dumps(
        {"arms": {k: str(v) for k, v in roots.items()}, "ref": a.ref, "grid": a.grid,
         "points": a.points, "step_s": a.step, "n_windows": len(usable),
         "excluded": {"painting": "timestamps are harness-generated uniform spacing, not model "
                                  "output (26 distinct gaps over 1,411 messages)"}}, indent=2))
    for g in grids:
        d = df[df.grid == g]
        print(f"\n=== grid={g} ===")
        print(d[["arm", "pts_per_window"] + [f"kurt_q{q}" for q in QS]]
              .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        print(d[["arm"] + [f"vr_q{q}" for q in QS[1:]] + ["absacf_l1", "absacf_l5", "racf_l1"]]
              .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
