#!/usr/bin/env python
"""Per-horizon bootstrap bands for the LOB-Bench divergence curves (official windows).

`summarize_divergence_official.py` keeps only the point estimate per statistic; the raw pickles
also carry the 100 bootstrap replicates that `slopes_divergence_official.py` uses to build its
slope CI. This script re-reads those pickles once and persists the *curve-level* band so the
figure script never has to touch the 150 MB pickles again.

For each (model, divergence_horizon):
    curve[b, g] = mean over the 16 non-degenerate statistics of bootstrap replicate b at group g
    b = 0 is the whole-data loss (the point estimate); b = 1..100 are the resamples.
    l1_*    = band on the absolute curve
    rel_*   = band on the *paired* drift curve  curve[b, g] - curve[b, 0]
              (paired within a replicate, so it is the band on the quantity the slope measures)

Groups are truncated exactly as the slope fit truncates them: `n_groups` from slopes.csv, which
is the last group where >= 98% of generated files still supply a full sub-sequence.

Output: runs/divergence/curve_bands.csv (read by scripts/figures/fig_divergence_drift.py)
"""
from __future__ import annotations
import argparse
import glob
import gzip
import os
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs/divergence"
# same exclusion list as summarize_divergence_official.py: these are undefined at short horizons
DEGENERATE = {"ofi", "ofi_up", "ofi_stay", "ofi_down", "vol_per_min"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", nargs="*", default=[str(OUT / "results*" / "scores")])
    ap.add_argument("--slopes", default=str(OUT / "slopes.csv"))
    ap.add_argument("--out", default=str(OUT / "curve_bands.csv"))
    a = ap.parse_args()

    slopes = pd.read_csv(a.slopes).set_index(["model", "divergence_horizon"]).n_groups

    rows = []
    files = sorted({f for pat in a.results for f in glob.glob(f"{pat}/scores_div_GOOG_*.pkl")})
    for p in files:
        b = os.path.basename(p)
        if "_REAL_" in b:
            continue
        m = re.match(r"scores_div_GOOG_(.+)_(\d+)_\d{8}_\d{6}\.pkl", b)
        model, H = m.group(1), int(m.group(2))
        if (model, H) not in slopes.index:
            continue
        n_g = int(slopes.loc[(model, H)])
        try:
            with gzip.open(p, "rb") as f:
                scores, _ = pickle.load(f)
        except EOFError:
            print(f"[skip, incomplete] {b}")
            continue
        stats = [s for s in scores if s not in DEGENERATE]
        curve = np.stack([np.stack([scores[s][g][2] for s in stats]).mean(0)
                          for g in range(n_g)], axis=1)          # (n_boot + 1, n_groups)
        rel = curve - curve[:, [0]]
        for g in range(n_g):
            rows.append({
                "model": model, "divergence_horizon": H, "group": g,
                "horizon_msgs": (g + 1) * H, "n_stats": len(stats),
                "l1_point": float(curve[0, g]),
                "l1_lo": float(np.percentile(curve[1:, g], 2.5)),
                "l1_hi": float(np.percentile(curve[1:, g], 97.5)),
                "rel_point": float(rel[0, g]),
                "rel_lo": float(np.percentile(rel[1:, g], 2.5)),
                "rel_hi": float(np.percentile(rel[1:, g], 97.5)),
                "n_boot": int(curve.shape[0] - 1),
            })
        print(f"[ok] {model} H={H} groups={n_g} stats={len(stats)}")

    df = pd.DataFrame(rows).sort_values(["divergence_horizon", "model", "group"])
    df.to_csv(a.out, index=False)
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
