#!/usr/bin/env python
"""Turn run_bench.py --div_only pickles into tidy drift curves.

Each pickle holds {score_name: [ (l1, ci[2], bootstrap[101]) per prediction-horizon group ]}.
The *_REAL_* pickle is the --div_error_bounds real-vs-real bound for the same grouping.
"""
from __future__ import annotations
import argparse, glob, gzip, os, pickle, re
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs/divergence"      # where run_divergence_official.sh writes

# eval.orderflow_imbalance takes a 100-message rolling mean, so a sub-sequence of <=100 messages
# yields zero usable values and l1_by_group then returns a FALSE 0.0. vol_per_min is binned by
# minute and is just as unusable at short horizons -- its real-vs-real floor is 0.33. Both are
# excluded from the headline mean; `l1_mean` over all 21 keeps them for reference.
DEGENERATE = {"ofi", "ofi_up", "ofi_stay", "ofi_down", "vol_per_min"}


def load(p):
    with gzip.open(p, "rb") as f:
        return pickle.load(f)


def n_windows_per_group(data_dir: Path, horizon: int, n_groups: int) -> list[int]:
    """_split_df chunks by row count, so group k exists iff the gen file has > k*horizon rows."""
    lens = []
    for f in glob.glob(str(data_dir / "GOOG" / "data_gen" / "*message*")):
        with open(f, "rb") as fh:
            lens.append(sum(1 for _ in fh))
    lens = np.array(lens)
    return [int((lens > k * horizon).sum()) for k in range(n_groups)]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default=str(OUT / "results*" / "scores"))
    ap.add_argument("--data", default=str(OUT / "data"))
    ap.add_argument("--out", default=str(OUT / "curves.csv"))
    a = ap.parse_args()

    rows = []
    bounds = {}
    for p in sorted(glob.glob(f"{a.results}/scores_div_GOOG_REAL_*.pkl")):
        H = int(re.search(r"REAL_(\d+)_", os.path.basename(p)).group(1))
        d, _ = load(p)
        bounds[H] = d

    for p in sorted(glob.glob(f"{a.results}/scores_div_GOOG_*.pkl")):
        b = os.path.basename(p)
        if "_REAL_" in b:
            continue
        m = re.match(r"scores_div_GOOG_(.+)_(\d+)_\d{8}_\d{6}\.pkl", b)
        model, H = m.group(1), int(m.group(2))
        try:
            scores, _ = load(p)
        except EOFError:                    # still being written by a live run
            print(f"[skip, incomplete] {b}")
            continue
        n_groups = max(len(v) for v in scores.values())
        nw = n_windows_per_group(Path(a.data) / model, H, n_groups)
        bd = bounds.get(H)
        for stat, series in scores.items():
            for k, (l1, ci, boot) in enumerate(series):
                r = {"model": model, "divergence_horizon": H, "group": k,
                     "horizon_msgs": (k + 1) * H, "statistic": stat,
                     "l1": float(l1), "ci_lo": float(ci[0]), "ci_hi": float(ci[1]),
                     "n_windows": nw[k] if k < len(nw) else np.nan}
                if bd is not None and stat in bd and k < len(bd[stat]):
                    r["real_bound_l1"] = float(bd[stat][k][0])
                    r["real_bound_hi"] = float(bd[stat][k][1][1])
                rows.append(r)

    df = pd.DataFrame(rows)
    df["degenerate"] = df.statistic.isin(DEGENERATE)
    df.to_csv(a.out, index=False)

    key = ["model", "divergence_horizon", "group", "horizon_msgs"]
    agg = (df.groupby(key)
             .agg(l1_mean=("l1", "mean"), real_bound=("real_bound_hi", "mean"),
                  n_stats=("l1", "size"), n_windows=("n_windows", "max"))
             .reset_index())
    clean = (df[~df.degenerate].groupby(key)
               .agg(l1_mean_clean=("l1", "mean"), n_clean=("l1", "size")).reset_index())
    agg = agg.merge(clean, on=key, how="left")
    agg.to_csv(a.out.replace(".csv", "_agg.csv"), index=False)
    print(f"-> {a.out}\n-> {a.out.replace('.csv', '_agg.csv')}")
    with pd.option_context("display.width", 200, "display.max_rows", 300):
        print(agg.to_string(index=False))


if __name__ == "__main__":
    main()
