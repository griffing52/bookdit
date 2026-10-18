#!/usr/bin/env python
"""Drift slope of each LOB-Bench divergence curve, with a bootstrap CI.

l1_by_group ships 100 bootstrap replicates alongside every point. For replicate b we rebuild the
headline curve (mean over the 16 well-defined statistics) and refit the slope, which gives a
bootstrap distribution for "L1 per 1000 generated messages".
"""
from __future__ import annotations
import argparse, glob, gzip, os, pickle, re
from pathlib import Path
import numpy as np, pandas as pd

DEGENERATE = {"ofi", "ofi_up", "ofi_stay", "ofi_down", "vol_per_min"}
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs/divergence"      # where run_divergence_official.sh writes


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default=str(OUT / "results*" / "scores"))
    ap.add_argument("--data", default=str(OUT / "data"))
    ap.add_argument("--min-full", type=float, default=0.98,
                    help="keep a horizon group only if this fraction of windows supplies a FULL "
                         "sub-sequence there; the ragged tail otherwise fakes a rising curve")
    ap.add_argument("--out", default=str(OUT / "slopes.csv"))
    ap.add_argument("--match", nargs="*",
                    default=["bookdit", "lobs5", "realgen"],
                    help="arms that must be compared to each other: at each divergence_horizon "
                         "every one of them is truncated to the SHALLOWEST depth any of them "
                         "reaches. Without this each arm is fitted over its own depth and the "
                         "slope ratio compares fits over different spans. '' to disable.")
    ap.add_argument("--headline", default=str(OUT / "headline.csv"))
    ap.add_argument("--floor-arm", default="realgen",
                    help="arm used as the real-vs-real control the excess is measured against")
    ap.add_argument("--bookdit-arm", default="bookdit",
                    help="BookDiT arm of the printed LOB-S5 : BookDiT excess ratio; must be in --match")
    ap.add_argument("--lobs5-arm", default="lobs5",
                    help="LOB-S5 arm of the printed ratio; must be in --match")
    a = ap.parse_args()

    # ---- pass 1: per (model, H), the full bootstrap curve and the depth the arm reaches alone
    cells = {}
    for p in sorted(glob.glob(f"{a.results}/scores_div_GOOG_*.pkl")):
        b = os.path.basename(p)
        if "_REAL_" in b:
            continue
        m = re.match(r"scores_div_GOOG_(.+)_(\d+)_\d{8}_\d{6}\.pkl", b)
        model, H = m.group(1), int(m.group(2))
        try:
            with gzip.open(p, "rb") as f:
                scores, _ = pickle.load(f)
        except EOFError:
            continue
        stats = [s for s in scores if s not in DEGENERATE]
        n_avail = min(len(scores[s]) for s in stats)
        lens = np.array([sum(1 for _ in open(f, "rb")) for f in
                         glob.glob(f"{a.data}/{model}/GOOG/data_gen/*message*")])
        frac_full = np.array([(lens >= (g + 1) * H).mean() for g in range(n_avail)])
        own = int(np.argmax(frac_full < a.min_full)) if (frac_full < a.min_full).any() else n_avail
        # curve[b, g] = mean over statistics of bootstrap replicate b at horizon group g
        curve = np.stack([np.stack([scores[s][g][2] for s in stats]).mean(0)
                          for g in range(n_avail)], axis=1)       # (n_boot + 1, n_avail)
        cells[(model, H)] = dict(curve=curve, frac_full=frac_full, own_depth=own)

    # ---- matching: arms that are compared to each other must be fitted over the SAME span.
    # Each arm's own depth is set by the left tail of ITS OWN per-window generated message counts,
    # so without this the ratio compares a fit over 0-300 msgs with a fit over 0-400 msgs.
    match = set(a.match or [])
    depth = {k: v["own_depth"] for k, v in cells.items()}
    if match:
        for H in sorted({h for _, h in cells}):
            present = [k for k in cells if k[0] in match and k[1] == H]
            if len(present) < 2:
                continue
            common = min(depth[k] for k in present)
            for k in present:
                if depth[k] != common:
                    print(f"[match] H={H} {k[0]}: depth {depth[k]} -> {common} groups "
                          f"({common * H} msgs), capped by "
                          f"{min(present, key=lambda q: depth[q])[0]}")
                depth[k] = common

    rows = []
    for (model, H), c in cells.items():
        n_g = depth[(model, H)]
        if n_g < 3:
            continue
        curve, frac_full = c["curve"][:, :n_g], c["frac_full"]
        n_b = curve.shape[0]
        x = (np.arange(n_g) + 1) * H / 1000.0                    # thousands of messages
        point = curve[0]
        slope_b = np.array([np.polyfit(x, curve[i], 1)[0] for i in range(n_b)])
        s0 = np.polyfit(x, point, 1)[0]
        lo, hi = np.percentile(slope_b, [2.5, 97.5])
        rows.append({
            "model": model, "divergence_horizon": H, "n_groups": n_g,
            "span_msgs": int(x[-1] * 1000), "l1_first": point[0], "l1_last": point[-1],
            "delta": point[-1] - point[0],
            "slope_per_1k": s0, "slope_lo": lo, "slope_hi": hi,
            "min_frac_full": float(frac_full[:n_g].min()),
            "own_depth": c["own_depth"], "matched": model in match,
            "sig": "yes" if lo > 0 or hi < 0 else "no",
            "n_boot": n_b,
        })
    df = pd.DataFrame(rows).sort_values(["divergence_horizon", "model"])
    df.to_csv(a.out, index=False)
    with pd.option_context("display.width", 240):
        print(df.round(5).to_string(index=False))
    print(f"\n-> {a.out}")

    # ---- headline: excess drift over the real-vs-real control, on the matched span
    # Display labels for the arms setup_divergence_official.py builds; any other model id is
    # reported under its own name.
    ARM = {"bookdit": "BookDiT", "lobs5": "LOB-S5 (released samples)",
           "realgen": "real-vs-real floor", "ditfloor": "IDM decode floor (real books)"}
    hrows = []
    for H, grp in df.groupby("divergence_horizon"):
        fl = grp[grp.model == a.floor_arm]
        if not len(fl):
            print(f"  H={H:4d}: no {a.floor_arm!r} row, so no excess slopes at this horizon")
            continue
        f0, fd = float(fl.slope_per_1k.iloc[0]), float(fl.delta.iloc[0])
        for _, r in grp.iterrows():
            hrows.append({"arm": ARM.get(r.model, r.model), **r.to_dict(),
                          "floor_slope_per_1k": f0,
                          "excess_slope_per_1k": r.slope_per_1k - f0,
                          "floor_delta": fd, "excess_delta": r.delta - fd})
    hd = pd.DataFrame(hrows)
    cols = ["arm", "model", "divergence_horizon", "n_groups", "span_msgs", "l1_first", "l1_last",
            "delta", "slope_per_1k", "slope_lo", "slope_hi", "sig", "floor_slope_per_1k",
            "excess_slope_per_1k", "floor_delta", "excess_delta", "min_frac_full", "own_depth",
            "matched", "n_boot"]
    hd = hd[cols].sort_values(["divergence_horizon", "arm"])
    hd.to_csv(a.headline, index=False)
    print(f"-> {a.headline}")
    for H, g in hd.groupby("divergence_horizon"):
        try:
            # both arms must be in --match: arms outside it are fitted at their own depth, so
            # quoting one here would silently compare different spans.
            b = g[g.model == a.bookdit_arm].iloc[0]
            l = g[g.model == a.lobs5_arm].iloc[0]
        except IndexError:
            print(f"  H={H:4d} SKIPPED: no {a.bookdit_arm!r} or {a.lobs5_arm!r} row")
            continue
        if not (bool(b.matched) and bool(l.matched)):
            print(f"  H={H:4d} SKIPPED: arms not both matched "
                  f"(BookDiT {b.matched}, LOB-S5 {l.matched})")
            continue
        print(f"  H={H:4d} span {int(b.span_msgs)} msgs | excess slope BookDiT "
              f"{b.excess_slope_per_1k:+.5f} vs LOB-S5 {l.excess_slope_per_1k:+.5f} "
              f"= {l.excess_slope_per_1k / b.excess_slope_per_1k:.2f}x | excess cumulative "
              f"{b.excess_delta:+.5f} vs {l.excess_delta:+.5f} "
              f"= {l.excess_delta / b.excess_delta:.2f}x")


if __name__ == "__main__":
    main()
