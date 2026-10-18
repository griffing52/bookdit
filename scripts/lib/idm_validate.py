#!/usr/bin/env python
"""Validate the inverse-dynamics annotator against a real-vs-real message-metric floor.

Replay real GOOG -> run the IDM (scripts/lib/idm_annotator.py) -> recovered message stream. Score the
recovered stream against the true messages with LOB-Bench's own message metrics + distance funcs
(metrics.l1_by_group total-variation, metrics.wasserstein). The floor bounds what the inverse map
itself costs on LOB-Bench's message metrics.

Comparisons per metric:
  floor = true GOOG messages  vs  IDM-recovered-from-real-books
  (with --true-type: oracle event_type, synthetic order_id -> isolates order-id-linkage error)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from _paths import lob_bench_dir                          # noqa: E402

sys.path.insert(0, str(lob_bench_dir(required=True)))                  # $LOB_BENCH_DIR, default <repo>/libs/lob_bench

from idm_annotator import annotate                       # noqa: E402
import eval as lb_eval                                    # noqa: E402  libs/lob_bench/eval.py
import metrics as lb_metrics                              # noqa: E402  libs/lob_bench/metrics.py


def to_datetime(df, date):
    out = df.copy()
    out["time"] = pd.to_datetime(date) + pd.to_timedelta(out["time"], unit="s")
    for c in ("event_type", "order_id", "size", "price", "direction"):
        out[c] = out[c].astype("int64")
    return out


def dist(real_vals, gen_vals, nbins=50):
    """LOB-Bench l1_by_group (TV) + wasserstein between two 1-D samples."""
    real_vals = np.asarray(real_vals, float)
    gen_vals = np.asarray(gen_vals, float)
    real_vals = real_vals[np.isfinite(real_vals)]
    gen_vals = gen_vals[np.isfinite(gen_vals)]
    if len(real_vals) == 0 or len(gen_vals) == 0:
        return float("nan"), float("nan"), len(real_vals), len(gen_vals)
    pooled = np.concatenate([real_vals, gen_vals])
    lo, hi = np.percentile(pooled, 1), np.percentile(pooled, 99)
    if hi <= lo:
        hi = lo + 1e-6
    bins = np.linspace(lo, hi, nbins)
    df = pd.DataFrame({
        "score": np.concatenate([real_vals, gen_vals]),
        "group": np.concatenate([np.digitize(real_vals, bins), np.digitize(gen_vals, bins)]),
        "type": ["real"] * len(real_vals) + ["generated"] * len(gen_vals),
    })
    l1 = float(lb_metrics.l1_by_group(df, bootstrap_ci=False))
    w = float(lb_metrics.wasserstein(df, bootstrap_ci=False))
    return l1, w, len(real_vals), len(gen_vals)


def td_seconds(s):
    return pd.to_timedelta(s).dt.total_seconds().to_numpy()


def message_metric_values(msgs, book):
    """Return {metric_name: 1-D np.array of the statistic} for the 5 message/lifecycle metrics."""
    vals = {}
    vals["log_inter_arrival_time"] = np.log1p(lb_eval.inter_arrival_time(msgs).to_numpy())
    vals["log_time_to_cancel"] = np.log1p(td_seconds(lb_eval.time_to_cancel(msgs)))
    vals["log_time_to_first_fill"] = np.log1p(td_seconds(lb_eval.time_to_first_fill(msgs)))
    a_d, b_d = lb_eval.limit_order_depth(msgs, book)
    vals["limit_ask_order_depth"] = a_d.to_numpy()
    vals["limit_bid_order_depth"] = b_d.to_numpy()
    a_l, b_l = lb_eval.limit_order_levels(msgs, book)
    vals["limit_ask_order_levels"] = a_l.to_numpy()
    vals["limit_bid_order_levels"] = b_l.to_numpy()
    return vals


def report_coverage(stats):
    tc = stats["type_confusion"]
    print("\n== IDM coverage / loss sources ==")
    print(f"  recovered visible events : {stats['n_seen']:,}")
    print(f"  hidden/halt (type 5/6/7) : {stats['n_hidden']:,}  (invisible to book -> dropped)")
    print(f"  deep reductions (lvl>10) : {stats['n_deep']:,}  (outside 10-level world-model view)")
    names = {1: "new-LO", 2: "cancel", 3: "delete", 4: "exec"}
    print("\n  event_type confusion (rows=true, cols=recovered):")
    hdr = "        " + "".join(f"{names[c]:>9}" for c in (1, 2, 3, 4))
    print(hdr)
    for r in (1, 2, 3, 4):
        row = "".join(f"{tc[r, c]:>9,}" for c in (1, 2, 3, 4))
        print(f"  {names[r]:>6}{row}")
    # execution recall/precision (the crux)
    exec_true = tc[4, 1:5].sum()
    exec_rec = tc[1:5, 4].sum()
    exec_tp = tc[4, 4]
    if exec_true and exec_rec:
        print(f"\n  execution recall={exec_tp/exec_true:.3f}  precision={exec_tp/exec_rec:.3f}"
              f"  (true exec={exec_true:,}, recovered exec={exec_rec:,})")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--day", default="20240701")
    ap.add_argument("--ticker", default="GOOG")
    ap.add_argument("--max-events", type=int, default=300000)
    ap.add_argument("--true-type", action="store_true",
                    help="oracle event_type ablation (isolates order-id-linkage error from type confusion)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    date = f"{args.day[:4]}-{args.day[4:6]}-{args.day[6:]}"

    print(f"IDM validate: day={args.day} ticker={args.ticker} max_events={args.max_events:,} "
          f"true_type={args.true_type}")
    rec_msgs, rec_book, true_msgs, true_book, stats = annotate(
        args.day, args.ticker, max_events=args.max_events, use_true_type=args.true_type)
    report_coverage(stats)

    rec_m = to_datetime(rec_msgs, date)
    true_m = to_datetime(true_msgs, date)

    tv = message_metric_values(true_m, true_book)
    rv = message_metric_values(rec_m, rec_book)

    rows = []
    print(f"\n{'message metric':28} {'FLOOR L1(TV)':>13} {'FLOOR Wass':>11} {'n_true':>9} {'n_rec':>9}")
    for k in tv:
        l1, w, nr, ng = dist(tv[k], rv[k])
        rows.append({"metric": k, "l1": l1, "wasserstein": w, "n_true": nr, "n_rec": ng})
        print(f"{k:28} {l1:13.3f} {w:11.3f} {nr:9,} {ng:9,}")

    df = pd.DataFrame(rows)
    out = Path(args.out or f"runs/lobbench/idm_floor_{args.day}"
                           f"{'_truetype' if args.true_type else ''}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    clean = df[df.l1 < 0.15].metric.tolist()
    print(f"\nclean floors (L1<0.15): {clean}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
