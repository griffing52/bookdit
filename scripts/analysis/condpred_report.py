#!/usr/bin/env python
"""Render conditional_prediction.py JSON output as markdown transfer matrices.

Besides the per-target matrices it prints the "R^2 carried to real" aggregate the paper's
Table 3 (tab:split) reports: for each training source X, the design-A X->real R^2 averaged over
the targets {spread, dimb, rvol} and every horizon in the run (10 and 20 frames for the paper's
runs).  The Table 3 sub-rows ("on spread", "on depth imbalance") are the per-target components
of that mean, and the real->X numbers quoted in the transfer discussion use the same average.

Usage
    python scripts/analysis/condpred_report.py runs/condpred/condpred_released_h1020.json
    python scripts/analysis/condpred_report.py runs/condpred/*.json --aggregate-only
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

AGG_TARGETS = ("spread", "dimb", "rvol")


def table(m: dict, sources: list[str], ref: str) -> list[str]:
    c = m["cells"]
    hdr = "| train \\ test | " + " | ".join(f"{s}" for s in sources) + " |"
    sep = "|" + "---|" * (len(sources) + 1)
    out = [hdr, sep]
    for a in sources:
        row = [f"| {a} "]
        for b in sources:
            row.append(f"| {c[f'{a}->{b}']['r2']:+.3f} ")
        out.append("".join(row) + "|")
    r0 = c[ref]["r2"]
    gaps = "  ".join(f"{k}: {r0 - v['r2']:+.3f}" for k, v in c.items() if k != ref)
    out.append("")
    out.append(f"gaps vs {ref} (R^2_ref - R^2_cell): {gaps}")
    return out


def aggregate(d: dict, design: str = "A", targets=AGG_TARGETS, ref: str = "real",
              direction: str = "to") -> dict[str, dict[str, float]]:
    """Mean R^2 over `targets` x the run's horizons, for every source X.

    direction="to"   averages the X->ref cells (the "carried to real" row);
    direction="from" averages the ref->X cells (a real-trained predictor evaluated on X).
    Returns {X: {"mean": ..., <target>: per-target mean over horizons}}."""
    res = d["results"]
    out = {}
    for x in d["sources"]:
        cell = f"{x}->{ref}" if direction == "to" else f"{ref}->{x}"
        per = {t: float(np.mean([res[f"{design}|h{h}|{t}"]["cells"][cell]["r2"]
                                 for h in d["horizons"]])) for t in targets}
        out[x] = {"mean": float(np.mean(list(per.values()))), **per}
    return out


def print_aggregate(d: dict) -> None:
    hs = ",".join(str(h) for h in d["horizons"])
    for direction, title in (("to", "R^2 carried to real (X -> real)"),
                             ("from", "real-trained predictor on X (real -> X)")):
        agg = aggregate(d, direction=direction)
        print(f"\n### {title}; design A, mean over {{{', '.join(AGG_TARGETS)}}} x h in {{{hs}}}")
        print("| X | mean | " + " | ".join(AGG_TARGETS) + " |")
        print("|" + "---|" * (len(AGG_TARGETS) + 2))
        for x, v in agg.items():
            print(f"| {x} | {v['mean']:+.3f} | "
                  + " | ".join(f"{v[t]:+.3f}" for t in AGG_TARGETS) + " |")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("json", nargs="+", help="conditional_prediction.py output file(s)")
    ap.add_argument("--aggregate-only", action="store_true",
                    help="print only the Table 3 aggregate, not every per-target matrix")
    a = ap.parse_args()
    for p in a.json:
        d = json.loads(Path(p).read_text())
        S = d["sources"]
        ref = f"{S[0]}->{S[0]}"
        print(f"\n## {Path(p).stem}")
        print(f"n_windows={d['n_windows']}  n_days={d['n_days']}  seen={d['n_files_seen']}  "
              f"snap={d['snap']}  ctxW={d['ctxW']}  selfW={d['selfW']}  feat_dim={d['feat_dim']}")
        print_aggregate(d)
        if a.aggregate_only:
            continue
        for k, m in d["results"].items():
            print(f"\n### {k}")
            print("\n".join(table(m, S, ref)))
            extra = []
            for name, cell in m["cells"].items():
                s = f"{name}: y_sd={cell['y_sd']:.4f} corr={cell['corr']:+.3f}"
                if cell["n_nonzero"]:
                    s += f" signacc={cell['sign_acc']:.3f}/base={cell['sign_base']:.3f}"
                extra.append(s)
            print("\n" + "\n".join("  " + e for e in extra))


if __name__ == "__main__":
    main()
