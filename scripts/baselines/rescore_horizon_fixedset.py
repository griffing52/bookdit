#!/usr/bin/env python
"""Recompute an official-protocol horizon profile on the windows present in every block.

`horizon_official_protocol.py` writes the window count entering the level-ten filter, not the
count actually scored, and the surviving set changes block to block; holding the set fixed keeps
the drift slope free of composition changes.  This script reads the block trees already on disk
and only re-runs the scorer, so there is no regeneration and no GPU.

Usage:
    python scripts/baselines/rescore_horizon_fixedset.py \
        --run runs/horizon/bookdit --block 500 --out-name profile_fixedset.csv

The output lands in the run directory, where `horizon_matched_bands.py --validate` looks for it.
"""
from __future__ import annotations
import argparse
import re
import shutil
import sys
from functools import reduce
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BOOK10_STATS = (
    "spread", "orderbook_imbalance", "ask_volume_touch", "bid_volume_touch",
    "ask_volume", "bid_volume", "ofi", "ofi_up", "ofi_stay", "ofi_down",
)
sys.path.insert(0, str(ROOT / "scripts" / "lib"))
from _paths import lob_bench_dir  # noqa: E402
sys.path.insert(0, str(lob_bench_dir(required=True)))   # $LOB_BENCH_DIR relocates the clone
import data_loading, scoring                                            # noqa: E402
from run_bench import DEFAULT_SCORING_CONFIG, DEFAULT_METRICS           # noqa: E402

RID_GEN = re.compile(r"real_id_(\d+)_gen")
RID_ANY = re.compile(r"real_id_(\d+)")


def block_dirs(run: Path):
    ds = sorted(run.glob("block_*"), key=lambda p: int(p.name.split("_")[1]))
    if not ds:
        raise SystemExit(f"no block_* directories under {run}")
    return ds


def rids_of(bd: Path) -> set[str]:
    return {RID_GEN.search(f.name).group(1) for f in (bd / "data_gen").glob("*message*")}

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--intersect-with", nargs="*", default=[],
                    help="peer horizon runs whose every-block IDs also constrain the fixed set")
    ap.add_argument("--run", required=True, help="a horizon_official_protocol.py --out directory")
    ap.add_argument("--block", type=int, default=500, help="messages per block (for the x axis)")
    ap.add_argument("--out-name", default="profile_fixedset.csv")
    ap.add_argument("--work", default=None, help="scratch dir for the subset trees")
    a = ap.parse_args()

    run = Path(a.run)
    bds = block_dirs(run)
    per_block = {bd.name: rids_of(bd) for bd in bds}
    all_sets = list(per_block.values())
    for peer_name in a.intersect_with:
        peer = Path(peer_name)
        peer_blocks = block_dirs(peer)
        all_sets.extend(rids_of(bd) for bd in peer_blocks)
    keep = reduce(lambda x, y: x & y, all_sets)
    counts = [len(per_block[bd.name]) for bd in bds]
    print(f"[fixedset] {len(bds)} blocks | scored per block: min {min(counts)} max {max(counts)} "
          f"| union {len(reduce(lambda x, y: x | y, per_block.values()))} "
          f"| INTERSECTION {len(keep)}", flush=True)
    if not keep:
        raise SystemExit("empty intersection -- nothing to score")

    work = Path(a.work) if a.work else run / "_fixedset_work"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)

    rows = []
    for bi, bd in enumerate(bds):
        sub = work / bd.name
        for d in ("data_real", "data_gen", "data_cond"):
            (sub / d).mkdir(parents=True)
            for f in (bd / d).glob("*.csv"):
                m = RID_ANY.search(f.name)
                if m and m.group(1) in keep:
                    # symlink: the scorer only reads, and this keeps the copy free
                    (sub / d / f.name).symlink_to(f.resolve())
        loader = data_loading.Simple_Loader(str(sub / "data_real"), str(sub / "data_gen"),
                                            str(sub / "data_cond"))
        for s in loader:
            s.materialize()
        sc, _, _ = scoring.run_benchmark(loader, DEFAULT_SCORING_CONFIG,
                                         default_metric=DEFAULT_METRICS)
        per = {k: float(v["l1"][0]) for k, v in sc.items()}
        l1 = float(np.mean(list(per.values())))
        # L1^18 excludes the three message-clock statistics (inter-arrival time, time-to-cancel,
        # volume-per-minute), so the horizon curve can be read on the SAME three views Table 1
        # reports rather than on L1^21 alone.
        drop18 = {"log_inter_arrival_time", "log_time_to_cancel", "vol_per_min"}
        sub18 = [v for k, v in per.items() if k not in drop18]
        l1_18 = float(np.mean(sub18)) if sub18 else float("nan")
        book10 = float(np.mean([per[name] for name in BOOK10_STATS]))
        n = len(list((sub / "data_gen").glob("*message*")))
        rows.append({"horizon": (bi + 1) * a.block, "l1": l1, "l1_18": l1_18, "book_10": book10,
                     "n_stats": len(per), "n_windows": n,
                     **{f"stat_{k}": v for k, v in per.items()}})
        print(f"  h={(bi + 1) * a.block:6d}  L1_21={l1:.4f}  L1_18={l1_18:.4f}  "
              f"({n} windows, {len(per)} stats)", flush=True)
        pd.DataFrame(rows).to_csv(run / a.out_name, index=False)

    d = pd.DataFrame(rows)
    from scipy import stats as sps
    r = sps.linregress(d.horizon / 1000.0, d.l1)
    print(f"\n[fixedset] {d.l1.iloc[0]:.4f} -> {d.l1.iloc[-1]:.4f} on a FIXED {len(keep)} windows | "
          f"slope {r.slope:+.5f} L1/1k events, p={r.pvalue:.3f}")
    print(f"[fixedset] wrote {run / a.out_name}")


if __name__ == "__main__":
    main()
