#!/usr/bin/env python
"""Score generated arms on the official LOB-Bench lane.

Takes one or more raw generation directories under `--gen-root` (one per arm, e.g. the
`gen_bookdit_on_lobbench_windows.py --out` trees), applies the LOB-Bench authors'
`cut_data_to_lvl(..., 10)` to each, pairs every surviving window with the same
`data_real`/`data_cond` files every other official arm uses, and scores the result with
LOB-Bench's own `scoring.run_benchmark`.

Reports the three aggregate views for each arm -- L1^21, L1^18, book-10 -- plus the full
per-statistic table.

Window set.  By default arms are scored on the intersection of the windows all arms retain, so a
difference between two rows is a difference in the models and not in which windows survived the
filter; the per-arm pre-intersection counts are printed as well.  `--no-intersect` scores each arm
on its own surviving windows instead.

Usage (Table 1's no-refinement and decode-floor rows):
    python scripts/baselines/score_official.py --gen-root $BOOKDIT_WORK/tab1 \
        --arms raw floor --no-intersect --out runs/lobbench/tab1
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "lib"))
from _paths import lob_bench_dir  # noqa: E402
sys.path.insert(0, str(lob_bench_dir(required=True)))   # $LOB_BENCH_DIR relocates the clone
from _paths import bench_work                                             # noqa: E402
import data_loading, scoring                                              # noqa: E402
from run_bench import DEFAULT_SCORING_CONFIG, DEFAULT_METRICS             # noqa: E402

CLOCK_STATS = {"log_inter_arrival_time", "log_time_to_cancel", "vol_per_min"}
BOOK10_STATS = ("spread", "orderbook_imbalance", "ask_volume_touch", "bid_volume_touch",
                "ask_volume", "bid_volume", "ofi", "ofi_up", "ofi_stay", "ofi_down")
RE_GEN = re.compile(r"GOOG_(\d{4}-\d{2}-\d{2})_message_real_id_(\d+)_gen_id_0\.csv$")


def ref_dir() -> Path:
    """The shared real/cond side every official arm is scored against, under `$BOOKDIT_WORK`.
    Resolved on use so `--help` needs no work root."""
    return bench_work() / "lobbench_repro/lob_bench/sample_data/data_saved/hf_lobs5_9d_f10/GOOG"


def filtered_keys(raw: Path, work: Path) -> tuple[Path, set]:
    """Level-10-filter a raw generation dir; return (filtered dir, surviving (date,rid) keys)."""
    f10 = work / f"_f10_{raw.name}"
    if f10.exists():
        shutil.rmtree(f10)
    f10.mkdir(parents=True)
    # cut_data_to_lvl globs in_path+'*', so the path must end in a separator
    data_loading.cut_data_to_lvl(str(raw) + "/", str(f10) + "/", 10)
    for f in list(f10.glob("*.csv")):          # the filter can empty a file; Simple_Loader chokes
        if f.stat().st_size < 2:
            f.unlink()
    keys = {(m.group(1), int(m.group(2)))
            for f in f10.glob("*message*_gen_id_0.csv") for m in [RE_GEN.match(f.name)] if m}
    return f10, keys


def build_tree(f10: Path, keys, tree: Path) -> int:
    ref = ref_dir()
    if tree.exists():
        shutil.rmtree(tree)
    for sub in ("data_real", "data_cond", "data_gen"):
        (tree / sub).mkdir(parents=True)
    n = 0
    for date, rid in sorted(keys):
        gen_ok = True
        for kind in ("message", "orderbook"):
            g = f10 / f"GOOG_{date}_{kind}_real_id_{rid}_gen_id_0.csv"
            r = ref / "data_real" / f"GOOG_{date}_{kind}_real_id_{rid}.csv"
            c = ref / "data_cond" / f"GOOG_{date}_{kind}_real_id_{rid}.csv"
            if not (g.exists() and r.exists() and c.exists()):
                gen_ok = False
        if not gen_ok:
            continue
        for kind in ("message", "orderbook"):
            (tree / "data_gen" / f"GOOG_{date}_{kind}_real_id_{rid}_gen_id_0.csv").symlink_to(
                (f10 / f"GOOG_{date}_{kind}_real_id_{rid}_gen_id_0.csv").resolve())
            for sub in ("data_real", "data_cond"):
                (tree / sub / f"GOOG_{date}_{kind}_real_id_{rid}.csv").symlink_to(
                    (ref / sub / f"GOOG_{date}_{kind}_real_id_{rid}.csv").resolve())
        n += 1
    return n


def score(tree: Path) -> dict:
    loader = data_loading.Simple_Loader(str(tree / "data_real"), str(tree / "data_gen"),
                                        str(tree / "data_cond"))
    for s in loader:
        s.materialize()
    sc, _, _ = scoring.run_benchmark(loader, DEFAULT_SCORING_CONFIG, default_metric=DEFAULT_METRICS)
    return {k: float(v["l1"][0]) for k, v in sc.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gen-root", required=True)
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-intersect", action="store_true",
                    help="score each arm on its own surviving windows instead of the common set")
    # Two-phase mode, so the arms can be scored as parallel processes without any of them changing
    # which windows the others see. Phase 1 (--filter-only) writes the level-10-filtered copies and
    # the intersection; phase 2 (--score-only, reading --keys-file) only reads them.
    # Running the whole thing per-process instead would both race on the shared filtered dirs and
    # redo the expensive filter N times.
    ap.add_argument("--filter-only", action="store_true",
                    help="phase 1: filter every arm, write keys.json, score nothing")
    ap.add_argument("--keys-file", default=None,
                    help="phase 2: read the window set from this json instead of re-filtering")
    ap.add_argument("--score-only", default=None, metavar="ARM",
                    help="phase 2: score just this arm into aggregates_<ARM>.csv")
    a = ap.parse_args()
    if a.score_only and a.score_only not in a.arms:
        raise SystemExit(f"--score-only {a.score_only} must be one of --arms {a.arms}")

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    work = out / "_work"; work.mkdir(exist_ok=True)
    root = Path(a.gen_root)

    keys_path = Path(a.keys_file) if a.keys_file else out / "keys.json"
    f10s = {arm: work / f"_f10_{arm}" for arm in a.arms}
    if a.keys_file:                                     # phase 2: reuse phase 1's filtered dirs
        blob = json.loads(keys_path.read_text())
        keysets = {arm: {(d, r) for d, r in v} for arm, v in blob["per_arm"].items()}
        common = {(d, r) for d, r in blob["common"]}
        for arm in a.arms:
            if not f10s[arm].is_dir():
                raise SystemExit(f"phase 2 needs the phase-1 filtered dir {f10s[arm]}")
        print(f"[filter] reusing {keys_path}: {len(common)} common windows", flush=True)
    else:
        keysets = {}
        for arm in a.arms:
            raw = root / arm
            if not raw.exists():
                raise SystemExit(f"missing generation dir {raw}")
            f10s[arm], keysets[arm] = filtered_keys(raw, work)
            print(f"[filter] {arm:10s} {len(keysets[arm]):5d} windows survive the level-10 filter",
                  flush=True)
        common = set.intersection(*keysets.values())
        print(f"[filter] common to all {len(a.arms)} arms: {len(common)} windows", flush=True)
        keys_path.write_text(json.dumps(
            {"per_arm": {k: sorted(v) for k, v in keysets.items()}, "common": sorted(common)}))
        print(f"[filter] wrote {keys_path}", flush=True)
    if a.no_intersect:
        common = None
    if a.filter_only:
        return

    rows, per_stat = [], {}
    todo = [a.score_only] if a.score_only else a.arms
    for arm in todo:
        keys = keysets[arm] if a.no_intersect else common
        tree = work / arm
        n = build_tree(f10s[arm], keys, tree)
        print(f"[score]  {arm:10s} scoring {n} windows ...", flush=True)
        per = score(tree)
        per_stat[arm] = per
        l1_21 = float(np.mean(list(per.values())))
        sub18 = [v for k, v in per.items() if k not in CLOCK_STATS]
        l1_18 = float(np.mean(sub18))
        book10 = float(np.mean([per[s] for s in BOOK10_STATS if s in per]))
        rows.append({"arm": arm, "n_windows": n, "n_survive_filter": len(keysets[arm]),
                     "l1_21": l1_21, "l1_18": l1_18, "book_10": book10, "n_stats": len(per)})
        print(f"[score]  {arm:10s} L1_21={l1_21:.4f}  L1_18={l1_18:.4f}  book10={book10:.4f}",
              flush=True)
        suffix = f"_{a.score_only}" if a.score_only else ""
        pd.DataFrame(rows).to_csv(out / f"aggregates{suffix}.csv", index=False)
        pd.DataFrame(per_stat).rename_axis("statistic").to_csv(out / f"per_stat_l1{suffix}.csv")

    d = pd.DataFrame(rows)
    print("\n" + d.to_string(index=False))
    print(f"\n[wrote] {out}/aggregates{'_' + a.score_only if a.score_only else ''}.csv")


if __name__ == "__main__":
    main()
