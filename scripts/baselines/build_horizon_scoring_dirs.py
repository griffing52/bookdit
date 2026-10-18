#!/usr/bin/env python
"""Adapt LOB-S5 shard / CST generation dirs into the layout `score_lobs5_horizon.py` expects.

The three producers disagree on naming, and the scorer wants a fourth thing:

    gen_lobs5_on_lobbench_windows.py -> gen_{rid}.npy , meta{windows:[[rid,s0]], gen, cond, proc}
    cst_generate.py                  -> gen_{wi}.npy  , meta{model_starts:[s0], gen, cond, proc}
    score_lobs5_horizon.py wants     -> gen_h_{wi}.npy, meta{starts:[s0], K, horizon, cond, proc}

`s0` means the SAME index in all three (the frame before the conditioning window: the seed slice is
proc[s0-warmup : s0+cond] and generation begins at s0+cond), which is what makes the remap safe.

It also restricts every arm to a COMMON rid set so the horizon curves are scored on identical
windows -- without that the arms differ in both model and window sample and no comparison is clean.

Usage
    python scripts/baselines/build_horizon_scoring_dirs.py \
        --lobs5-shards <lobs5 gen>/shard0 <lobs5 gen>/shard1 ... \
        --cst-dir <cst gen> --cst-prep <cst prep> --starts-json <starts.json> \
        --restrict-to-run runs/horizon/bookdit \
        --out-lobs5 <lobs5 scoring dir> --out-cst <cst scoring dir>

The two --out-* dirs are what score_lobs5_horizon.py and horizon_score_cache.py take as --gen-dir.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from functools import reduce
from pathlib import Path


RID_GEN = re.compile(r"real_id_(\d+)_gen")


def bookdit_fixed_rids(run: Path) -> set[int]:
    """The windows BookDiT supports at EVERY block. profile.csv's n_windows counts windows entering
    the level-ten filter, so it overstates this; see rescore_horizon_fixedset.py."""
    sets = []
    for bd in sorted(run.glob("block_*"), key=lambda p: int(p.name.split("_")[1])):
        sets.append({int(RID_GEN.search(f.name).group(1))
                     for f in (bd / "data_gen").glob("*message*")})
    return reduce(lambda a, b: a & b, sets)


def write_tree(out: Path, pairs: list[tuple[int, Path]], proc: str, cond: int,
               horizon: int, warmup: int, label: str) -> None:
    """pairs = [(s0, source .npy)] already restricted and ordered."""
    if out.exists():
        for f in out.glob("*"):
            f.unlink()
    out.mkdir(parents=True, exist_ok=True)
    starts = []
    for wi, (s0, src) in enumerate(pairs):
        os.symlink(src.resolve(), out / f"gen_h_{wi}.npy")
        starts.append(int(s0))
    meta = {"proc": proc, "cond": cond, "horizon": horizon, "warmup": warmup,
            "starts": starts, "K": len(starts)}
    (out / "meta.json").write_text(json.dumps(meta))
    print(f"[{label}] {len(starts)} windows -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lobs5-shards", nargs="*", default=[])
    ap.add_argument("--cst-dir", default=None)
    ap.add_argument("--cst-prep", default=None)
    ap.add_argument("--starts-json", default=None,
                    help="the starts json ([{rid, start}]) the generations were built from; "
                         "gives the start-index -> rid map CST needs")
    ap.add_argument("--restrict-to-run", default=None,
                    help="a horizon_official_protocol.py run dir (e.g. runs/horizon/bookdit); "
                         "restrict to the rids it supports at EVERY block")
    ap.add_argument("--out-lobs5", default=None)
    ap.add_argument("--out-cst", default=None)
    a = ap.parse_args()

    keep = bookdit_fixed_rids(Path(a.restrict_to_run)) if a.restrict_to_run else None
    if keep is not None:
        print(f"[restrict] BookDiT all-block rid set: {len(keep)}")

    # ---- LOB-S5: merge shards, key by rid
    l5 = {}
    proc = cond = horizon = warmup = None
    for sh in a.lobs5_shards:
        sh = Path(sh)
        m = json.loads((sh / "meta.json").read_text())
        proc, cond, horizon, warmup = m["proc"], m["cond"], m["gen"], m.get("warmup", 8000)
        for rid, s0 in m["windows"]:
            f = sh / f"gen_{rid}.npy"
            if f.exists():
                l5[int(rid)] = (int(s0), f)
    if l5:
        rids = sorted(l5) if keep is None else sorted(set(l5) & keep)
        missing = (keep - set(l5)) if keep is not None else set()
        print(f"[lobs5] generated {len(l5)} | usable after restrict {len(rids)}"
              + (f" | MISSING from the fixed set: {len(missing)}" if missing else ""))
        write_tree(Path(a.out_lobs5), [l5[r] for r in rids], proc, cond, horizon, warmup, "lobs5")
        json.dump(rids, open(Path(a.out_lobs5) / "rids.json", "w"))

    # ---- CST: gen_{wi}.npy indexes prep meta's model_starts POSITIONALLY, and the CST prep need not
    # cover the same window list, in the same order, as the LOB-S5 shards.  Positional alignment
    # against `sorted(l5)` could therefore mismatch SILENTLY.  Map by the start index instead, which
    # is generator-independent: s0 = entry["start"] - cond.
    if a.cst_dir:
        pm = json.loads((Path(a.cst_prep) / "meta.json").read_text())
        starts = [int(s) for s in pm["model_starts"]]
        s0_to_rid = {}
        if a.starts_json:
            for e in json.loads(Path(a.starts_json).read_text()):
                s0_to_rid[int(e["start"]) - int(pm["cond"])] = int(e["rid"])
        pairs, cst_rids = [], []
        for wi, s0 in enumerate(starts):
            f = Path(a.cst_dir) / f"gen_{wi}.npy"
            if not f.exists():
                continue
            rid = s0_to_rid.get(s0)
            if keep is not None and (rid is None or rid not in keep):
                continue
            pairs.append((s0, f)); cst_rids.append(rid)
        if s0_to_rid:
            print(f"[cst] start->rid resolved for {sum(1 for r in cst_rids if r is not None)}"
                  f"/{len(cst_rids)} kept windows")
        print(f"[cst] generated {len(glob.glob(str(Path(a.cst_dir) / 'gen_*.npy')))} | "
              f"usable after restrict {len(pairs)}")
        write_tree(Path(a.out_cst), pairs, pm["proc"], pm["cond"], pm["gen"],
                   pm.get("warmup", 8000), "cst")
        json.dump(cst_rids, open(Path(a.out_cst) / "rids.json", "w"))


if __name__ == "__main__":
    main()
