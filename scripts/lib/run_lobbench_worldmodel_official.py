#!/usr/bin/env python
"""Write a LOB-Bench cond/real/gen tree and score it with LOB-Bench's `scoring.run_benchmark`.

`build_tree` decodes book windows to messages with `idm_decoder.decode_book_sequence`;
`write_pair` writes one message/orderbook CSV pair; `score_tree` runs every LOB-Bench statistic
on a tree and returns {stat: (l1, wasserstein, status)}. A statistic that raises is recorded as
NaN with an ``ERR:<exception>`` status rather than aborting the whole tree.
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "lib"))

from _paths import lob_bench_dir                                            # noqa: E402

sys.path.insert(0, str(lob_bench_dir(required=True)))  # $LOB_BENCH_DIR, default <repo>/libs/lob_bench

from idm_decoder import decode_book_sequence                                # noqa: E402
import scoring, data_loading, metrics as M                                 # noqa: E402
from run_bench import DEFAULT_SCORING_CONFIG                                # noqa: E402

# Placeholder date for the written filenames: LOB-Bench's loader parses a date out of each
# filename, but scoring does not use it, so every tree (whatever its real dates) carries this one.
DATE = "2024-07-01"


def write_pair(d: Path, rid, gid, msg_df, book):
    suf = f"real_id_{rid}" + ("" if gid is None else f"_gen_id_{gid}")
    msg_df.to_csv(d / f"GOOG_{DATE}_message_{suf}.csv", header=False, index=False)
    pd.DataFrame(book).to_csv(d / f"GOOG_{DATE}_orderbook_{suf}.csv", header=False, index=False)


def build_tree(base: Path, reals, conds, gens):
    for sub in ("data_real", "data_gen", "data_cond"):
        (base / sub).mkdir(parents=True, exist_ok=True)
    kept = 0
    for rid, (rbk, cbk, gbk) in enumerate(zip(reals, conds, gens)):
        mr, br = decode_book_sequence(rbk)
        mc, bc = decode_book_sequence(cbk)
        mg, bg = decode_book_sequence(gbk)
        if min(len(mr), len(mc), len(mg)) < 8:        # too few events to score
            continue
        write_pair(base / "data_real", rid, None, mr, br)
        write_pair(base / "data_cond", rid, None, mc, bc)
        write_pair(base / "data_gen", rid, 0, mg, bg)
        kept += 1
    return kept


def score_tree(base: Path, stats=None):
    loader = data_loading.Simple_Loader(str(base / "data_real"), str(base / "data_gen"), str(base / "data_cond"))
    for s in loader:
        s.materialize()
    keys = stats or list(DEFAULT_SCORING_CONFIG.keys())
    per = {}
    for name in keys:
        if name not in DEFAULT_SCORING_CONFIG:
            continue
        try:
            scores, _, _ = scoring.run_benchmark(loader, {name: DEFAULT_SCORING_CONFIG[name]},
                                                 default_metric={"l1": M.l1_by_group, "wasserstein": M.wasserstein})
            l1 = float(scores[name]["l1"][0]); w = float(scores[name]["wasserstein"][0])
            per[name] = (l1, w, "ok")
        except Exception as e:  # noqa: BLE001
            per[name] = (float("nan"), float("nan"), f"ERR:{type(e).__name__}")
    return per
