#!/usr/bin/env python
"""Message-model scoring helpers, plus a scorer for generations keyed by window position.

A message-model sampler (the LOB-S5 retrain, or cst_generate.py) emits, per window, the sampled
continuation messages (time, event_type, size, abs_price, direction) with no order ids. Here (main
venv) every stream (real / cond / gen) is rebuilt through the FIFO QueueBook the book-state path
uses, to fabricate order ids and row-aligned 10-level book snapshots; the cond/real/gen tree is
written exactly as run_lobbench_worldmodel_official.write_pair does and scored with W.score_tree
(lob_bench's DEFAULT_SCORING_CONFIG). The floor is a disjoint real slice (real vs real).

`proc_events`, `build_stream` and `build_tree_msgs` are shared by lobs5_gen_to_lobster.py,
rescore_matching.py, score_lobs5_horizon.py, cst_prep.py and horizon_score_cache.py; those that
need the crossing-aware replay swap `QueueBook` for `rescore_matching.MatchingBook` first.

`main()` reads the layout cst_generate.py writes (gen_<wi>.npy + meta.json with `model_starts`).
LOB-S5 generations on the official windows (gen_<rid>.npy) go through lobs5_gen_to_lobster.py
and LOB-Bench's run_bench.py instead.
"""
from __future__ import annotations
import argparse, json, shutil, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "lib"))

from _paths import lob_bench_dir                       # noqa: E402

sys.path.insert(0, str(lob_bench_dir(required=True)))               # $LOB_BENCH_DIR, default <repo>/libs/lob_bench

import run_lobbench_worldmodel_official as W          # write_pair, score_tree
from idm_annotator import QueueBook                    # FIFO order-id + book snapshots

COLS = ["time", "event_type", "order_id", "size", "price", "direction"]


def proc_events(proc, a, b):
    """rows a..b of the 14-field proc npy -> list of (time, type, size, price, dir[+1 bid/-1 ask])."""
    out = []
    for r in proc[a:b]:
        etype, dir01, pabs, size = int(r[1]), int(r[2]), int(r[3]), int(r[5])
        t = int(r[8]) + int(r[9]) / 1e9
        out.append((t, etype, size, pabs, 2 * dir01 - 1))
    return out


def build_stream(warmup_events, window_events):
    """Seed a QueueBook with warmup_events (no messages emitted), then replay window_events,
    emitting a msg row + book-AFTER snapshot per event (skipping reductions with no tracked liquidity)."""
    qb = QueueBook()
    for (_t, typ, size, price, side) in warmup_events:
        if typ == 1:
            qb.add_limit(side, price, size)
        elif typ in (2, 3, 4):
            qb.reduce(side, price, size, "exec" if typ == 4 else "cancel")
    rows, books = [], []
    for (t, typ, size, price, side) in window_events:
        if typ == 1:
            oid = qb.add_limit(side, price, size)
        elif typ in (2, 3, 4):
            oid = qb.reduce(side, price, size, "exec" if typ == 4 else "cancel")
            if oid is None:
                continue
        else:
            continue
        rows.append((t, typ, oid, size, price, side))
        books.append(qb.snapshot())
    return pd.DataFrame(rows, columns=COLS), pd.DataFrame(np.asarray(books))


def build_tree_msgs(base: Path, triples):
    """triples: list of (real, cond, gen) where each is (msg_df, book_df)."""
    for sub in ("data_real", "data_gen", "data_cond"):
        (base / sub).mkdir(parents=True, exist_ok=True)
    kept = 0
    for rid, ((mr, br), (mc, bc), (mg, bg)) in enumerate(triples):
        if min(len(mr), len(mc), len(mg)) < 8:
            continue
        W.write_pair(base / "data_real", rid, None, mr, br)
        W.write_pair(base / "data_cond", rid, None, mc, bc)
        W.write_pair(base / "data_gen", rid, 0, mg, bg)
        kept += 1
    return kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-dir", required=True,
                    help="generations in the gen_<wi>.npy + meta.json{model_starts} layout")
    ap.add_argument("--warmup", type=int, default=8000)
    ap.add_argument("--out", default="runs/lobbench/lobs5_retrained")
    ap.add_argument("--stats", nargs="*", default=None)
    ap.add_argument("--gen-only", action="store_true",
                    help="LOB-Bench standard: score only the generated continuation, not cond+gen")
    args = ap.parse_args()

    gen_dir = Path(args.gen_dir)
    meta = json.loads((gen_dir / "meta.json").read_text())
    proc = np.load(meta["proc"], allow_pickle=True).astype(np.int64)
    cond, gen_n = meta["cond"], meta["gen"]
    T = cond + gen_n
    Wu = args.warmup

    out = ROOT / args.out
    if out.exists():
        shutil.rmtree(out)

    # ---- model tree: real=cond+realcont, cond=cond, gen=cond+generated ----
    model_triples = []
    for wi, s in enumerate(meta["model_starts"]):
        gpath = gen_dir / f"gen_{wi}.npy"
        if not gpath.exists():
            continue
        g = np.load(gpath, allow_pickle=True)
        gen_events = [(float(r[0]), int(r[1]), int(r[2]), int(r[3]), int(r[4])) for r in g]
        warm = proc_events(proc, s - Wu, s)
        cond_ev = proc_events(proc, s, s + cond)
        condp = build_stream(warm, cond_ev)
        if args.gen_only:                                     # LOB-Bench standard: continuation only
            warm_c = proc_events(proc, s - Wu, s + cond)      # warm book THROUGH the cond
            real = build_stream(warm_c, proc_events(proc, s + cond, s + T))   # real continuation
            gener = build_stream(warm_c, gen_events)          # generated continuation
        else:
            real = build_stream(warm, proc_events(proc, s, s + T))           # cond + real continuation
            gener = build_stream(warm, cond_ev + gen_events)                 # cond + model continuation
        model_triples.append((real, condp, gener))
    print(f"[model] {len(model_triples)} windows assembled")

    # ---- floor tree: gen replaced by a disjoint real slice (real-vs-realB) ----
    # realB drawn from the SAME (model_starts) window distribution -> a properly-calibrated
    # real-vs-real floor (floor_starts sit in a different intraday regime and inflate it).
    mstarts = meta["model_starts"]
    floor_triples = []
    for wi, s in enumerate(mstarts):
        sb = mstarts[(wi + 1) % len(mstarts)]
        if sb == s:
            continue
        warm = proc_events(proc, s - Wu, s)
        cond_ev = proc_events(proc, s, s + cond)
        condp = build_stream(warm, cond_ev)
        if args.gen_only:
            warm_c = proc_events(proc, s - Wu, s + cond)
            warmB_c = proc_events(proc, sb - Wu, sb + cond)
            real = build_stream(warm_c, proc_events(proc, s + cond, s + T))
            realB = build_stream(warmB_c, proc_events(proc, sb + cond, sb + T))
        else:
            real = build_stream(warm, proc_events(proc, s, s + T))
            realB = build_stream(proc_events(proc, sb - Wu, sb), proc_events(proc, sb, sb + T))
        floor_triples.append((real, condp, realB))

    km = build_tree_msgs(out / "model", model_triples)
    print(f"[model] {km} scored triples -> official scoring ...")
    model_stats = W.score_tree(out / "model", args.stats)
    kf = build_tree_msgs(out / "floor", floor_triples)
    print(f"[floor] {kf} scored triples -> official scoring ...")
    floor_stats = W.score_tree(out / "floor", args.stats)

    from run_bench import DEFAULT_SCORING_CONFIG
    rows = []
    print(f"\n{'statistic':26} {'floor_l1':>9} {'model_l1':>9} {'floor_w':>9} {'model_w':>9}  ok")
    for k in (args.stats or DEFAULT_SCORING_CONFIG.keys()):
        fl1, fw, fok = floor_stats.get(k, (np.nan, np.nan, "miss"))
        ml1, mw, mok = model_stats.get(k, (np.nan, np.nan, "miss"))
        rows.append((k, fl1, ml1, fw, mw, fok, mok))
        print(f"{k:26} {fl1:9.3f} {ml1:9.3f} {fw:9.3f} {mw:9.3f}  {mok}")
    df = pd.DataFrame(rows, columns=["statistic", "floor_l1", "model_l1", "floor_w", "model_w", "floor_ok", "model_ok"])
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "scores.csv", index=False)

    ok = df[(df.model_ok == "ok") & df.model_l1.notna() & df.floor_l1.notna()]
    clean = ok[ok.floor_l1 < 0.15]
    print(f"\n=== LOB-Bench aggregate ({gen_dir}) ===")
    print(f"  ALL scored ({len(ok)}): L1={ok.model_l1.mean():.3f}  W={ok.model_w.mean():.3f}   (floor L1={ok.floor_l1.mean():.3f})")
    if len(clean):
        print(f"  CLEAN-floor ({len(clean)}, floor<0.15): L1={clean.model_l1.mean():.3f}  W={clean.model_w.mean():.3f}   (floor L1={clean.floor_l1.mean():.3f})")
    print(f"wrote {out/'scores.csv'}")


if __name__ == "__main__":
    main()
