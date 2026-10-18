#!/usr/bin/env python
"""Long-horizon fidelity profile for a message model (the LOB-S5 retrain, or CST).

The same horizon profile as the book-state path, for a message model: for each successive
`block`-event slice of the generated region, build cond/real/gen LOBSTER trees and score them with
`W.score_tree`, giving aggregate L1 against rollout horizon. The gen block at horizon h is seeded by
the model's own accumulated stream (cond + generated[:h]); the real block is the time-aligned real
continuation.

Every stream is replayed through the crossing-aware `MatchingBook` (rescore_matching.py): a plain
QueueBook rests a crossing limit order at its stated price and leaves a crossed book. The official
level-10 filter is applied by default, as in horizon_official_protocol.py for BookDiT.

--gen-dir is the layout build_horizon_scoring_dirs.py writes (gen_h_<wi>.npy + meta.json).
Writes <--out>/profile.csv.
"""
from __future__ import annotations
import argparse, json, shutil, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "lib"))
sys.path.insert(0, str(ROOT / "scripts" / "baselines"))
from _paths import lob_bench_dir                                             # noqa: E402
sys.path.insert(0, str(lob_bench_dir(required=True)))  # $LOB_BENCH_DIR, default <repo>/libs/lob_bench

import run_lobbench_worldmodel_official as W                                 # noqa: E402
import score_lobs5_retrained as S                                            # noqa: E402
from rescore_matching import MatchingBook                                    # noqa: E402
S.QueueBook = MatchingBook          # build_stream() picks this up via the module global
from score_lobs5_retrained import proc_events, build_stream, build_tree_msgs  # noqa: E402


def aggregate(per):
    l1 = [l for l, _, ok in per.values() if ok == "ok" and np.isfinite(l)]
    w = [w_ for _, w_, ok in per.values() if ok == "ok" and np.isfinite(w_)]
    return float(np.mean(l1)), float(np.mean(w)), len(l1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-dir", required=True,
                    help="long-horizon generations in the build_horizon_scoring_dirs.py layout")
    ap.add_argument("--warmup", type=int, default=8000)
    ap.add_argument("--block", type=int, default=500)
    ap.add_argument("--out", default="runs/horizon/lobs5",
                    help="default runs/horizon/lobs5, where horizon_matched_bands.py --validate looks")
    ap.add_argument("--lvl", type=int, default=10,
                    help="LOB-Bench level-N filter applied before scoring (default 10, the "
                         "official protocol; 0 = off)")
    args = ap.parse_args()

    gd = Path(args.gen_dir)
    meta = json.loads((gd / "meta.json").read_text())
    proc = np.load(meta["proc"], allow_pickle=True).astype(np.int64)
    cond, horizon = meta["cond"], meta["horizon"]
    Wu = args.warmup
    starts = meta["starts"]
    K = meta["K"]

    # preload generated streams per window
    gens = []
    for wi in range(K):
        g = np.load(gd / f"gen_h_{wi}.npy", allow_pickle=True)
        gens.append([(float(r[0]), int(r[1]), int(r[2]), int(r[3]), int(r[4])) for r in g])
    n_gen = [len(g) for g in gens]
    print(f"[data] K={K} horizon={horizon} mean_generated/window={np.mean(n_gen):.0f} "
          f"min={min(n_gen)} max={max(n_gen)}")

    # cond + full real continuation events per window (as event lists)
    cond_ev = [proc_events(proc, s, s + cond) for s in starts]
    real_cont = [proc_events(proc, s + cond, s + cond + horizon) for s in starts]
    warm_pre = [proc_events(proc, s - Wu, s) for s in starts]

    out = ROOT / args.out
    if out.exists():
        shutil.rmtree(out)
    rows = []
    for b0 in range(0, horizon, args.block):
        b1 = b0 + args.block
        triples = []
        for wi in range(K):
            if b1 > len(gens[wi]) or b1 > len(real_cont[wi]):
                continue
            # cond: shared 500 real events
            condp = build_stream(warm_pre[wi], cond_ev[wi])
            # real block: seed with cond + real continuation up to b0, score [b0:b1]
            real_seed = warm_pre[wi] + cond_ev[wi] + real_cont[wi][:b0]
            real = build_stream(real_seed[-Wu:], real_cont[wi][b0:b1])
            # gen block: seed with cond + generated up to b0 (model's OWN accumulated stream)
            gen_seed = warm_pre[wi] + cond_ev[wi] + gens[wi][:b0]
            gen = build_stream(gen_seed[-Wu:], gens[wi][b0:b1])
            triples.append((real, condp, gen))
        base = out / f"h{b0:06d}"
        kept = build_tree_msgs(base, triples)
        if args.lvl:
            # horizon_official_protocol.py applies cut_data_to_lvl(..., 10) to all three subdirs
            # before scoring; W.score_tree does not. Without it the message-lane arms would be
            # scored unfiltered, on a different protocol from BookDiT's horizon profile. Same steps
            # as there, including the empty-file cleanup (the filter can empty a CSV, which crashes
            # Simple_Loader's time parser) and the gen_ids sync that keeps the folders consistent.
            import data_loading as _dl
            for sub in ("data_real", "data_gen", "data_cond"):
                tmp = base / f"{sub}_f{args.lvl}"; tmp.mkdir(exist_ok=True)
                _dl.cut_data_to_lvl(str(base / sub) + "/", str(tmp) + "/", args.lvl)
                shutil.rmtree(base / sub); tmp.rename(base / sub)
            for sub in ("data_real", "data_gen", "data_cond"):
                for f in (base / sub).glob("*.csv"):
                    if f.stat().st_size < 2:
                        f.unlink()
            gen_ids = {f.name.split("_real_id_")[1].split("_gen")[0]
                       for f in (base / "data_gen").glob("*message*")}
            for sub in ("data_real", "data_cond"):
                for f in (base / sub).glob("*.csv"):
                    if f.name.split("_real_id_")[1].split(".")[0] not in gen_ids:
                        f.unlink()
            kept = len(list((base / "data_gen").glob("*message*")))
        if kept < 4:                      # too few windows reach this horizon -> stop
            shutil.rmtree(base, ignore_errors=True)
            print(f"h={b0+args.block:6d}: kept={kept} (<4) -> horizon coverage exhausted, stopping")
            break
        per = W.score_tree(base)
        l1, w_, n_ok = aggregate(per)
        row = {"horizon": b0 + args.block, "l1": l1, "w": w_, "n_stats": n_ok, "kept": kept,
               "spread": per.get("spread", (np.nan,)*3)[0],
               "inter_arrival": per.get("log_inter_arrival_time", (np.nan,)*3)[0],
               "ask_volume": per.get("ask_volume", (np.nan,)*3)[0]}
        for _s, _v in per.items():
            row[f"stat_{_s}"] = _v[0]
        rows.append(row)
        print(f"h={row['horizon']:6d}: L1={l1:.3f} W={w_:.3f} spread={row['spread']:.3f} "
              f"inter_arr={row['inter_arrival']:.3f} kept={kept}", flush=True)
        # cleanup tree to save disk
        shutil.rmtree(base, ignore_errors=True)
    df = pd.DataFrame(rows)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "profile.csv", index=False)
    print(f"\n=== LONG-HORIZON PROFILE ({gd}) ===")
    print(f"  start (h={df.horizon.iloc[0]}): L1={df.l1.iloc[0]:.3f}")
    print(f"  end   (h={df.horizon.iloc[-1]}): L1={df.l1.iloc[-1]:.3f}")
    print(f"  mean over horizon: L1={df.l1.mean():.3f}")
    print(f"wrote {out/'profile.csv'}")


if __name__ == "__main__":
    main()
