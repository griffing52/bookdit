#!/usr/bin/env python
"""Cache the PER-WINDOW LOB-Bench score arrays for every block of a long-horizon run.

The window bootstrap behind the long-horizon drift bands (runs/figures/horizon_bands.pdf)
needs the LOB-Bench L1 recomputed on resampled window sets, and a full re-score costs ~77 s per
block.  The L1 is a deterministic function of the
per-window score arrays (see _l1replay.py), and the expensive part of run_benchmark is
metrics.l1_by_group / metrics.wasserstein, each of which runs its own 100-replicate internal
bootstrap.  So partitioning.score_real_gen is called once per block -- the scoring functions only,
no metrics -- and the ragged per-window arrays are cached.  The bootstrap then replays the binning
in numpy at ~0.1 s per block.

Arms:
    bookdit  block trees already on disk under the horizon_official_protocol.py run (--run)/block_*
    lobs5    trees are NOT on disk (score_lobs5_horizon.py rmtree's them); rebuilt here from the
             gen-dir with the SAME crossing-aware MatchingBook replay and level-10 filter the
             published profile used, then cached and deleted.
    cst      idem.

Usage:
    horizon_score_cache.py --arm bookdit --run runs/horizon/bookdit   # -> runs/horizon/cache/bookdit
    horizon_score_cache.py --arm lobs5 --gen-dir <lobs5 scoring dir>  # -> runs/horizon/cache/lobs5
    horizon_score_cache.py --arm cst --gen-dir <cst scoring dir>      # -> runs/horizon/cache/cst

The default --out, runs/horizon/cache/<arm>, is where horizon_matched_bands.py looks
(its --cache-root).
"""
from __future__ import annotations
import argparse
import json
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "lib"))
sys.path.insert(0, str(ROOT / "scripts" / "baselines"))
from _paths import lob_bench_dir                                        # noqa: E402
sys.path.insert(0, str(lob_bench_dir(required=True)))

import data_loading                                                    # noqa: E402
import partitioning                                                    # noqa: E402
from run_bench import DEFAULT_SCORING_CONFIG                           # noqa: E402

RID_GEN = re.compile(r"real_id_(\d+)_gen")


def ragged(arrs: list[np.ndarray]):
    """flat values + offsets, so a resample is a couple of slices."""
    lens = np.array([len(a) for a in arrs], dtype=np.int64)
    off = np.concatenate([[0], np.cumsum(lens)])
    v = np.concatenate(arrs).astype(np.float64) if len(arrs) else np.zeros(0)
    return v, off


def cache_tree(base: Path, out_npz: Path, horizon: int) -> int:
    """Score one block tree and write its per-window arrays."""
    loader = data_loading.Simple_Loader(str(base / "data_real"), str(base / "data_gen"),
                                        str(base / "data_cond"))
    for s in loader:
        s.materialize()
    rids = [p[1] for p in loader.paths]
    d = {"rids": np.array(rids), "stats": np.array(list(DEFAULT_SCORING_CONFIG)),
         "horizon": np.array([horizon])}
    for name, cfg in DEFAULT_SCORING_CONFIG.items():
        sr, sg = partitioning.score_real_gen(loader, cfg["fn"])
        # scores_gen is one tuple per window (one gen_id each) -> flatten that inner level
        sg = [np.concatenate([np.asarray(x, dtype=float).ravel() for x in g])
              if isinstance(g, (tuple, list)) else np.asarray(g, dtype=float).ravel() for g in sg]
        sr = [np.asarray(r, dtype=float).ravel() for r in sr]
        d[f"real__{name}__v"], d[f"real__{name}__o"] = ragged(sr)
        d[f"gen__{name}__v"], d[f"gen__{name}__o"] = ragged(sg)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, **d)
    return len(rids)


# --------------------------------------------------------------------------- lobs5 / cst trees
class MessageLane:
    """Rebuilds block trees for a message-lane arm, exactly as score_lobs5_horizon.py does
    (MatchingBook replay monkeypatched in from `rescore_matching`, then the official level filter),
    but keeping the window -> rid map that build_tree_msgs throws away and hoisting the per-window
    event decode out of the block loop."""

    def __init__(self, gen_dir: Path):
        import score_lobs5_retrained as S
        from rescore_matching import MatchingBook
        S.QueueBook = MatchingBook                 # build_stream() picks this up via module global
        import run_lobbench_worldmodel_official as W
        from score_lobs5_retrained import proc_events, build_stream
        self.W, self.build_stream = W, build_stream

        meta = json.loads((gen_dir / "meta.json").read_text())
        self.rids = json.loads((gen_dir / "rids.json").read_text())
        proc = np.load(meta["proc"], allow_pickle=True).astype(np.int64)
        cond, horizon, self.Wu = meta["cond"], meta["horizon"], meta["warmup"]
        starts, self.K = meta["starts"], meta["K"]
        self.gen, self.cond_ev, self.real_cont, self.warm = [], [], [], []
        for wi in range(self.K):
            g = np.load(gen_dir / f"gen_h_{wi}.npy", allow_pickle=True)
            self.gen.append([(float(r[0]), int(r[1]), int(r[2]), int(r[3]), int(r[4]))
                             for r in g])
            s = starts[wi]
            self.cond_ev.append(proc_events(proc, s, s + cond))
            self.real_cont.append(proc_events(proc, s + cond, s + cond + horizon))
            self.warm.append(proc_events(proc, s - self.Wu, s))
        # the conditioning tree is identical at every block -- build it once
        self.condp = [self.build_stream(self.warm[wi], self.cond_ev[wi]) for wi in range(self.K)]

    def block(self, b0: int, block: int, work: Path, lvl: int) -> int:
        b1, Wu = b0 + block, self.Wu
        if work.exists():
            shutil.rmtree(work)
        for sub in ("data_real", "data_gen", "data_cond"):
            (work / sub).mkdir(parents=True)
        n = 0
        for wi in range(self.K):
            if b1 > len(self.gen[wi]) or b1 > len(self.real_cont[wi]):
                continue
            pre = self.warm[wi] + self.cond_ev[wi]
            real = self.build_stream((pre + self.real_cont[wi][:b0])[-Wu:],
                                     self.real_cont[wi][b0:b1])
            gen = self.build_stream((pre + self.gen[wi][:b0])[-Wu:], self.gen[wi][b0:b1])
            condp = self.condp[wi]
            if min(len(real[0]), len(condp[0]), len(gen[0])) < 8:
                continue
            rid = self.rids[wi]              # the OFFICIAL window id, shared across arms
            self.W.write_pair(work / "data_real", rid, None, *real)
            self.W.write_pair(work / "data_cond", rid, None, *condp)
            self.W.write_pair(work / "data_gen", rid, 0, *gen)
            n += 1
        return _level_filter(work, lvl, n)


def _level_filter(work: Path, lvl: int, n: int) -> int:
    if lvl:
        for sub in ("data_real", "data_gen", "data_cond"):
            tmp = work / f"{sub}_f{lvl}"
            tmp.mkdir(exist_ok=True)
            data_loading.cut_data_to_lvl(str(work / sub) + "/", str(tmp) + "/", lvl)
            shutil.rmtree(work / sub)
            tmp.rename(work / sub)
        for sub in ("data_real", "data_gen", "data_cond"):
            for f in (work / sub).glob("*.csv"):
                if f.stat().st_size < 2:
                    f.unlink()
        gen_ids = {f.name.split("_real_id_")[1].split("_gen")[0]
                   for f in (work / "data_gen").glob("*message*")}
        for sub in ("data_real", "data_cond"):
            for f in (work / sub).glob("*.csv"):
                if f.name.split("_real_id_")[1].split(".")[0] not in gen_ids:
                    f.unlink()
        n = len(list((work / "data_gen").glob("*message*")))
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", required=True, choices=["bookdit", "lobs5", "cst"])
    ap.add_argument("--run", default=str(ROOT / "runs/horizon/bookdit"),
                    help="bookdit only: the horizon_official_protocol.py --out whose block_* "
                         "trees are on disk")
    ap.add_argument("--gen-dir", default=None, help="lobs5/cst: the scoring dir (gen_h_*.npy)")
    ap.add_argument("--block", type=int, default=500)
    ap.add_argument("--n-blocks", type=int, default=32)
    ap.add_argument("--lvl", type=int, default=10)
    ap.add_argument("--work", default=None)
    ap.add_argument("--out", default=None, help="default runs/horizon/cache/<arm>")
    a = ap.parse_args()
    if a.arm != "bookdit" and not a.gen_dir:
        ap.error(f"--arm {a.arm} needs --gen-dir")

    out = Path(a.out) if a.out else ROOT / "runs/horizon/cache" / a.arm
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if a.arm == "bookdit":
        bds = sorted(Path(a.run).glob("block_*"), key=lambda p: int(p.name.split("_")[1]))
        for bi, bd in enumerate(bds[:a.n_blocks]):
            h = (bi + 1) * a.block
            n = cache_tree(bd, out / f"block_{bi:02d}.npz", h)
            print(f"[{a.arm}] h={h:6d} cached {n} windows  ({time.time()-t0:6.0f}s)", flush=True)
    else:
        lane = MessageLane(Path(a.gen_dir))
        print(f"[{a.arm}] loaded {lane.K} windows ({time.time()-t0:.0f}s)", flush=True)
        work = Path(a.work) if a.work else out / "_work"
        for bi in range(a.n_blocks):
            b0 = bi * a.block
            h = b0 + a.block
            n = lane.block(b0, a.block, work, a.lvl)
            if n < 4:
                print(f"[{a.arm}] h={h}: only {n} windows -> stop", flush=True)
                break
            n = cache_tree(work, out / f"block_{bi:02d}.npz", h)
            shutil.rmtree(work, ignore_errors=True)
            print(f"[{a.arm}] h={h:6d} cached {n} windows  ({time.time()-t0:6.0f}s)", flush=True)
    print(f"[{a.arm}] done in {time.time()-t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
