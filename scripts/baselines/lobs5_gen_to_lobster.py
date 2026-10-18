#!/usr/bin/env python
"""Convert retrained-LOB-S5 sampled messages into LOBSTER (message, orderbook) CSVs.

The sampler emits raw (time, event_type, size, abs_price, direction) tuples with no order ids and no
book. Replay them through the crossing-aware QueueBook -- warmed on the real messages preceding each
window -- to attach order ids and a row-aligned book, and write them under the official LOB-Bench
file naming so the retrain is scored on exactly the windows the released model is scored on.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "lib"))
sys.path.insert(0, str(ROOT / "scripts" / "baselines"))
import score_lobs5_retrained as S                                            # noqa: E402
from rescore_matching import MatchingBook                                    # noqa: E402

S.QueueBook = MatchingBook          # crossing-aware replay: a generated order that crosses the book
build_stream = S.build_stream       # must EXECUTE against the far side, not rest at its stated price


def proc_events(proc, lo, hi):
    """proc rows -> (time, event_type, size, abs_price, side) tuples."""
    out = []
    for r in proc[lo:hi]:
        etype, dir01, pabs, size = int(r[1]), int(r[2]), int(r[3]), int(r[5])
        t = float(r[8]) + float(r[9]) / 1e9
        out.append((t, etype, size, pabs, 2 * dir01 - 1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-dir", required=True,
                    help="gen_lobs5_on_lobbench_windows.py --out (gen_<rid>.npy + meta.json)")
    ap.add_argument("--proc", required=True,
                    help="the same LOB-S5 proc npy the generations were conditioned on")
    ap.add_argument("--date", required=True, help="trading day of --proc, e.g. 2023-01-03")
    ap.add_argument("--out", required=True)
    ap.add_argument("--warmup", type=int, default=8000)
    ap.add_argument("--cond", type=int, default=500)
    a = ap.parse_args()

    proc = np.load(a.proc, allow_pickle=True).astype(np.int64)
    meta = json.load(open(Path(a.gen_dir) / "meta.json"))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    written = 0
    for rid, s0 in meta["windows"]:
        f = Path(a.gen_dir) / f"gen_{rid}.npy"
        if not f.exists():
            continue
        g = np.load(f, allow_pickle=True)
        gen_ev = [(float(r[0]), int(r[1]), int(r[2]), int(r[3]), int(r[4])) for r in g]
        warm = proc_events(proc, s0 - a.warmup, s0 + a.cond)       # through the conditioning window
        m, b = build_stream(warm, gen_ev)
        if len(m) < 8:
            continue
        m.to_csv(out / f"GOOG_{a.date}_message_real_id_{rid}_gen_id_0.csv", header=False, index=False)
        b.to_csv(out / f"GOOG_{a.date}_orderbook_real_id_{rid}_gen_id_0.csv", header=False, index=False)
        written += 1
    print(f"DONE: {written} windows -> {out}")


if __name__ == "__main__":
    main()
