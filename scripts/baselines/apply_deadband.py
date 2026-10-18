#!/usr/bin/env python
"""Size-only deadband on generated book frames, then decode to an official LOB-Bench arm dir.

Generated books carry size jitter at unchanged prices; every jittering size becomes an add/cancel
message in the inverse-dynamics decode.  This script suppresses sub-threshold size changes before
decoding and writes one arm directory per setting for `score_official.py`.

ROLE IN THE PAPER.  Both Table 1 BookDiT rows are the `abs50` setting of this script, decoded
with the decoder's own built-in deadband switched off (`IDM_DEADBAND_ABS=0`, which this script
enforces) so the theta = 50 band is applied exactly once, here, in quantized (side, price) space.
"BookDiT + refinement" applies it to frames dumped by
`gen_bookdit_on_lobbench_windows.py --refine --dump-frames`, and "without refinement" to frames
dumped by the same command without `--refine`; the two rows differ only in the median filter.
The remaining settings are the carry-forward deadband variants of the refinement sweep described
in the paper's Refinement-stage section.

WHERE THE DEADBAND ACTS.  Generated prices are not on the $0.01 tick grid (`gen_book % 100` takes
every residue), and `decode_book_sequence` quantizes them with `_quantize` before diffing, so
deadbanding in raw column space would be the wrong operation -- column l at t and t+1 are
generally different prices.  This script reproduces the decoder's own view
(`_side_maps(_quantize(frame))`, which also merges levels that collapse onto one tick) and
deadbands in that (side, price) space.

WHAT IS AND IS NOT TOUCHED.  Only sizes at prices present in both the previous emitted frame and
the current one are ever changed.  The price set is passed through verbatim: levels that appear
or disappear are never suppressed, so no price is invented or deleted.  Hysteresis is against the
last emitted size, not the last raw one, so a slow drift still eventually fires.  The previous
state is seeded from the last conditioning frame, which the model was itself conditioned on, so
the first generated frame is deadbanded too.

Usage (Table 1's BookDiT rows; CPU only, no generation; docs/REPRODUCE.md section 3.6)
-----
  IDM_DEADBAND_ABS=0 python scripts/baselines/apply_deadband.py \
      --frames $BOOKDIT_WORK/refine_frames --days all --settings raw abs50 \
      --out-root $BOOKDIT_WORK/tab1_refine
  python scripts/baselines/score_official.py --gen-root $BOOKDIT_WORK/tab1_refine \
      --arms raw abs50 --no-intersect --out runs/lobbench/tab1_refine
  # "without refinement": the same on the unrefined dump
  IDM_DEADBAND_ABS=0 python scripts/baselines/apply_deadband.py \
      --frames $BOOKDIT_WORK/norefine_frames --days all --settings abs50 \
      --out-root $BOOKDIT_WORK/tab1_norefine

  Here `raw` means "no deadband at all" (median-refined frames, decoder band off), not
  "no refinement".
"""
from __future__ import annotations

import argparse
import os
import sys
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "lib"))
from _paths import bench_work, lob_bench_dir                              # noqa: E402
sys.path.insert(0, str(lob_bench_dir(required=True)))
# The deadband is applied here, in quantized (side, price) space, so the decoder's own built-in
# band must be off or theta would be applied twice. idm_decoder reads IDM_DEADBAND_ABS at import
# time, so force it before importing.
if os.environ.get("IDM_DEADBAND_ABS", "50") != "0":
    print(f"[apply_deadband] IDM_DEADBAND_ABS={os.environ.get('IDM_DEADBAND_ABS', '50 (default)')}"
          " -> 0: the decoder's built-in band is disabled so the deadband is applied once, here",
          file=sys.stderr)
    os.environ["IDM_DEADBAND_ABS"] = "0"
from idm_decoder import decode_book_sequence, _quantize, _side_maps, TICK  # noqa: E402

# name -> (kind, threshold, first level the band applies to)
#   kind "rel": suppress if |new-old| <  frac * max(old, 1)
#   kind "abs": suppress if |new-old| <  n_shares
#   kind "or" : suppress if |new-old| <  max(frac*old, n_shares)   (threshold is (frac, shares))
#   deep_from = 0 -> every level; 3 -> only levels 4..10 from the touch (the "deep" level set)
SETTINGS = {
    "raw":       (None,  None,          0),
    "rel005":    ("rel", 0.05,          0),
    "rel0075":   ("rel", 0.075,         0),
    "rel010":    ("rel", 0.10,          0),
    "rel015":    ("rel", 0.15,          0),
    "rel025":    ("rel", 0.25,          0),
    "rel040":    ("rel", 0.40,          0),
    "rel060":    ("rel", 0.60,          0),
    "abs25":     ("abs", 25.0,          0),
    "abs50":     ("abs", 50.0,          0),
    "abs75":     ("abs", 75.0,          0),
    "abs100":    ("abs", 100.0,         0),
    "abs250":    ("abs", 250.0,         0),
    "rel010deep": ("rel", 0.10,         3),
    "rel025deep": ("rel", 0.25,         3),
    "or025_100": ("or",  (0.25, 100.0), 0),
    "or005_50":  ("or",  (0.05, 50.0),  0),
    "or010_50":  ("or",  (0.10, 50.0),  0),
}


def _maps(frame40: np.ndarray) -> tuple[dict, dict]:
    """The decoder's own view of a frame: tick-quantized (side -> {price: size})."""
    return _side_maps(_quantize(frame40, TICK))


def _to_frame40(ask: dict, bid: dict, n_lvl: int = 10) -> np.ndarray:
    """(ask map, bid map) -> a 40-col book row, asks ascending / bids descending.

    `decode_book_sequence` reads a frame only through `_side_maps(_quantize(.))`, which is
    order-insensitive and merges duplicate ticks, so this round-trip is loss-free for the decode.
    """
    out = np.zeros(4 * n_lvl, dtype=np.float64)
    for i, p in enumerate(sorted(ask)[:n_lvl]):
        out[4 * i], out[4 * i + 1] = p, ask[p]
    for i, p in enumerate(sorted(bid, reverse=True)[:n_lvl]):
        out[4 * i + 2], out[4 * i + 3] = p, bid[p]
    return out


def deadband_sizes(gen: np.ndarray, seed_frame: np.ndarray | None,
                   kind: str, thr, deep_from: int = 0) -> np.ndarray:
    """Size-only deadband on generated frames, in the decoder's quantized (side, price) space.

    Returns a (T,40) book. Prices are the quantized generated prices; the price SET per frame is
    exactly the generated one. Only sizes at prices carried over from the previous emitted frame
    can change, and only by being replaced with that previous size.
    """
    prev = _maps(seed_frame) if seed_frame is not None else _maps(gen[0])
    out = np.zeros_like(gen, dtype=np.float64)
    for t in range(len(gen)):
        cur = _maps(gen[t])
        emit = []
        for s in (0, 1):
            pm, cm = prev[s], dict(cur[s])
            # rank from the touch: asks ascending, bids descending
            order = sorted(cm, reverse=bool(s))
            for rank, p in enumerate(order):
                if rank < deep_from or p not in pm:
                    continue
                old, new = pm[p], cm[p]
                d = abs(new - old)
                if kind == "rel":
                    band = thr * max(old, 1)
                elif kind == "abs":
                    band = thr
                else:                                   # "or"
                    band = max(thr[0] * max(old, 1), thr[1])
                if d < band:
                    cm[p] = old                          # carry the previous emitted size forward
            emit.append(cm)
        out[t] = _to_frame40(emit[0], emit[1], gen.shape[1] // 4)
        prev = (emit[0], emit[1])
    return out


def _decode_window(path: Path, settings: list[str], out_root: Path) -> dict:
    """Decode one window under every setting. Returns {setting: (n_msgs | None)}."""
    d = np.load(path)
    cond_b, cond_t = d["cond_book"].astype(np.float64), d["cond_time"].astype(np.float64)
    gen_b, gen_t = d["gen_book"].astype(np.float64), d["gen_time"].astype(np.float64)
    date, rid = path.stem.split("_rid")
    res = {}
    for name in settings:
        kind, thr, deep = SETTINGS[name]
        g = gen_b if kind is None else deadband_sizes(gen_b, cond_b[-1], kind, thr, deep)
        mg, bg = decode_book_sequence(np.concatenate([cond_b, g], axis=0),
                                      frame_times=np.concatenate([cond_t, gen_t]))
        keep = mg["time"].to_numpy() > cond_t[-1]
        mg, bg = mg[keep].reset_index(drop=True), np.asarray(bg)[keep]
        if len(mg) < 8:
            res[name] = None
            continue
        arm = out_root / name
        mg[["time", "event_type", "order_id", "size", "price", "direction"]].to_csv(
            arm / f"GOOG_{date}_message_real_id_{rid}_gen_id_0.csv", header=False, index=False)
        pd.DataFrame(bg).to_csv(
            arm / f"GOOG_{date}_orderbook_real_id_{rid}_gen_id_0.csv", header=False, index=False)
        res[name] = len(mg)
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", default=None,
                    help="dumped generated frames (*.npz) from gen_bookdit_on_lobbench_windows.py "
                         "--dump-frames; default $BOOKDIT_WORK/refine_frames")
    ap.add_argument("--days", nargs="+", default=["2023-01-03"],
                    help="YYYY-MM-DD list, or 'all' for every dumped day")
    ap.add_argument("--settings", nargs="+", default=["raw", "rel010"],
                    help=f"one or more of: {' '.join(SETTINGS)}")
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--workers", type=int, default=22)
    ap.add_argument("--limit", type=int, default=0, help="debug: only this many windows")
    a = ap.parse_args()
    for s in a.settings:
        if s not in SETTINGS:
            raise SystemExit(f"unknown setting {s}; known: {' '.join(SETTINGS)}")

    frames = Path(a.frames) if a.frames else bench_work() / "refine_frames"
    files = sorted(frames.glob("*.npz")) if a.days == ["all"] else \
        sorted(f for d in a.days for f in frames.glob(f"{d}_*.npz"))
    if a.limit:
        files = files[:a.limit]
    if not files:
        raise SystemExit(f"no frame files matched under {frames} for --days {' '.join(a.days)}")
    out_root = Path(a.out_root)
    for s in a.settings:
        (out_root / s).mkdir(parents=True, exist_ok=True)
    print(f"[deadband] {len(files)} windows x {len(a.settings)} settings -> {out_root}", flush=True)

    fn = partial(_decode_window, settings=a.settings, out_root=out_root)
    counts = {s: [] for s in a.settings}
    with Pool(a.workers) as pool:
        for i, res in enumerate(pool.imap_unordered(fn, files, chunksize=4)):
            for s, n in res.items():
                counts[s].append(n)
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(files)}", flush=True)

    n_gen_frames = 50
    rows = []
    for s in a.settings:
        v = [x for x in counts[s] if x is not None]
        rows.append({"setting": s, "kind": str(SETTINGS[s][0]), "thr": str(SETTINGS[s][1]),
                     "deep_from": SETTINGS[s][2], "n_windows": len(v),
                     "n_dropped": len(counts[s]) - len(v),
                     "msgs_per_window": float(np.mean(v)),
                     "msgs_per_frame": float(np.mean(v)) / n_gen_frames})
    df = pd.DataFrame(rows)
    df.to_csv(out_root / "msg_counts.csv", index=False)
    print("\n" + df.to_string(index=False))
    print(f"\n[wrote] {out_root}/msg_counts.csv")


if __name__ == "__main__":
    main()
