#!/usr/bin/env python
"""Rebuild LOB-Bench data_real/data_cond for the HuggingFace lob_bench eval windows.

The HF release (peernagy/lob_bench) ships only data_gen (LOBSTER licensing), so the real
counterparts must be rebuilt locally: align each generated window to the real message stream by
its first timestamp, then replay the LOBSTER messages to snapshot the 10-level book.
"""
from __future__ import annotations
import argparse, glob, re, sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from _paths import lobster_cache
from reconstruct_event_book import Book  # validated LOBSTER message replay

MKT_OPEN, MKT_CLOSE = 34200, 57600
KEEP_TYPES = (1, 2, 3, 4)          # visible-book events only (matches LOB-Bench real files)


def build_day(day, date_str, gen_dir, out_root, ctx=500, horizon=500):
    csv = lobster_cache() / "csv"          # resolved per call, never at import
    msg = pd.read_csv(f"{csv}/{day}/GOOG_message.csv", header=None, usecols=range(6),
                      names=["t", "type", "oid", "size", "price", "dir"])
    t_all = msg.t.to_numpy(np.float64)
    typ_all = msg.type.to_numpy(np.int64)
    keep = np.isin(typ_all, KEEP_TYPES) & (t_all >= MKT_OPEN) & (t_all <= MKT_CLOSE)
    F = np.flatnonzero(keep)                      # positions in the raw stream
    tF = t_all[F]
    print(f"[{day}] raw={len(msg):,} filtered={len(F):,}")

    gens = sorted(glob.glob(f"{gen_dir}/*{date_str}*message*_gen_id_*.csv"))
    windows = []
    for f in gens:
        rid = int(re.search(r"real_id_(\d+)_", f).group(1))
        t0 = float(pd.read_csv(f, header=None, nrows=1).iloc[0, 0])
        p = int(np.searchsorted(tF, t0))
        if p - ctx < 0 or p + horizon > len(F):
            continue
        windows.append((rid, p))
    print(f"[{day}] {len(windows)} windows aligned")

    needed = set()
    for _, p in windows:
        needed.update(F[p - ctx:p + horizon].tolist())

    bk = Book()
    snaps = {}
    size_a = msg["size"].to_numpy(np.int64); px = msg.price.to_numpy(np.int64)
    dr = msg["dir"].to_numpy(np.int64)
    for k in range(len(t_all)):
        bk.apply(typ_all[k], size_a[k], px[k], dr[k])
        if k in needed:
            snaps[k] = bk.snapshot()
    print(f"[{day}] replayed, {len(snaps):,} snapshots")

    out = Path(out_root) / "GOOG"
    for sub in ("data_real", "data_cond"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    cols = ["t", "type", "oid", "size", "price", "dir"]
    for rid, p in windows:
        for sub, idx in (("data_cond", F[p - ctx:p]), ("data_real", F[p:p + horizon])):
            m = msg.iloc[idx][cols]
            b = np.stack([snaps[k] for k in idx]).astype(np.int64)
            m.to_csv(out / sub / f"GOOG_{date_str}_message_real_id_{rid}.csv", header=False, index=False)
            pd.DataFrame(b).to_csv(out / sub / f"GOOG_{date_str}_orderbook_real_id_{rid}.csv",
                                   header=False, index=False)
    return len(windows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", nargs="+", default=["20230103"])
    ap.add_argument("--gen_dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    tot = 0
    for day in a.days:
        ds = f"{day[:4]}-{day[4:6]}-{day[6:]}"
        tot += build_day(day, ds, a.gen_dir, a.out)
    print("TOTAL windows:", tot)
