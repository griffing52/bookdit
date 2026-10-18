#!/usr/bin/env python
"""Build the event-level training corpus: extract every raw archive under $LOBSTER_RAW_ROOT,
replay the matching engine, and emit CONTIGUOUS per-day artifacts (books int32 + aligned times
float64, `emit_contiguous`) for the streaming loader. Parallel over processes, resumable
(skips existing outputs), deletes CSVs it extracted (keeps pre-existing ones).

INPUT FORMAT.  Each archive `$LOBSTER_RAW_ROOT/<YYYYMMDD>/<TICKER>.7z` must hold that
ticker-day's FULL-DEPTH message file (every price level) covering the WHOLE SESSION from the 04:00
pre-market open, as exported by an institutional LOBSTER server, with the LOBSTER message columns
(time, type, order id, size, price, direction; an extra trailing column is ignored). The replay
starts from an empty book, so a standard 09:30-16:00 level-N LOBSTER download, which omits deeper
and pre-open orders, gives wrong books. Check one day with
`scripts/lib/reconstruct_event_book.py --validate` first (see that module's docstring and
docs/REPRODUCE.md section 1.1).

Also writes the corpus manifest (one row per ticker-day) with leak-safe splits:
  - test  : GOOG 20241112..20241126
  - val   : GOOG 20241108/20241111 (benchmark validation days) + 20241230/20241231, all names
  - train : everything else
"""
from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))

from _paths import contig_root, lobster_cache, lobster_raw_root, scratch_dir  # noqa: E402

# Machine-specific roots come from .env -- see .env.example and scripts/lib/_paths.py.
# LOBSTER_RAW_ROOT (.7z archives), LOBSTER_CACHE (csv/), LOBSTER_CONTIG_ROOT (output), plus an
# optional fast scratch for 7z extraction (None -> system temp). Resolved on FIRST USE, not at
# import, so `--help` and any importer work without a configured .env; the workers re-resolve from
# their inherited environment.
@lru_cache(maxsize=None)
def roots():
    """(raw .7z root, csv dir, contiguous-output root, scratch or None)."""
    return lobster_raw_root(), lobster_cache() / "csv", contig_root(), scratch_dir()


GOOG_TEST = {f"202411{d}" for d in ("12", "13", "14", "15", "18", "19", "20", "21", "22", "25", "26")}
GOOG_VAL = {"20241108", "20241111"}
GLOBAL_VAL = {"20241230", "20241231"}


def split_for(ticker: str, day: str) -> str:
    if ticker == "GOOG" and day in GOOG_TEST:
        return "test"
    if (ticker == "GOOG" and day in GOOG_VAL) or day in GLOBAL_VAL:
        return "val"
    return "train"


def process_one(arc: str, stride: int) -> str:
    from reconstruct_event_book import emit_contiguous
    _, csv_dir, out, scratch = roots()
    arc = Path(arc)
    day, ticker = arc.parent.name, arc.stem
    out_fn = out / f"{ticker}_{day}_event{stride}_contig.pt"
    if out_fn.exists():
        return f"SKIP {ticker} {day}"
    csv_fn = csv_dir / day / f"{ticker}_message.csv"
    we_extracted = False
    try:
        if not csv_fn.exists():
            csv_fn.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(dir=str(scratch) if scratch else None) as tmp:
                subprocess.run(["7z", "x", f"-o{tmp}", str(arc)], check=True,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                srcs = list(Path(tmp).glob("*message*.csv"))
                if not srcs:
                    return f"NOMSG {ticker} {day}"
                shutil.move(str(srcs[0]), csv_fn)
                we_extracted = True
        emit_contiguous(day, out, ticker, stride)
        return f"OK {ticker} {day}"
    except Exception as e:  # noqa: BLE001
        return f"FAIL {ticker} {day} {type(e).__name__}: {str(e)[:80]}"
    finally:
        if we_extracted and csv_fn.exists():
            csv_fn.unlink()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--manifest-out", default="runs/crossasset/corpus_manifest_2024.csv")
    ap.add_argument("--limit", type=int, default=0, help="debug: only first N archives")
    args = ap.parse_args()

    raw, _, out, _ = roots()
    archives = sorted(str(p) for p in raw.glob("*/[A-Z]*.7z"))
    if args.limit:
        archives = archives[: args.limit]
    out.mkdir(parents=True, exist_ok=True)
    # create the manifest's directory up front, so a bad path fails before hours of extraction
    Path(args.manifest_out).parent.mkdir(parents=True, exist_ok=True)
    print(f"{len(archives)} archives, {args.workers} workers -> {out}")

    counts = {"OK": 0, "SKIP": 0, "FAIL": 0, "NOMSG": 0}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(process_one, a, args.stride) for a in archives]
        for i, f in enumerate(as_completed(futs)):
            msg = f.result()
            counts[msg.split()[0]] = counts.get(msg.split()[0], 0) + 1
            if msg.startswith(("FAIL", "NOMSG")):
                print(msg, flush=True)
            if (i + 1) % 100 == 0:
                print(f"progress {i+1}/{len(archives)}  {counts}", flush=True)
    print(f"DONE {counts}")

    hdr = ["ticker", "day", "year", "split", "tensor_path", "status"]
    rows = []
    for p in sorted(out.glob(f"*_event{args.stride}_contig.pt")):
        ticker, day = p.name.split("_")[0], p.name.split("_")[1]
        rows.append({"ticker": ticker, "day": day, "year": day[:4],
                     "split": split_for(ticker, day), "tensor_path": str(p), "status": "ready"})
    with open(args.manifest_out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=hdr)
        w.writeheader()
        w.writerows(rows)
    from collections import Counter
    print(f"manifest: {args.manifest_out}  {Counter(r['split'] for r in rows)}")


if __name__ == "__main__":
    main()
