#!/usr/bin/env python
"""Regenerate the toy LOBSTER fixtures shipped in ``quickstart/toy_data/``.

These are synthetic random walks, not market data. Their only job is to exercise the
local-CSV LOBSTER path end to end, so the quickstart runs on any laptop.

Two properties of the output are load-bearing:

1. Every filename stem marks the file as a book or a message file -- that is what
   ``discover_lobster_pairs`` keys on (``src/bookdit/data/lobster.py``).
2. There are **two** days. ``split_lobster_pairs`` reserves the last pair for
   validation and raises ``ValueError: Validation fraction leaves no training pairs``
   on a single day (``split_lobster_pairs`` in the same file).

Column layout matches real LOBSTER level-10 exports:

* orderbook: 40 integer columns, ``[ask_price, ask_size, bid_price, bid_size] x 10``,
  prices in 1/10000 of a dollar.
* message: 6 columns, ``time, event_type, order_id, size, price, direction``.

Usage::

    uv run --no-sync python quickstart/make_toy_lobster.py
    uv run --no-sync python quickstart/make_toy_lobster.py --rows 3000 --out-dir /tmp/toy
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

DEFAULT_OUT = Path(__file__).resolve().parent / "toy_data"
DAYS = ["2020-01-02", "2020-01-03"]
# LOBSTER encodes the session window in the filename: 09:30:00 -> 34200000 ms,
# 16:00:00 -> 57600000 ms.
SESSION = "34200000_57600000"
LEVELS = 10
TICK = 100  # 1/10000 dollar units


def _make_day(rows: int, levels: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(orderbook, message)`` arrays for one synthetic day."""
    rng = np.random.default_rng(seed)

    # Mid price: a random walk on the tick grid, starting at $100.00.
    steps = rng.integers(-1, 2, size=rows)
    mid = 1_000_000 + np.cumsum(steps) * TICK

    book = np.empty((rows, 4 * levels), dtype=np.int64)
    for lv in range(levels):
        offset = (lv + 1) * TICK
        book[:, 4 * lv + 0] = mid + offset            # ask price
        book[:, 4 * lv + 1] = rng.integers(1, 500, rows)  # ask size
        book[:, 4 * lv + 2] = mid - offset            # bid price
        book[:, 4 * lv + 3] = rng.integers(1, 500, rows)  # bid size

    # Message file: monotone timestamps across the 6.5 h session.
    gaps = rng.exponential(1.0, size=rows)
    times = 34200.0 + np.cumsum(gaps) / gaps.sum() * (57600.0 - 34200.0 - 1.0)
    msg = np.empty((rows, 6), dtype=object)
    msg[:, 0] = times
    msg[:, 1] = rng.integers(1, 5, rows)              # event type 1..4
    msg[:, 2] = rng.integers(1, 1_000_000, rows)      # order id
    msg[:, 3] = rng.integers(1, 500, rows)            # size
    msg[:, 4] = mid + TICK * rng.choice([-1, 1], rows)  # price
    msg[:, 5] = rng.choice([-1, 1], rows)             # direction
    return book, msg


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT,
                    help="where to write the CSVs (default: quickstart/toy_data)")
    ap.add_argument("--rows", type=int, default=1500, help="rows per day (default 1500)")
    ap.add_argument("--levels", type=int, default=LEVELS, help="book levels (default 10)")
    ap.add_argument("--prefix", default="TOY", help="ticker stand-in (default TOY)")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for seed, day in enumerate(DAYS):
        book, msg = _make_day(args.rows, args.levels, seed)
        stem = f"{args.prefix}_{day}_{SESSION}"
        ob_path = args.out_dir / f"{stem}_orderbook_{args.levels}.csv"
        msg_path = args.out_dir / f"{stem}_message_{args.levels}.csv"
        np.savetxt(ob_path, book, fmt="%d", delimiter=",")
        with open(msg_path, "w") as fh:
            for row in msg:
                fh.write(f"{row[0]:.9f},{row[1]},{row[2]},{row[3]},{row[4]},{row[5]}\n")
        print(f"wrote {ob_path}  ({args.rows} x {4 * args.levels})")
        print(f"wrote {msg_path}  ({args.rows} x 6)")

    print(f"\n{len(DAYS)} day-pairs in {args.out_dir} -- now run quickstart/train_toy.py.")


if __name__ == "__main__":
    main()
