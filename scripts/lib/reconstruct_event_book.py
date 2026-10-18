#!/usr/bin/env python
"""Reconstruct the EVENT-LEVEL limit order book by replaying LOBSTER messages.

A message stream that contains every event at every price fully determines the *visible* book,
so a replay of it is exact. This script replays a day's messages to emit the contiguous
event-level snapshots the corpus is built from (`emit_contiguous`; only the message file is
needed), and can compare the replay with LOBSTER's own orderbook file for the same day
(`--validate`).

REQUIRED INPUT.  The replay starts from an empty book, so it is exact only for a FULL-DEPTH
message file (every price level) that covers the WHOLE SESSION from the 04:00 pre-market open, one
file per ticker-day, as exported by an institutional LOBSTER server. A standard LOBSTER download
(09:30-16:00, level N) does not qualify. It carries only the events inside its requested levels
and none before the open, so an order that rests deeper, or was entered pre-market, and later
moves into view has no submission message, and the replay of such a file drifts from its
orderbook file. On two standard level-10 days the replay matched LOBSTER's orderbook file on
under 0.5% of rows. `--validate` reports how often the two agree; it does not assume they must.

Message cols (LOBSTER): time, type(1 sub / 2 partial-cancel / 3 delete / 4 visible-exec /
5 hidden-exec / 6 cross / 7 halt), orderID, size, price, direction(1=buy/bid side, -1=sell/ask
side; same for exec). `load_msgs` reads columns 1, 2, 4, 5, 6 only, so the order id and any extra
trailing column (the full-session files carry a seventh, e.g. `null` or a participant id) are
ignored. Types 5, 6 and 7 leave the visible book unchanged (`Book.apply`). Every row, pre-open rows
included, is replayed; `emit_contiguous` keeps only frames inside market hours.
Book vector layout: [ask_px, ask_sz, bid_px, bid_sz] x N levels, as in LOBSTER's orderbook file.

Files are read from `$LOBSTER_CACHE/csv/<day>/<ticker>_message.csv` (and, for `--validate`,
`<ticker>_orderbook.csv` next to it; `--orderbook` overrides that path).

Usage:
  python scripts/lib/reconstruct_event_book.py --day 20240304 --validate
  python scripts/lib/reconstruct_event_book.py --day 20240304 --emit-contiguous --stride 10 --out <dir>
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sortedcontainers import SortedList

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import lobster_cache  # noqa: E402

# Rooted at $LOBSTER_CACHE (.env; see .env.example and scripts/lib/_paths.py). Resolved LAZILY:
# importing this module (for `Book`, `MKT_OPEN`, ...) must not require the cache to be configured.
def csv_dir() -> str:
    return str(lobster_cache() / "csv")


LEVELS = 10
MKT_OPEN, MKT_CLOSE = 34200, 57600   # 09:30:00 .. 16:00:00 (seconds after midnight)


class Book:
    __slots__ = ("bpx", "apx", "bsz", "asz")

    def __init__(self):
        self.bpx = SortedList()   # bid prices ascending (best = last)
        self.apx = SortedList()   # ask prices ascending (best = first)
        self.bsz = {}             # price -> size
        self.asz = {}

    def apply(self, typ, size, price, direction):
        if typ == 5 or typ == 7 or typ == 6:      # hidden exec / halt / cross: no visible change
            return
        if direction == 1:
            szmap, plist = self.bsz, self.bpx
        else:
            szmap, plist = self.asz, self.apx
        if typ == 1:                              # new limit order
            if price in szmap:
                szmap[price] += size
            else:
                szmap[price] = size; plist.add(price)
        elif typ == 2 or typ == 3 or typ == 4:    # cancel / delete / execution -> reduce
            if price in szmap:
                szmap[price] -= size
                if szmap[price] <= 0:
                    del szmap[price]; plist.remove(price)

    def snapshot(self, n=LEVELS):
        v = np.zeros(4 * n, dtype=np.float64)
        ap = self.apx[:n]
        bp = self.bpx[-n:][::-1]
        for i, p in enumerate(ap):
            v[4 * i] = p; v[4 * i + 1] = self.asz[p]
        for i, p in enumerate(bp):
            v[4 * i + 2] = p; v[4 * i + 3] = self.bsz[p]
        return v


def load_msgs(day, ticker="GOOG"):
    df = pd.read_csv(f"{csv_dir()}/{day}/{ticker}_message.csv", header=None, usecols=[0, 1, 3, 4, 5],
                     names=["t", "type", "size", "price", "dir"])
    return (df["t"].to_numpy(np.float64), df["type"].to_numpy(np.int64),
            df["size"].to_numpy(np.int64), df["price"].to_numpy(np.int64),
            df["dir"].to_numpy(np.int64))


# LOBSTER fills unpopulated levels with these dummy prices (and size 0); `Book.snapshot` uses 0.
LOBSTER_DUMMY_PRICE = 9_999_999_999


def validate(day, ticker="GOOG", orderbook=None):
    """Replay the day's messages and compare the book after EVERY message with LOBSTER's own
    orderbook file, whose row k is the visible book after message k. Returns the fraction of
    rows that match exactly over all levels, and the same over market hours only."""
    ob_path = Path(orderbook) if orderbook else Path(csv_dir()) / day / f"{ticker}_orderbook.csv"
    if not ob_path.exists():
        raise SystemExit(f"no orderbook file at {ob_path}; --validate compares the replay with "
                         "LOBSTER's orderbook file for the same day (pass --orderbook PATH)")
    ob = pd.read_csv(ob_path, header=None).to_numpy(np.int64)
    levels = ob.shape[1] // 4
    ob[np.abs(ob) == LOBSTER_DUMMY_PRICE] = 0
    t, typ, size, price, direction = load_msgs(day, ticker)
    if len(t) != len(ob):
        raise SystemExit(f"{ob_path.name}: {len(ob)} rows but {len(t)} messages; "
                         "the two files must come from the same LOBSTER request")
    bk = Book()
    recon = np.empty_like(ob)
    for k in range(len(t)):
        bk.apply(typ[k], size[k], price[k], direction[k])
        recon[k] = bk.snapshot(n=levels)
    row_ok = np.all(recon == ob, axis=1)
    hours = (t >= MKT_OPEN) & (t <= MKT_CLOSE)
    full_match = float(row_ok.mean())
    hours_match = float(row_ok[hours].mean()) if hours.any() else float("nan")
    print(f"[{day}] {ticker}: replay vs {ob_path.name}  ({len(t):,} messages, {levels} levels)")
    print(f"  exact match, every message : {full_match:.6f}")
    print(f"  exact match, market hours  : {hours_match:.6f}")
    bad = np.flatnonzero(~row_ok)
    if len(bad):
        i = int(bad[0])
        c = int(np.flatnonzero(recon[i] != ob[i])[0])
        name = ("ask_px", "ask_sz", "bid_px", "bid_sz")[c % 4]
        print(f"  first mismatch at message {i} (t={t[i]:.6f}, type {typ[i]}): "
              f"level {c // 4 + 1} {name} replay {recon[i, c]} vs file {ob[i, c]}")
    return full_match, hours_match


def frame_trade_aggregates(typ, size, direction, frame_raw_idx, zero_first=True):
    """Per-frame TRADE aggregates for an arbitrary frame grid over a raw message stream.

    `emit_contiguous` below computes these for the corpus, where the grid is `k % stride == 0`
    over the RAW stream.  The official LOB-Bench window builders use a different grid -- every
    `stride`-th message of the type-(1,2,3,4) market-hours FILTERED stream -- so they cannot reuse
    that loop, but a 125-channel model must be conditioned on aggregates computed the same WAY or
    it sees out-of-distribution inputs.  This function is that shared definition; keep it and
    `emit_contiguous` in agreement.

    `frame_raw_idx` is the RAW index of each frame's terminating message (for the filtered grid,
    `F[pos]`).  Frame i aggregates the raw messages in `(frame_raw_idx[i-1], frame_raw_idx[i]]` --
    which is why raw indices are required rather than filtered ones: type-5 hidden executions are
    absent from the filtered stream but must still be counted.

    Row 0 is zeroed by default, matching `Δt[0]=0` in `book_to_midspread_dt`: neither quantity is
    defined without a preceding frame, and the encoders applied to a carved window zero the first
    Δt the same way.

    Returns (n_frames, 4) int64 with TRADE_NAMES column order
    (buy_vol, sell_vol, n_trades, hidden_vol).
    """
    idx = np.asarray(frame_raw_idx, dtype=np.int64)
    if idx.ndim != 1 or len(idx) < 1:
        raise ValueError(f"frame_raw_idx must be 1-D and non-empty, got shape {idx.shape}")
    if np.any(np.diff(idx) <= 0):
        raise ValueError("frame_raw_idx must be strictly increasing")
    typ = np.asarray(typ); size = np.asarray(size); direction = np.asarray(direction)

    out = np.zeros((len(idx), 4), dtype=np.int64)
    vis = typ == 4
    hid = typ == 5
    # cumulative sums let every frame interval be read off in O(1) instead of re-scanning
    cum_buy = np.concatenate([[0], np.cumsum(np.where(vis & (direction == -1), size, 0))])
    cum_sell = np.concatenate([[0], np.cumsum(np.where(vis & (direction != -1), size, 0))])
    cum_n = np.concatenate([[0], np.cumsum(vis.astype(np.int64))])
    cum_hid = np.concatenate([[0], np.cumsum(np.where(hid, size, 0))])
    lo, hi = idx[:-1] + 1, idx[1:] + 1          # half-open [lo, hi) == (prev, this]
    for col, cum in enumerate((cum_buy, cum_sell, cum_n, cum_hid)):
        out[1:, col] = cum[hi] - cum[lo]
    if not zero_first:
        raise NotImplementedError("frame 0 has no preceding frame in a carved window")
    return out


def emit_contiguous(day, out, ticker="GOOG", stride=10, levels=LEVELS):
    """Contiguous per-day artifact for the streaming loader (the event-level corpus): the full
    unwindowed snapshot stream (one snapshot after every `stride`-th message) + aligned per-event
    times in ONE file — books int32 (exact: prices are dollars*1e4, sizes are share counts), times
    float64 seconds-after-midnight. Windows are carved at load time, so nothing is stored twice.

    Also emits `trades` (T,4) int32 — the TRADE aggregates of the messages that produced each
    frame, i.e. of messages (prev_snapshot_index, this_snapshot_index]:

        col 0  buy_vol    type-4 shares executed against a resting ASK  (buyer-initiated)
        col 1  sell_vol   type-4 shares executed against a resting BID  (seller-initiated)
        col 2  n_trades   number of type-4 messages in the frame
        col 3  hidden_vol type-5 (hidden execution) shares -- invisible to the visible ladder

    Row 0 is all-zero (no preceding snapshot), matching the Δt convention. These are the raw
    counts consumed by `lob_representation.book_to_midspread_dt_logvol_trades`; readers that
    only want books/times are unaffected. `has_trades` flags the new format.
    """
    import torch
    t, typ, size, price, direction = load_msgs(day, ticker)
    bk = Book()
    rows, times, trades = [], [], []
    acc = np.zeros(4, dtype=np.int64)
    for k in range(len(t)):
        bk.apply(typ[k], size[k], price[k], direction[k])
        if t[k] < MKT_OPEN or t[k] > MKT_CLOSE:
            continue
        if typ[k] == 4:                       # visible execution of a resting order
            acc[0 if direction[k] == -1 else 1] += size[k]
            acc[2] += 1
        elif typ[k] == 5:                     # hidden execution: no visible book change
            acc[3] += size[k]
        if k % stride == 0:
            rows.append(bk.snapshot(n=levels))
            times.append(t[k])
            trades.append(acc.copy())
            acc[:] = 0
    if len(rows) < 1000:
        print(f"[{day}] {ticker}: too few snapshots ({len(rows)}) — skipped"); return None
    books = np.asarray(rows, dtype=np.int64)
    tr = np.asarray(trades, dtype=np.int64)
    tr[0] = 0                                 # frame 0 has no preceding snapshot (cf. Δt[0]=0)
    assert books.max() < 2**31 and books.min() >= 0, f"{ticker} {day}: values exceed int32"
    assert tr.max() < 2**31 and tr.min() >= 0, f"{ticker} {day}: trade counts exceed int32"
    outp = Path(out); outp.mkdir(parents=True, exist_ok=True)
    suffix = "" if levels == LEVELS else f"_L{levels}"
    fn = outp / f"{ticker}_{day}_event{stride}{suffix}_contig.pt"
    torch.save({"books": torch.from_numpy(books.astype(np.int32)),
                "times": torch.from_numpy(np.asarray(times, dtype=np.float64)),
                "trades": torch.from_numpy(tr.astype(np.int32)),
                "has_trades": True}, fn)
    print(f"[{day}] {ticker}: contig ({len(rows):,},{4*levels}) +trades -> {fn}")
    return fn


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--day", required=True)
    ap.add_argument("--ticker", default="GOOG")
    ap.add_argument("--validate", action="store_true",
                    help="compare the replay with LOBSTER's orderbook file, message by message")
    ap.add_argument("--orderbook", default=None,
                    help="orderbook file for --validate (default: <csv dir>/<day>/<ticker>_orderbook.csv)")
    ap.add_argument("--emit-contiguous", action="store_true")
    ap.add_argument("--stride", type=int, default=10, help="messages per snapshot (paper: 10)")
    ap.add_argument("--levels", type=int, default=LEVELS, help="book levels per snapshot (10 -> 40 cols)")
    ap.add_argument("--out", default=None, help="output dir (default: $LOBSTER_CACHE/tensors_event)")
    args = ap.parse_args()
    if args.validate:
        validate(args.day, args.ticker, args.orderbook)
    if args.emit_contiguous:
        if args.out is None:
            args.out = str(lobster_cache() / "tensors_event")
        emit_contiguous(args.day, args.out, args.ticker, args.stride, args.levels)


if __name__ == "__main__":
    main()
