#!/usr/bin/env python
"""Inverse-dynamics message annotator (IDM): the order-level rules behind the book -> messages decoder.

A book-state world model emits book snapshots but no messages, so it cannot produce LOB-Bench's
message / order-lifecycle metrics (inter_arrival, time_to_cancel, time_to_first_fill,
limit_order_depth, limit_order_levels). This module derives an inverse map

    g : (book_t, book_{t+1})  ->  (event_type, side, price, size, dt) + a synthetic order_id

at event resolution (one book diff ~= one message). It is rule-based plus a FIFO queue tracker:
the book delta determines (side, price, |size|, add-vs-reduce) outright, and the residual
ambiguities (event_type sub-classification, order_id linkage, dt) are resolved by explicit rules
and price-time-priority queue matching.

This module holds the FIFO queue tracker (`QueueBook`) and the reduction-type rule
(`classify_reduction_type`) that the frame-level decoder `idm_decoder.decode_book_sequence` uses,
plus `annotate`, an event-resolution replay of real messages used only for validation.

Validation lives in `scripts/lib/idm_validate.py`: it replays real held-out GOOG, runs g on the
resulting book sequence, and scores the recovered message stream against the true messages with
LOB-Bench's own message metrics; the real-vs-real floor per metric bounds what the inverse map
itself costs. Running this file directly forwards to that validator:

  python scripts/lib/idm_validate.py --day 20240701 --max-events 300000
  python scripts/lib/idm_validate.py --day 20240701 --max-events 300000 --true-type   # oracle-type ablation
"""
from __future__ import annotations

import sys
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
from sortedcontainers import SortedList

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from _paths import lob_bench_dir, lobster_cache  # noqa: E402

sys.path.insert(0, str(lob_bench_dir()))          # $LOB_BENCH_DIR, default <repo>/libs/lob_bench

from reconstruct_event_book import MKT_OPEN, MKT_CLOSE, LEVELS  # noqa: E402


# Extracted LOBSTER message CSVs, rooted at $LOBSTER_CACHE (.env; scripts/lib/_paths.py).
# Resolved LAZILY: importing `QueueBook` & co. must not require the cache to be configured.
def csv_dir() -> str:
    return str(lobster_cache() / "csv")


def load_msgs_full(day, ticker="GOOG"):
    """Like reconstruct_event_book.load_msgs but ALSO returns the real order_id (col 2)."""
    df = pd.read_csv(f"{csv_dir()}/{day}/{ticker}_message.csv", header=None, usecols=[0, 1, 2, 3, 4, 5],
                     names=["t", "type", "oid", "size", "price", "dir"])
    return (df["t"].to_numpy(np.float64), df["type"].to_numpy(np.int64),
            df["oid"].to_numpy(np.int64), df["size"].to_numpy(np.int64),
            df["price"].to_numpy(np.int64), df["dir"].to_numpy(np.int64))


# --------------------------------------------------------------------------------------
# Full-depth book + FIFO price-time-priority queue tracker with SYNTHETIC order ids.
# --------------------------------------------------------------------------------------
class QueueBook:
    """Replays messages into a full-depth book AND maintains, per (side, price), a FIFO queue of
    resting orders with fabricated ids. The queue is the engine of the inverse: it re-derives an
    order_id for cancels/executions from price-time priority (the matching engine's own rule).
    """

    def __init__(self):
        self.bpx = SortedList()      # bid prices ascending (best = last)
        self.apx = SortedList()      # ask prices ascending (best = first)
        self.bsz: dict[int, int] = {}
        self.asz: dict[int, int] = {}
        self.bq: dict[int, deque] = {}   # price -> deque of [oid, remaining]
        self.aq: dict[int, deque] = {}
        self._next_oid = 1

    # ---- book views ------------------------------------------------------------------
    def best_bid(self):
        return self.bpx[-1] if self.bpx else None

    def best_ask(self):
        return self.apx[0] if self.apx else None

    def snapshot(self, n=LEVELS):
        """40-dim LOBSTER vector [ask_px,ask_sz,bid_px,bid_sz] x n (state AFTER last applied event)."""
        v = np.zeros(4 * n, dtype=np.int64)
        for i, p in enumerate(self.apx[:n]):
            v[4 * i] = p
            v[4 * i + 1] = self.asz[p]
        for i, p in enumerate(self.bpx[-n:][::-1]):
            v[4 * i + 2] = p
            v[4 * i + 3] = self.bsz[p]
        return v

    def level_of(self, side, price):
        """1-indexed level of `price` on `side` in the current book, or None if not present/visible."""
        if side == 1:
            lst = self.bpx[-LEVELS:][::-1]
        else:
            lst = self.apx[:LEVELS]
        for i, p in enumerate(lst):
            if p == price:
                return i + 1
        return None

    # ---- mutation + inverse decode ---------------------------------------------------
    def add_limit(self, side, price, size):
        szmap, plist, q = (self.bsz, self.bpx, self.bq) if side == 1 else (self.asz, self.apx, self.aq)
        if price in szmap:
            szmap[price] += size
        else:
            szmap[price] = size
            plist.add(price)
            q[price] = deque()
        oid = self._next_oid
        self._next_oid += 1
        q[price].append([oid, size])
        return oid

    def reduce(self, side, price, size, kind):
        """Reduce resting liquidity at (side, price) by `size`. `kind` in {"exec","cancel"} chooses the
        queue-matching rule. Returns the synthetic order_id charged with the reduction.

        - exec:  price-time priority -> consume from the FRONT (oldest) resting order.
        - cancel: a delete/partial-cancel targets a SPECIFIC order the book can't identify; we match
                  by size first (a full delete's size == a resting order's full size, a strong signal),
                  else fall back to the oldest order (front).

        Known simplification, kept because the reported numbers were computed with it: a cancel that
        matches no order by size and is larger than the front order removes only the front order.
        The excess is not taken from later orders, so the FIFO queue can hold more shares than the
        level's aggregate size until that level empties. For example, orders of 10 and 10 at one
        price followed by a 15-share cancel leave a level size of 5 but a queue holding the second
        order's 10. Aggregate sizes, and therefore book snapshots, are unaffected; what changes is
        which synthetic order id later reductions are charged to, i.e. the order-lifecycle
        statistics (time_to_cancel and the like). Taking the remainder from later orders, as the
        exec branch does, would change those statistics, so re-score after any such change.
        """
        szmap, plist, q = (self.bsz, self.bpx, self.bq) if side == 1 else (self.asz, self.apx, self.aq)
        if price not in szmap:
            return None
        dq = q[price]
        oid = None
        if kind == "cancel":
            # try exact size match (full-order delete) anywhere in the queue
            hit = next((idx for idx, (_, rem) in enumerate(dq) if rem == size), None)
            if hit is not None:
                oid = dq[hit][0]
                del dq[hit]
            elif dq:                      # partial cancel -> take from front
                oid = dq[0][0]
                dq[0][1] -= size
                if dq[0][1] <= 0:
                    dq.popleft()
        else:                             # exec: FIFO from front, possibly across orders
            remaining = size
            while remaining > 0 and dq:
                if oid is None:
                    oid = dq[0][0]
                if dq[0][1] <= remaining:
                    remaining -= dq[0][1]
                    dq.popleft()
                else:
                    dq[0][1] -= remaining
                    remaining = 0
        # update aggregate size / remove empty level
        szmap[price] -= size
        if szmap[price] <= 0:
            del szmap[price]
            plist.remove(price)
            q.pop(price, None)
        return oid


# --------------------------------------------------------------------------------------
# The inverse-dynamics decoder g.
# --------------------------------------------------------------------------------------
def _delete_or_partial(dq, size):
    """Delete(3) vs partial-cancel(2) over a FIFO queue `dq` of [oid, remaining]:
    a reduction matching some whole resting order removes an order, anything else is a partial."""
    if dq is None or len(dq) == 0:
        return 2
    return 3 if any(rem == size for _, rem in dq) else 2


def classify_reduction_type(side, price, book: QueueBook, size):
    """Book-only classification of a size-*reduction* into an event_type.

    Fundamental ambiguity: from two book snapshots a cancel (2/3) and a visible execution (4) are
    indistinguishable at the touch; the only book-derivable signal is depth, since an execution can
    only occur at the current touch (an aggressor sweeps from the best price).

    The split between delete(3) and partial-cancel(2) is an order-level question, not a level-level
    one: a delete removes one whole resting order, which rarely empties a price level holding many
    orders, so keying it off "does the level clear" would label almost every real delete a partial
    cancel.  We ask the FIFO queue whether the reduction matches a whole resting order (the same exact-size
    test `QueueBook.reduce` already uses to pick the victim), and reserve execution for a touch
    reduction that clears the touch level -- the sweep signature. Touch *cancels* that clear a
    single-order level are still mislabelled executions; that error is irreducible from book states.
    """
    touch = book.best_bid() if side == 1 else book.best_ask()
    szmap = book.bsz if side == 1 else book.asz
    if price == touch and szmap.get(price, 0) - size <= 0:
        return 4                          # sweep: touch level fully consumed -> visible execution
    dq = (book.bq if side == 1 else book.aq).get(price)
    return _delete_or_partial(dq, size)


def annotate(day, ticker="GOOG", max_events=None, use_true_type=False, carry_true_dt=True):
    """Replay real GOOG, decode each visible book transition into a recovered message, and also
    collect the true messages + aligned books. Returns (rec_msgs, rec_book, true_msgs, true_book, stats).

    rec_msgs / true_msgs : DataFrames [time, event_type, order_id, size, price, direction]
                           (time is float seconds-after-midnight; caller converts to datetime)
    rec_book  / true_book: DataFrames of 40 LOBSTER cols, row-aligned with their messages.

    The TRUE stream keeps the REAL order_id from the CSV, so its lifecycle metrics are the genuine
    ground truth we score the RECOVERED (synthetic-id) stream against.
    """
    t, typ, true_oid, size, price, direction = load_msgs_full(day, ticker)
    bk = QueueBook()

    rec_rows, rec_books = [], []
    true_rows, true_books = [], []
    n_seen = 0
    n_hidden = 0                 # type 5/6/7: invisible to the book -> unrecoverable
    n_deep = 0                   # reductions at level > LEVELS (outside the world-model's 10-level view)
    type_confusion = np.zeros((6, 6), dtype=np.int64)   # true_type x recovered_type

    for k in range(len(t)):
        if t[k] < MKT_OPEN or t[k] > MKT_CLOSE:
            continue
        et, sz, px, dr, toid = (int(typ[k]), int(size[k]), int(price[k]),
                                int(direction[k]), int(true_oid[k]))

        if et in (5, 6, 7):
            # hidden execution / cross / halt: no visible book change -> g emits NOTHING.
            # Still part of the true stream LOB-Bench sees (affects inter-arrival density only).
            n_hidden += 1
            true_rows.append((t[k], et, toid, sz, px, dr))
            true_books.append(bk.snapshot())
            continue

        lvl = bk.level_of(dr, px)              # level in book BEFORE the event (None if new/deep)
        is_add = (et == 1)

        if is_add:
            rec_type = 1
            oid = bk.add_limit(dr, px, sz)
        else:
            rec_type = et if use_true_type else classify_reduction_type(dr, px, bk, sz)
            kind = "exec" if rec_type == 4 else "cancel"
            oid = bk.reduce(dr, px, sz, kind)
            if oid is None:                    # reducing liquidity we never tracked (warm-up) -> skip
                true_rows.append((t[k], et, toid, sz, px, dr))
                true_books.append(bk.snapshot())
                continue

        snap = bk.snapshot()                   # book state AFTER the event (both streams share it)
        # Δt is NOT a book quantity -- two consecutive snapshots say nothing about how long the gap
        # between them was. For the floor we therefore carry the true event time,
        # isolating event_type/price/size/order_id recovery from timing. carry_true_dt=False would defer
        # timing to a generator cadence / timing head (not exercised in the floor run).
        rec_time = t[k]
        rec_rows.append((rec_time, rec_type, oid, sz, px, dr))
        rec_books.append(snap)
        true_rows.append((t[k], et, toid, sz, px, dr))
        true_books.append(snap)

        n_seen += 1
        type_confusion[et, rec_type] += 1
        if lvl is None and not is_add:
            n_deep += 1
        if max_events and n_seen >= max_events:
            break

    cols = ["time", "event_type", "order_id", "size", "price", "direction"]
    rec_msgs = pd.DataFrame(rec_rows, columns=cols)
    true_msgs = pd.DataFrame(true_rows, columns=cols)
    rec_book = pd.DataFrame(np.asarray(rec_books))
    true_book = pd.DataFrame(np.asarray(true_books))
    stats = dict(n_seen=n_seen, n_hidden=n_hidden, n_deep=n_deep, type_confusion=type_confusion)
    return rec_msgs, rec_book, true_msgs, true_book, stats


if __name__ == "__main__":
    from idm_validate import main
    main()
