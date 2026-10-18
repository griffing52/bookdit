#!/usr/bin/env python
"""The inverse-dynamics decoder: book snapshots -> LOBSTER messages.

`decode_book_sequence` diffs consecutive visible book frames into events, tracks order ids with
the FIFO `idm_annotator.QueueBook`, and applies the resting-size deadband and intra-frame gap
profile documented in `docs/PROTOCOL.md` §4.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from idm_annotator import (QueueBook, classify_reduction_type,                 # noqa: E402
                           _delete_or_partial)
from reconstruct_event_book import MKT_OPEN                                    # noqa: E402

TICK = 100   # GOOG LOBSTER price tick (== $0.01); real books are integer multiples of this.

# The standard decoder is IDM_GAP_PROFILE=1 IDM_DEADBAND_ABS=50, i.e. the two defaults below;
# the env vars remain as overrides. See docs/PROTOCOL.md §4.
#
# INTRA-FRAME GAP PROFILE. A frame's events are placed inside (t[i-1], t[i]] by sampling the
# frame's composition from the empirical marginal of real intra-frame gaps rather than by spacing
# them equally; real intra-frame timing is bursty, and `log_inter_arrival_time` is a marginal
# statistic, so reproducing the marginal is sufficient. The profile table ships with the repo
# (scripts/lib/intraframe_gap_profile.npz) and was fitted on GOOG 2024 message times, a different year
# from the 2023 official windows.
#
# RESTING-SIZE DEADBAND. Real book sizes are piecewise constant; an MSE objective produces small
# nonzero moves in every cell instead, so the decoder emits far more messages per frame than the
# real stream. A carry-forward deadband restores the sparsity: a level's size is held until the
# pending change exceeds the threshold, then the whole accumulated change is emitted at once, so
# suppressed displacement is conserved rather than discarded.
DEADBAND_ABS = float(os.environ.get("IDM_DEADBAND_ABS", "50"))


def _carry_forward_deadband(book, theta):
    """(T,4L) -> same, with size columns held until a pending change exceeds `theta` shares."""
    b = np.asarray(book, dtype=np.float64).copy()
    L = b.shape[1] // 4
    cols = [4 * l + c for l in range(L) for c in (1, 3)]
    held = b[0, cols].copy()
    for t in range(1, len(b)):
        pend = b[t, cols] - held
        held = np.where(np.abs(pend) >= theta, b[t, cols], held)
        b[t, cols] = held
    return b


GAP_PROFILE = os.environ.get("IDM_GAP_PROFILE", "1") == "1"
_GAP_TAB = None


def _gap_weights(n, rng):
    """`n` positive weights summing to 1, drawn from the real intra-frame gap marginal."""
    global _GAP_TAB
    if _GAP_TAB is None:
        _GAP_TAB = np.load(Path(__file__).resolve().parent / "intraframe_gap_profile.npz")["weights"]
    w = np.interp(rng.random(n), np.linspace(0.0, 1.0, len(_GAP_TAB)), _GAP_TAB)
    s = w.sum()
    return (w / s) if s > 0 else np.full(n, 1.0 / n)


# =================================================================================================
# The snapshot-diff inverse-dynamics decoder `g` (book sequence -> synthetic message stream).
# =================================================================================================
def _quantize(frame40, tick=TICK):
    """Raw 4L-vec -> quantized (L,4) [ask_px,ask_sz,bid_px,bid_sz]: prices snap to tick, sizes to int.
    L is inferred from the vector width, so 20-level (80-col) books decode too."""
    g = np.asarray(frame40, float).reshape(-1, 4).copy()
    g[:, 0] = np.round(g[:, 0] / tick) * tick     # ask_px
    g[:, 2] = np.round(g[:, 2] / tick) * tick     # bid_px
    g[:, 1] = np.maximum(np.round(g[:, 1]), 0)    # ask_sz
    g[:, 3] = np.maximum(np.round(g[:, 3]), 0)    # bid_sz
    return g


def _side_maps(g):
    """(L,4) quantized frame -> (ask price->size, bid price->size) dicts over visible levels."""
    ask, bid = {}, {}
    for i in range(len(g)):
        ap, asz, bp, bsz = g[i]
        if ap > 0 and asz > 0:
            ask[int(ap)] = ask.get(int(ap), 0) + int(asz)
        if bp > 0 and bsz > 0:
            bid[int(bp)] = bid.get(int(bp), 0) + int(bsz)
    return ask, bid


def _cancel_type(side, price, book, size):
    """The cancel half of `classify_reduction_type`, with execution excluded by construction.

    Needed because under trade-aware decoding the frame's executed volume is already fully
    accounted for by the allocation, so a leftover reduction CANNOT be an execution -- but
    `classify_reduction_type` would still label a touch-clearing one type 4 and double-count it.
    """
    dq = (book.bq if side == 1 else book.aq).get(price)
    return _delete_or_partial(dq, size)


def _allocate_executions(events, frame_trade_row):
    """Touch-outward allocation of a frame's executed volume over its net reductions.

    From book states alone an execution and a cancellation look the same, so the book-diff
    heuristic in `classify_reduction_type` can only guess. A model that also generates the frame's
    executed volume per side supplies the missing budget; spending it from the touch outward
    recovers per-frame per-side executed volume.

    `frame_trade_row` is (buy_vol, sell_vol, n_trades, hidden_vol) in RAW share units, where
    buy_vol is volume taken from resting ASKS (side -1) and sell_vol from resting BIDS (side +1).

    Returns {(side, price): executed_shares}.  A reduction larger than its remaining budget is
    split by the caller into an execution and a cancellation at the same price, which is what the
    real message stream does.  `n_trades` is not used as a second constraint: only the volume
    budget is validated, and an unvalidated message-count cap could conflict with it.
    """
    budget = {-1: float(frame_trade_row[0]), 1: float(frame_trade_row[1])}
    alloc = {}
    for side in (-1, 1):
        if budget[side] <= 0:
            continue
        # touch outward: ascending price on the ask side, descending on the bid side
        reds = sorted(((p, -d) for (s, p, d, *_) in events if s == side and d < 0),
                      key=lambda pd: pd[0] if side == -1 else -pd[0])
        for price, sz in reds:
            if budget[side] <= 0:
                break
            take = min(budget[side], float(sz))
            if take > 0:
                alloc[(side, price)] = int(round(take))
                budget[side] -= take
    return alloc


def decode_book_sequence(book, tick=TICK, base_t=MKT_OPEN, frame_times=None,
                         frame_trades=None, emit_hidden=False, deadband_from=0):
    """Inverse-dynamics `g` on a raw book window (T,40): diff consecutive visible books into events,
    apply the SAME classification rule + FIFO queue as `idm_annotator` to fabricate order_ids.

    Returns (rec_msgs, rec_book): a message DataFrame [time,event_type,order_id,size,price,direction]
    (side convention bid=+1/ask=-1, matching QueueBook) and the row-aligned book-AFTER-event frame.

    Timing: if `frame_times` (len T, seconds) is given — the true/generated snapshot times, e.g. from
    the midspread_dt Δt channel — the events of frame t are placed inside the real interval
    (frame_times[t-1], frame_times[t]], recovering the actual inter-arrival density. If it is None,
    one second per book frame is fabricated instead, applied identically to real and gen, which
    leaves the timing statistics near-degenerate.

    Trades: if `frame_trades` (T,4) is given — the true/generated per-frame trade aggregates in the
    TRADE_NAMES order (buy_vol, sell_vol, n_trades, hidden_vol), e.g. from a
    `midspread_dt_logvol_trades` model — a reduction is labelled an execution by spending that
    side's volume budget from the touch outward (`_allocate_executions`) instead of by the
    book-diff sweep heuristic. `emit_hidden` additionally fabricates type-5 messages carrying
    hidden_vol; it is off by default because those messages have no book effect but do move the
    event-conditioned inter-arrival and volume statistics.
    """
    if DEADBAND_ABS > 0:
        # only from `deadband_from` onward: an observed REAL conditioning prefix is already
        # sparse, and smoothing it would corrupt the warm queue the lifecycle stats rely on
        book = np.asarray(book, dtype=np.float64).copy()
        _k = int(deadband_from)
        book[_k:] = _carry_forward_deadband(book[_k:], DEADBAND_ABS)
    T = book.shape[0]
    if frame_times is not None:
        frame_times = np.asarray(frame_times, dtype=np.float64)
    if frame_trades is not None:
        frame_trades = np.asarray(frame_trades, dtype=np.float64)
        if frame_trades.shape != (T, 4):
            raise ValueError(f"frame_trades must have shape ({T}, 4), got {frame_trades.shape}")
        frame_trades = np.maximum(frame_trades, 0.0)
    bk = QueueBook()
    prev_ask, prev_bid = _side_maps(_quantize(book[0], tick))
    for p, s in prev_ask.items():
        bk.add_limit(-1, p, s)           # seed resting liquidity (no messages emitted for frame 0)
    for p, s in prev_bid.items():
        bk.add_limit(1, p, s)

    rec_rows, rec_books = [], []
    for t in range(1, T):
        cur_ask, cur_bid = _side_maps(_quantize(book[t], tick))
        events = []                       # (side, price, dsize)  dsize>0 add, <0 reduce
        for side, prev_m, cur_m in ((-1, prev_ask, cur_ask), (1, prev_bid, cur_bid)):
            for p in set(prev_m) | set(cur_m):
                d = cur_m.get(p, 0) - prev_m.get(p, 0)
                if d != 0:
                    events.append((side, p, d, None))     # 4th slot = forced event type
        if frame_trades is not None:
            # Label BEFORE ordering, so the emission order below is untouched. A reduction bigger
            # than its remaining budget splits into an execution and a cancellation at the same
            # price -- two messages, as the real stream would carry.
            #
            # Every reduction the budget does NOT cover is forced to a cancel (sentinel "c").
            # Falling back to `classify_reduction_type` here would let the book-diff heuristic add
            # its own executions on top of the budgeted ones; the frame's executed volume is the
            # budget, and nothing more.
            alloc = _allocate_executions(events, frame_trades[t])
            split = []
            for ev in events:
                side, p, d, _ = ev
                if d > 0:
                    split.append(ev)
                    continue
                ex = alloc.get((side, p), 0)
                if ex <= 0:
                    split.append((side, p, d, "c"))
                elif ex >= -d:
                    split.append((side, p, d, 4))
                else:
                    split.append((side, p, -ex, 4))               # executed part
                    split.append((side, p, d + ex, "c"))          # cancelled remainder
            events = split
        # reductions before adds so touch classification sees pre-add depth; deeper prices first
        events.sort(key=lambda e: (e[2] > 0, -e[1] if e[0] == -1 else e[1]))
        n_hidden = 0
        if emit_hidden and frame_trades is not None:
            n_hidden = int(round(frame_trades[t][3] > 0))     # one aggregate type-5 per frame
        n_ev = len(events) + n_hidden
        if frame_times is not None:
            t0, t1 = frame_times[t - 1], frame_times[t]      # true interval this frame's events fill
        else:
            t0, t1 = base_t + t, base_t + t + 1              # no times given: one second per frame
        # event offsets inside (t0, t1]: equal spacing by default, or a draw from the real
        # intra-frame gap marginal when IDM_GAP_PROFILE=1 (seeded per frame -> reproducible)
        if GAP_PROFILE and n_ev > 0:
            _off = np.cumsum(_gap_weights(n_ev + 1, np.random.default_rng(t)))[:n_ev]
        else:
            _off = None
        for j, (side, price, d, forced) in enumerate(events):
            frac = _off[j] if _off is not None else (j + 1) / (n_ev + 1)
            tt = t0 + frac * (t1 - t0)                    # spread events across the interval
            if d > 0:
                rec_type, sz = 1, int(d)
                oid = bk.add_limit(side, price, sz)
            else:
                sz = int(-d)
                if forced == "c":
                    rec_type = _cancel_type(side, price, bk, sz)
                elif forced is not None:
                    rec_type = forced
                else:
                    rec_type = classify_reduction_type(side, price, bk, sz)
                oid = bk.reduce(side, price, sz, "exec" if rec_type == 4 else "cancel")
                if oid is None:            # reducing liquidity we never tracked -> skip
                    continue
            rec_rows.append((tt, rec_type, oid, sz, price, side))
            rec_books.append(bk.snapshot())
        if n_hidden:
            # Hidden executions leave the visible ladder untouched, so no QueueBook mutation and no
            # order id; price is the touch, direction follows whichever side traded more.
            hv = int(round(frame_trades[t][3]))
            side = -1 if frame_trades[t][0] >= frame_trades[t][1] else 1
            px = (min(cur_ask) if (side == -1 and cur_ask)
                  else (max(cur_bid) if cur_bid else 0))
            if hv > 0 and px:
                rec_rows.append((t0 + n_ev / (n_ev + 1) * (t1 - t0), 5, 0, hv, px, side))
                rec_books.append(bk.snapshot())
        prev_ask, prev_bid = cur_ask, cur_bid

    cols = ["time", "event_type", "order_id", "size", "price", "direction"]
    return pd.DataFrame(rec_rows, columns=cols), pd.DataFrame(np.asarray(rec_books))
