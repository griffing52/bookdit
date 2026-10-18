#!/usr/bin/env python
"""Book representation transforms: raw book40 <-> the model's coordinate charts.

book40 layout (per frame, reshape (T,10,4)): channels [0=ask_px, 1=ask_sz, 2=bid_px, 3=bid_sz]
per level, flattened as index = level*4 + channel. So in the flat 40-vector:
  PRICE channels (ask_px, bid_px) = level*4 + {0,2}
  SIZE  channels (ask_sz, bid_sz) = level*4 + {1,3}

Every transform works on (T,40) or (...,T,40); time axis is -2, feature axis is -1.
The paper's BookDiT's representation is `midspread_dt` (41 channels), laid out as:

  0            log spread               log(ask_px L1 - bid_px L1)
  2            mid increment            mid_t - mid_{t-1}  (0 at frame 0)
  4k, k=1..9   log ask gap              log(ask_px L(k+1) - ask_px Lk)
  4k+2, k=1..9 log bid gap              log(bid_px Lk - bid_px L(k+1))
  odd 1..39    sizes, absolute shares   ask_sz / bid_sz of each level, in book40 order
  40           elapsed time             log1p(Δt), Δt in seconds (0 at frame 0)

so the log ladder gaps are every even channel from 4 to 38. `CHART_ABLATIONS` at the end of
this file registers the reduced charts of the coordinate-chart ablation (the paper's Table 4).
"""
from __future__ import annotations
import numpy as np

PRICE_IDX = [lvl * 4 + c for lvl in range(10) for c in (0, 2)]   # ask_px, bid_px  (len 20)
SIZE_IDX = [lvl * 4 + c for lvl in range(10) for c in (1, 3)]     # ask_sz, bid_sz  (len 20)


def n_levels(w) -> int:
    """Levels carried by a raw book array, from its last-axis width (40 -> 10, 80 -> 20)."""
    return _np(w).shape[-1] // 4


def level_idx(L: int):
    """(ask_px, bid_px, price, size) column indices for an L-level book."""
    ask = [4 * l for l in range(L)]
    bid = [4 * l + 2 for l in range(L)]
    price = [l * 4 + c for l in range(L) for c in (0, 2)]
    size = [l * 4 + c for l in range(L) for c in (1, 3)]
    return ask, bid, price, size


def _np(w):
    try:
        import torch
        if isinstance(w, torch.Tensor):
            return w.detach().cpu().numpy()
    except ImportError:
        pass
    return np.asarray(w)


# --------------------------------------------------------------- mid+spread representation
# Spread and per-level gaps are generated in log space, so any decoded book has positive spread
# and strictly monotone price ladders on both sides. Same 40-col layout, so the model code is
# unchanged:
#   col 0        (ask_px L1)      -> log(spread)
#   col 2        (bid_px L1)      -> temporal delta of mid  (d[0]=0, anchor = t0 mid)
#   col 4k       (ask_px L(k+1))  -> log(ask_px L(k+1) - ask_px Lk)      k=1..9
#   col 4k+2     (bid_px L(k+1))  -> log(bid_px Lk - bid_px L(k+1))
#   sizes unchanged (absolute).
MIN_GAP = 1.0            # raw price units; real gaps are >= 1 tick (100), so this never binds
ASK_PX = [4 * l for l in range(10)]
BID_PX = [4 * l + 2 for l in range(10)]


def book_to_midspread(w, min_gap: float | None = None):
    """(...,T,40) raw book -> mid+spread rep (see module comment).

    ``min_gap`` floors the gaps before the log. The module default (1.0) is calibrated for
    raw LOBSTER integer ticks, where real gaps are >= 100 so it never binds. On a rescaled
    book (e.g. FI-2010 DecPre, where gaps are ~1e-4) that default would clamp every gap to
    log(1)=0 and silently destroy the representation -- pass a scale-appropriate value.
    """
    mg = MIN_GAP if min_gap is None else float(min_gap)
    a = _np(w).astype(np.float64).copy()
    L = a.shape[-1] // 4                                      # 10 for book40, 20 for book80
    _ask_px, _bid_px, _, _ = level_idx(L)
    ask, bid = a[..., _ask_px], a[..., _bid_px]               # (...,T,L) each
    mid = 0.5 * (ask[..., 0] + bid[..., 0])                   # (...,T)
    dmid = np.zeros_like(mid)
    dmid[..., 1:] = mid[..., 1:] - mid[..., :-1]
    a[..., 0] = np.log(np.maximum(ask[..., 0] - bid[..., 0], mg))
    a[..., 2] = dmid
    for k in range(1, L):
        a[..., 4 * k] = np.log(np.maximum(ask[..., k] - ask[..., k - 1], mg))
        a[..., 4 * k + 2] = np.log(np.maximum(bid[..., k - 1] - bid[..., k], mg))
    return a.astype(np.float32)


def midspread_to_book(h, anchor_mid):
    """Inverse. anchor_mid = t0 absolute mid, shape (...,) matching leading dims.
    Decoded books are VALID by construction: spread>0, ask ladder increasing, bid decreasing."""
    a = _np(h).astype(np.float64).copy()
    L = a.shape[-1] // 4
    mid = np.cumsum(a[..., 2], axis=-1) + np.asarray(anchor_mid, np.float64)[..., None]
    half = 0.5 * np.exp(a[..., 0])
    ask, bid = mid + half, mid - half
    out = a
    out[..., 0], out[..., 2] = ask, bid
    for k in range(1, L):
        ask = ask + np.exp(a[..., 4 * k])
        bid = bid - np.exp(a[..., 4 * k + 2])
        out[..., 4 * k], out[..., 4 * k + 2] = ask, bid
    return out.astype(np.float32)


def midspread_anchor(w):
    """t0 absolute mid of raw book window(s) (...,T,40) -> (...,)."""
    a = _np(w).astype(np.float64)
    return 0.5 * (a[..., 0, 0] + a[..., 0, 2])


# --------------------------------------------------------------- midspread + Δt (41-col) representation
# book40 carries NO time channel, so a book-only model cannot fit LOB-Bench's timing statistics:
# the decoder would have to fabricate one second per book frame. Here we append a 41st channel =
# log1p(Δt), the inter-event time in seconds between consecutive snapshots, so the model GENERATES
# timing directly. Δt comes from the LOBSTER message times -- it is NOT derivable from the book.
# Cols 0..39 are the mid+spread rep unchanged; col 40 is the only addition.
DT_COL = 40


def book_to_midspread_dt(w, t):
    """(...,T,40) raw book + (...,T) snapshot times (seconds) -> (...,T,41) midspread+log1p(Δt)."""
    ms = book_to_midspread(w)                                 # (...,T,40) float32
    tt = _np(t).astype(np.float64)
    dt = np.zeros_like(tt)
    dt[..., 1:] = np.maximum(tt[..., 1:] - tt[..., :-1], 0.0)  # LOBSTER times are non-decreasing
    out = np.concatenate([ms.astype(np.float64), np.log1p(dt)[..., None]], axis=-1)
    return out.astype(np.float32)


def midspread_dt_to_book(h, anchor_mid, anchor_t):
    """Inverse. Returns (book40 (...,T,40), times (...,T)). anchor_mid = t0 mid, anchor_t = t0 time.
    Decoded books are valid by construction (from midspread); times are monotone non-decreasing."""
    a = _np(h).astype(np.float64)
    nb = (a.shape[-1] - 1) // 4 * 4                            # book cols (40, or 80 at 20 levels)
    book = midspread_to_book(a[..., :nb], anchor_mid)          # (...,T,nb)
    dt = np.expm1(np.maximum(a[..., nb], 0.0))                 # clip generated Δt >= 0
    dt[..., 0] = 0.0                                           # frame 0 is the anchor
    times = np.cumsum(dt, axis=-1) + np.asarray(anchor_t, np.float64)[..., None]
    return book, times.astype(np.float64)


def midspread_dt_anchors(w, t):
    """(mid, time) at frame 0 for raw window(s) (...,T,40) + times (...,T) -> ((...,), (...,))."""
    return midspread_anchor(w), _np(t)[..., 0].astype(np.float64)


# ------------------------------------------------- midspread_dt + log-volume (41-col) representation
# A variant of midspread_dt; the paper's model does not use it. Volumes are heavy-tailed counts,
# modeled in midspread_dt in absolute shares under a single global standardization; log1p
# compresses the tail and makes scale differences additive (per-name/per-year volume regimes become
# shifts, not stretches). Cols as midspread_dt; the 20 size cols move to log1p.
VOL_IDX = [lvl * 4 + c for lvl in range(10) for c in (1, 3)]     # ask_sz, bid_sz (len 20)


def vol_idx(width: int):
    """Size-channel indices for a rep of the given width (41 -> 10 levels, 81 -> 20)."""
    return [lvl * 4 + c for lvl in range((width - 1) // 4) for c in (1, 3)]


def book_to_midspread_dt_logvol(w, t):
    """(...,T,40) raw book + (...,T) times -> (...,T,41) midspread + log1p sizes + log1p(Δt)."""
    a = book_to_midspread_dt(w, t).astype(np.float64)
    _vi = vol_idx(a.shape[-1])
    a[..., _vi] = np.log1p(np.maximum(a[..., _vi], 0.0))
    return a.astype(np.float32)


def midspread_dt_logvol_to_book(h, anchor_mid, anchor_t):
    """Inverse. Decoded sizes are non-negative by construction (expm1 of clipped log)."""
    a = _np(h).astype(np.float64).copy()
    _vi = vol_idx(a.shape[-1])
    a[..., _vi] = np.expm1(np.maximum(a[..., _vi], 0.0))
    return midspread_dt_to_book(a, anchor_mid, anchor_t)


# --------------------------------------------- midspread_dt_logvol + TRADE channels (45-col)
# A variant the paper's model does not use. Executions are visible to the ladder only through
# the reduction they cause, and at stride 10 a frame-to-frame book diff cannot tell an execution
# from a cancellation: almost every visible execution sits at the touch, but most do not clear the
# touch level, so a book-only sweep rule both misses executions and labels cancels as executions.
# Four appended channels make the traded flow an explicit, generated part of the state:
#
#   col 41  log1p(buy_vol)    visible (type-4) shares executed against the ASK  -> buyer-initiated
#   col 42  log1p(sell_vol)   visible (type-4) shares executed against the BID  -> seller-initiated
#   col 43  log1p(n_trades)   number of visible executions in the frame (fragmentation / msg count)
#   col 44  log1p(hidden_vol) type-5 hidden-execution shares -- STRICTLY invisible to the
#                             visible ladder
#
# Deliberately NOT included: trade price / VWAP offset (executions are almost all at the touch,
# whose price the book already carries exactly) and a per-side hidden split (one channel already
# carries the hidden tape's magnitude).
# Encoded/decoded by delegation to the 41-col rep, so the level-count inference in `vol_idx` /
# `midspread_dt_to_book` keeps seeing a book+Δt width and stays correct at 10 or 20 levels.
N_TRADE_CH = 4
TRADE_NAMES = ("buy_vol", "sell_vol", "n_trades", "hidden_vol")


def split_trades(h, n_trade: int = N_TRADE_CH):
    """(...,C+4) rep -> ((...,C) book+Δt part, (...,4) trade part). Use this before handing an
    array to any helper that infers the level count from the width."""
    a = _np(h)
    return a[..., :-n_trade], a[..., -n_trade:]


def book_to_midspread_dt_logvol_trades(w, t, tr):
    """(...,T,40) raw book + (...,T) times + (...,T,4) raw trade counts -> (...,T,45).

    ``tr`` columns are TRADE_NAMES in raw units (shares / message counts, non-negative)."""
    a = book_to_midspread_dt_logvol(w, t).astype(np.float64)          # (...,T,41)
    q = np.log1p(np.maximum(_np(tr).astype(np.float64), 0.0))         # (...,T,4)
    return np.concatenate([a, q], axis=-1).astype(np.float32)


def midspread_dt_logvol_trades_to_book(h, anchor_mid, anchor_t):
    """Inverse -> (book40, times, trades). Trades are non-negative by construction (expm1 of a
    clipped log); counts are left real-valued, the caller rounds if it needs integers."""
    body, q = split_trades(h)
    book, times = midspread_dt_logvol_to_book(body, anchor_mid, anchor_t)
    trades = np.expm1(np.maximum(np.asarray(q, np.float64), 0.0))
    return book, times, trades.astype(np.float32)


# --------------------------------------------- rep-aware level inference (single source of truth)
# `(C - 1) // 4` is only correct when Δt is the sole non-book channel; the trades reps carry four
# more. Import `levels_of` rather than re-deriving the level count at each consumer.
N_EXTRA_CH = {
    # Chart ablations are registered at the end of this file, once CHART_ABLATIONS exists.
    "midspread_dt": 0,
    "midspread_dt_logvol": 0,
    "midspread_dt_logvol_trades": N_TRADE_CH,      # 4
}


def levels_of(rep: str, width: int) -> int:
    """Book levels carried by a width-`width` array of representation `rep`.

    >>> levels_of("midspread_dt", 41), levels_of("midspread_dt_logvol_trades", 125)
    (10, 30)
    """
    if rep not in N_EXTRA_CH:
        raise ValueError(f"unknown representation {rep!r}; known: {sorted(N_EXTRA_CH)}")
    body = width - 1 - N_EXTRA_CH[rep]
    if body <= 0 or body % 4:
        raise ValueError(f"width {width} is not a valid {rep} array "
                         f"(book part {body} is not a positive multiple of 4)")
    return body // 4


# ============================================================ chart ablations (41-col)
# The structured chart of `book_to_midspread` is a composition of two independent choices, each
# carrying a structural guarantee:
#
#   TOUCH   col 2 = Δmid          -> the price coordinate is stationary (E[Δmid] ~ 0)
#           col 0 = log(spread)   -> exp() > 0, so the touch can never lock or cross
#   LADDER  cols 4k, 4k+2 (k>=1) = log(gap)
#                                 -> exp() > 0, so the ladder can never be non-monotone,
#                                    and level k is built cumulatively from level k-1
#
# `touch` and `ladder` below select one option each; a row of the ablation table changes exactly one
# of them, and the two endpoints (`chart_full`, `chart_raw`) bracket the ladder. Width stays 41 in
# every arm, so parameter count, batch, steps, lr, seed, decoder and generation draw are identical
# across the table and only the meaning of four column groups changes.
#
# `linspread` and `linoffset` keep the quantity stationary and O(ticks) and remove only positivity,
# so those rows isolate a single guarantee; `absmid` has no such option, since the absolute mid IS
# the non-stationary coordinate. `rawtouch` and `abspx` predict the level-one ask and bid (and, for
# `abspx`, every level's price) directly -- the parameterization a book model with no
# representation would use -- and bound the table from the other end.
CHART_TOUCH = {
    "midspread": "col0 = log(ask0-bid0), col2 = Δmid            [both guarantees]",
    "absmid":    "col0 = log(ask0-bid0), col2 = mid             [stationarity removed]",
    "linspread": "col0 = ask0-bid0,      col2 = Δmid            [spread positivity removed]",
    "rawtouch":  "col0 = ask0,           col2 = bid0            [naive: neither]",
}
CHART_LADDER = {
    "loggap":    "col 4k = log(ask_k - ask_{k-1}), col 4k+2 = log(bid_{k-1} - bid_k)",
    "linoffset": "col 4k = ask_k - ask_0,          col 4k+2 = bid_0 - bid_k",
    "abspx":     "col 4k = ask_k,                  col 4k+2 = bid_k          [naive]",
}
CHART_FULL = ("midspread", "loggap")


def book_to_chart(w, touch="midspread", ladder="loggap", min_gap: float | None = None):
    """(...,T,40) raw book -> 40-col chart under the named touch/ladder parameterization.

    ``touch="midspread", ladder="loggap"`` reproduces `book_to_midspread` bit for bit."""
    if touch not in CHART_TOUCH or ladder not in CHART_LADDER:
        raise ValueError(f"unknown chart ({touch!r}, {ladder!r})")
    mg = MIN_GAP if min_gap is None else float(min_gap)
    a = _np(w).astype(np.float64).copy()
    L = a.shape[-1] // 4
    _ask_px, _bid_px, _, _ = level_idx(L)
    ask, bid = a[..., _ask_px], a[..., _bid_px]
    mid = 0.5 * (ask[..., 0] + bid[..., 0])
    spread = ask[..., 0] - bid[..., 0]
    if touch == "rawtouch":
        a[..., 0], a[..., 2] = ask[..., 0], bid[..., 0]
    else:
        if touch == "absmid":
            a[..., 2] = mid
        else:
            dmid = np.zeros_like(mid)
            dmid[..., 1:] = mid[..., 1:] - mid[..., :-1]
            a[..., 2] = dmid
        a[..., 0] = spread if touch == "linspread" else np.log(np.maximum(spread, mg))
    for k in range(1, L):
        if ladder == "loggap":
            a[..., 4 * k] = np.log(np.maximum(ask[..., k] - ask[..., k - 1], mg))
            a[..., 4 * k + 2] = np.log(np.maximum(bid[..., k - 1] - bid[..., k], mg))
        elif ladder == "linoffset":
            a[..., 4 * k] = ask[..., k] - ask[..., 0]
            a[..., 4 * k + 2] = bid[..., 0] - bid[..., k]
        else:                                      # abspx
            a[..., 4 * k] = ask[..., k]
            a[..., 4 * k + 2] = bid[..., k]
    return a.astype(np.float32)


def chart_to_book(h, anchor_mid, touch="midspread", ladder="loggap"):
    """Inverse of `book_to_chart`. Output validity is exactly what the arm removes: with
    ``midspread``/``loggap`` the book is valid by construction; ``linspread``/``rawtouch`` allow a
    locked or crossed touch; ``linoffset``/``abspx`` allow a non-monotone ladder."""
    a = _np(h).astype(np.float64).copy()
    L = a.shape[-1] // 4
    if touch == "rawtouch":
        ask0, bid0 = a[..., 0], a[..., 2]
    else:
        mid = (a[..., 2] if touch == "absmid" else
               np.cumsum(a[..., 2], axis=-1) + np.asarray(anchor_mid, np.float64)[..., None])
        half = 0.5 * (a[..., 0] if touch == "linspread" else np.exp(a[..., 0]))
        ask0, bid0 = mid + half, mid - half
    out = a
    ask, bid = ask0, bid0
    for k in range(1, L):
        if ladder == "loggap":
            ask = ask + np.exp(a[..., 4 * k])
            bid = bid - np.exp(a[..., 4 * k + 2])
            out[..., 4 * k], out[..., 4 * k + 2] = ask, bid
        elif ladder == "linoffset":
            out[..., 4 * k] = ask0 + a[..., 4 * k]
            out[..., 4 * k + 2] = bid0 - a[..., 4 * k + 2]
        # abspx: the column already IS the price, leave it
    out[..., 0], out[..., 2] = ask0, bid0
    return out.astype(np.float32)


def book_to_chart_dt(w, t, chart=CHART_FULL):
    """(...,T,40) + times -> (...,T,41): `book_to_chart` plus the unchanged log1p(Δt) channel."""
    ch = book_to_chart(w, *chart)
    tt = _np(t).astype(np.float64)
    dt = np.zeros_like(tt)
    dt[..., 1:] = np.maximum(tt[..., 1:] - tt[..., :-1], 0.0)
    return np.concatenate([ch.astype(np.float64), np.log1p(dt)[..., None]],
                          axis=-1).astype(np.float32)


def chart_dt_to_book(h, anchor_mid, anchor_t, chart=CHART_FULL):
    """Inverse of `book_to_chart_dt` -> (book, times). Δt handling is identical in every arm."""
    a = _np(h).astype(np.float64)
    nb = (a.shape[-1] - 1) // 4 * 4
    book = chart_to_book(a[..., :nb], anchor_mid, *chart)
    dt = np.expm1(np.maximum(a[..., nb], 0.0))
    dt[..., 0] = 0.0
    times = np.cumsum(dt, axis=-1) + np.asarray(anchor_t, np.float64)[..., None]
    return book, times.astype(np.float64)


# Registered rep names -> (touch, ladder). `chart_full` is an alias of the paper's model's
# `midspread_dt`; it is byte-identical to that path and is NOT a separate model.
CHART_ABLATIONS = {
    "chart_full":      ("midspread", "loggap"),     # control = midspread_dt
    "chart_absmid":    ("absmid",    "loggap"),     # stationarity removed
    "chart_linspread": ("linspread", "loggap"),     # spread positivity removed
    "chart_linlevels": ("midspread", "linoffset"),  # ladder monotonicity removed
    "chart_rawtouch":  ("rawtouch",  "loggap"),     # naive touch: predict ask0/bid0 directly
    "chart_raw":       ("rawtouch",  "abspx"),      # naive throughout: predict every price directly
}

# Every chart ablation is width 41 (10 levels + Δt) like midspread_dt, so none carries extra
# channels.  Registered here rather than in the literal above because CHART_ABLATIONS is defined
# after it; `levels_of` is what reads this and is only ever called at run time.
N_EXTRA_CH.update({_r: 0 for _r in CHART_ABLATIONS})
