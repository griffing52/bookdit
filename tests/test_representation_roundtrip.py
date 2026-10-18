"""The coordinate chart is the paper's central structural claim, so it is guarded here.

`scripts/lib/lob_representation.py` maps a raw ten-level book to the 41-column
`midspread_dt` chart BookDiT generates in. Two properties have to hold:

  ROUND-TRIP  encode then decode returns the original book exactly (prices and sizes
              are recovered bit-for-bit; times only to float32 precision, because the
              chart stores log1p(Δt) and the decoder cumsums it back).
  VALIDITY    decoding an ARBITRARY point of the chart -- Gaussian noise, i.e. whatever
              the flow happens to sample -- still yields a book with a positive spread
              and strictly monotone ladders. That is what makes the chart a chart and
              not just a normalisation.

Synthetic input only (no LOBSTER, no checkpoint, no GPU); runs in well under a second.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "lib"))

from lob_representation import (  # noqa: E402
    CHART_ABLATIONS,
    book_to_chart_dt,
    book_to_midspread_dt,
    level_idx,
    midspread_dt_anchors,
    midspread_dt_to_book,
)

TICK = 100  # raw LOBSTER price units; every price sits on this grid


def synthetic_book(rng, shape=(240,)):
    """A valid ten-level LOBSTER book40 (+ event times) of leading shape `shape`.

    Mid is a tick-grid random walk, the spread and every level gap are >= 1 tick, and
    sizes are non-negative integer share counts -- i.e. the assumptions the chart makes.
    """
    mid = (10_000 + np.cumsum(rng.integers(-2, 3, shape + (1,)), axis=-1)) * TICK
    half = TICK * rng.integers(1, 4, shape + (1,))                    # half-spread >= 1 tick
    ask = mid + half + TICK * np.cumsum(rng.integers(1, 5, shape + (10,)), axis=-1) - TICK
    bid = mid - half - TICK * np.cumsum(rng.integers(1, 5, shape + (10,)), axis=-1) + TICK
    sizes = np.round(rng.lognormal(4.5, 1.2, shape + (10, 2)))
    book = np.empty(shape + (40,), np.float64)
    book[..., 0::4], book[..., 2::4] = ask, bid
    book[..., 1::4], book[..., 3::4] = sizes[..., 0], sizes[..., 1]
    times = 34_200.0 + np.cumsum(rng.exponential(0.05, shape), axis=-1)
    return book, times


def test_midspread_dt_roundtrips_a_ten_level_book():
    rng = np.random.default_rng(0)
    book, times = synthetic_book(rng)
    _, _, px, sz = level_idx(10)

    chart = book_to_midspread_dt(book, times)
    assert chart.shape == book.shape[:-1] + (41,)

    rec_book, rec_times = midspread_dt_to_book(chart, *midspread_dt_anchors(book, times))
    assert np.abs(rec_book[..., px] - book[..., px]).max() == 0.0   # prices: exact
    assert np.abs(rec_book[..., sz] - book[..., sz]).max() == 0.0   # sizes pass through
    assert np.abs(rec_times - times).max() < 1e-3                   # float32 log1p/cumsum


def test_midspread_dt_roundtrips_batched_windows():
    rng = np.random.default_rng(1)
    book, times = synthetic_book(rng, (6, 100))                     # (B, T, 40) / (B, T)

    chart = book_to_midspread_dt(book, times)
    rec_book, rec_times = midspread_dt_to_book(chart, *midspread_dt_anchors(book, times))
    assert np.abs(rec_book - book).max() == 0.0
    assert np.abs(rec_times - times).max() < 1e-3


def assert_valid(book, times):
    ask, bid = book[..., 0::4], book[..., 2::4]
    assert (ask[..., 0] > bid[..., 0]).all()                        # spread > 0, never crossed
    assert (np.diff(ask, axis=-1) > 0).all()                        # ask ladder increasing
    assert (np.diff(bid, axis=-1) < 0).all()                        # bid ladder decreasing
    assert (np.diff(times, axis=-1) >= 0).all()                     # time never runs backwards


def test_decoded_book_is_valid_for_arbitrary_chart_points():
    """The guarantee: ANY point of the chart decodes to a well-formed book."""
    rng = np.random.default_rng(2)
    chart = rng.standard_normal((4, 100, 41)).astype(np.float32)    # not a real book at all
    assert_valid(*midspread_dt_to_book(chart, np.zeros(4), np.full(4, 34_200.0)))

    # ...and again at LOBSTER magnitudes. The anchor mid has to be carried here: decoded
    # books come back in float32, whose resolution near a 1e6 mid is ~0.06 raw units, so a
    # sub-tick gap would round away into a tie. Real gaps are >= 1 tick = 100 units (the
    # same assumption MIN_GAP encodes), so the log-gap channels are drawn at that scale.
    chart[..., 0::4] = rng.normal(np.log(200.0), 0.5, chart[..., 0::4].shape)   # spread + ask gaps
    chart[..., 2::4] = rng.normal(np.log(200.0), 0.5, chart[..., 2::4].shape)   # bid gaps
    chart[..., 2] = rng.standard_normal((4, 100))                              # ...but col 2 is Δmid
    assert_valid(*midspread_dt_to_book(chart, np.full(4, 1e6), np.full(4, 34_200.0)))


def test_chart_full_ablation_is_the_bookdit_representation():
    """`chart_full` is an alias, not a second model -- it must be bit-identical."""
    rng = np.random.default_rng(3)
    book, times = synthetic_book(rng, (60,))
    assert np.array_equal(book_to_chart_dt(book, times, CHART_ABLATIONS["chart_full"]),
                          book_to_midspread_dt(book, times))
