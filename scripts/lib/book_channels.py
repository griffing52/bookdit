#!/usr/bin/env python
"""Book-state-only dynamics channels shared by the book-lane analysis scripts.

One definition of the ten per-frame channels (mid change, spread, touch imbalance, order-flow
imbalance, log sizes/depths) so every script that summarises a window of book frames reads the
same quantities.  Computed from BOOK STATE ONLY -- no event types, no order ids, no inter-arrival
times -- so it applies unchanged to arms whose message lane comes from an inverse-dynamics decode.
"""
from __future__ import annotations

import numpy as np


def _book_channels(b: np.ndarray) -> dict:
    """Channels computed from BOOK STATE ONLY -- no event types, no order ids, no inter-arrival."""
    ask_p, ask_s, bid_p, bid_s = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    mid = 0.5 * (ask_p + bid_p)
    dmid = np.diff(mid, prepend=mid[0])
    imb = (bid_s - ask_s) / np.maximum(bid_s + ask_s, 1.0)
    adepth = b[:, 1::4][:, :10].sum(1) if b.shape[1] >= 40 else ask_s
    bdepth = b[:, 3::4][:, :10].sum(1) if b.shape[1] >= 40 else bid_s
    bp, bs, ap, as_ = bid_p[:-1], bid_s[:-1], ask_p[:-1], ask_s[:-1]
    ofi = np.zeros(len(b))
    ofi[1:] = ((bid_p[1:] >= bp) * bid_s[1:] - (bid_p[1:] <= bp) * bs
               - (ask_p[1:] <= ap) * ask_s[1:] + (ask_p[1:] >= ap) * as_)
    return {"dmid": dmid, "absdmid": np.abs(dmid), "spread": (ask_p - bid_p).astype(float),
            "imb": imb, "ofi": ofi, "sofi": np.sign(ofi),
            "lasz": np.log1p(ask_s.astype(float)), "lbsz": np.log1p(bid_s.astype(float)),
            "ladep": np.log1p(adepth.astype(float)), "lbdep": np.log1p(bdepth.astype(float))}
