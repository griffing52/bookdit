#!/usr/bin/env python
"""Rescore a message-LM gen-dir with a crossing-aware (matching-engine) replay.

`idm_annotator.QueueBook.add_limit` rests every limit order at its stated price, so an order that
crosses the book is parked on the wrong side and produces a negative spread. `MatchingBook`
executes the crossing part against the opposite side and rests only the remainder. It is
monkeypatched into `score_lobs5_retrained`, whose scoring is otherwise reused unchanged.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "lib"))
from _paths import lob_bench_dir                # noqa: E402
sys.path.insert(0, str(lob_bench_dir(required=True)))

import score_lobs5_retrained as S              # noqa: E402
from idm_annotator import QueueBook            # noqa: E402


class MatchingBook(QueueBook):
    """QueueBook that executes the crossing portion of an incoming limit order before resting."""
    def add_limit(self, side, price, size):
        size = int(size)
        if side == 1:                                   # incoming buy: match asks priced <= price
            while size > 0 and self.apx and self.apx[0] <= price:
                p = self.apx[0]
                take = min(size, self.asz[p])
                self.reduce(-1, p, take, "exec")
                size -= take
        else:                                           # incoming sell: match bids priced >= price
            while size > 0 and self.bpx and self.bpx[-1] >= price:
                p = self.bpx[-1]
                take = min(size, self.bsz[p])
                self.reduce(1, p, take, "exec")
                size -= take
        if size > 0:
            return QueueBook.add_limit(self, side, price, size)
        oid = self._next_oid                            # fully executed: mint an id for the msg row
        self._next_oid += 1
        return oid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gen-only", action="store_true")
    args = ap.parse_args()
    S.QueueBook = MatchingBook                          # build_stream() picks this up via module global
    print("[replay] MATCHING (crossing-aware)")
    argv = ["score", "--gen-dir", args.gen_dir, "--out", args.out]
    if args.gen_only:
        argv.append("--gen-only")
    sys.argv = argv
    S.main()


if __name__ == "__main__":
    main()
