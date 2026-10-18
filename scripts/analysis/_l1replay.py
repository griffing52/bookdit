"""Exact numpy re-implementation of LOB-Bench's per-statistic L1, from cached per-window scores.

A window-level bootstrap of the horizon curve needs the L1 recomputed thousands of times on
resampled window sets, and `scoring.run_benchmark` costs ~77 s per block.  The L1 is a pure
function of the per-window score arrays: pool real and gen scores, bin them (np.unique for
discrete statistics, Freedman-Diaconis edges otherwise), and take half the total-variation
distance between the two normalised group histograms.  So the score arrays are cached once and
the binning is replayed in numpy.

This module reproduces libs/lob_bench/{partitioning.group_by_score, metrics.l1_by_group._calc_l1}
exactly -- validated to <1e-12 against the published profile CSVs by
`scripts/analysis/horizon_matched_bands.py --validate`.
"""
from __future__ import annotations
import numpy as np

DROP18 = {"log_inter_arrival_time", "log_time_to_cancel", "vol_per_min"}
# discrete=True in libs/lob_bench/run_bench.py DEFAULT_SCORING_CONFIG; everything else falls
# through to bin_method='fd'.
DISCRETE = {"spread", "limit_ask_order_levels", "limit_bid_order_levels",
            "ask_cancellation_levels", "bid_cancellation_levels"}


def l1_of(real: np.ndarray, gen: np.ndarray, discrete: bool) -> float:
    """Half the total variation between the binned real and generated score distributions.

    Mirrors metrics._calc_l1: pandas .count() ignores NaN, so NaN scores contribute to neither
    histogram; the column normalisation is therefore by the NON-NaN count of each type.
    """
    if real.size == 0 or gen.size == 0:
        return 1.0
    both = np.concatenate([real, gen]).astype(float)
    if discrete:
        thr = np.unique(both)
    else:
        fin = both[np.isfinite(both)]
        if fin.size == 0:
            return 1.0
        thr = np.histogram_bin_edges(fin, bins="fd")
    gr = np.searchsorted(thr, real, side="right")
    gg = np.searchsorted(thr, gen, side="right")
    kr, kg = ~np.isnan(real), ~np.isnan(gen)
    nb = len(thr) + 1
    cr = np.bincount(gr[kr], minlength=nb).astype(float)
    cg = np.bincount(gg[kg], minlength=nb).astype(float)
    if cr.sum() == 0 or cg.sum() == 0:
        return 1.0
    return float(np.abs(cr / cr.sum() - cg / cg.sum()).sum() / 2.0)


class BlockCache:
    """Per-window score arrays for one (arm, block), ragged, stored flat + offsets."""

    def __init__(self, npz):
        self.rids = [str(r) for r in npz["rids"]]
        self.stats = [str(s) for s in npz["stats"]]
        self._d = {}
        for s in self.stats:
            for side in ("real", "gen"):
                self._d[(s, side)] = (npz[f"{side}__{s}__v"], npz[f"{side}__{s}__o"])

    def pooled(self, s: str, side: str, idx: np.ndarray) -> np.ndarray:
        v, o = self._d[(s, side)]
        return np.concatenate([v[o[i]:o[i + 1]] for i in idx]) if len(idx) else v[:0]

    def l1(self, s: str, idx: np.ndarray) -> float:
        return l1_of(self.pooled(s, "real", idx), self.pooled(s, "gen", idx), s in DISCRETE)

    def l1_18(self, idx: np.ndarray) -> float:
        keep = [s for s in self.stats if s not in DROP18]
        assert len(keep) == 18, f"expected 18 statistics, got {len(keep)}"
        return float(np.mean([self.l1(s, idx) for s in keep]))

    def l1_21(self, idx: np.ndarray) -> float:
        return float(np.mean([self.l1(s, idx) for s in self.stats]))
