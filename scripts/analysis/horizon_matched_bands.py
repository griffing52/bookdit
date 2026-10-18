#!/usr/bin/env python
"""Long-horizon drift bands (runs/figures/horizon_bands.pdf) on a fixed three-way window set,
with a real-vs-real floor arm.

The population is the intersection over every block of every arm, so all arms see identical
windows at all 32 blocks and no composition term survives.

The floor arm is the same L1 between two disjoint halves of the real windows, averaged over R
seeded splits; it measures how much of the level is the estimator rather than the model.  Each
half holds half the windows, so the floor sits above what a same-size comparison would give; its
slope, which is what the figure compares, is unaffected.  `--floor-scaling` measures the level
inflation directly.

Inputs are the per-arm caches written by horizon_score_cache.py, read from
<--cache-root>/{bookdit,lobs5,cst}.  `--validate` additionally reads each arm's LOB-Bench profile:
BookDiT's profile_fixedset.csv (rescore_horizon_fixedset.py over the horizon_official_protocol.py
run) and the LOB-S5 / CST profile.csv (score_lobs5_horizon.py); override with --profile ARM=PATH.

Usage:
    horizon_matched_bands.py --out runs/horizon/bands
    horizon_matched_bands.py --population union --out ...     # BookDiT-and-CST window set
    horizon_matched_bands.py --floor-scaling                  # level-vs-window-count diagnostic
    horizon_matched_bands.py --validate                       # replay == published profiles
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))
from _l1replay import BlockCache, DROP18, DISCRETE, l1_of                      # noqa: E402

ARMS = ("bookdit", "lobs5", "cst")
# Defaults follow the documented horizon lane: horizon_score_cache.py writes
# runs/horizon/cache/<arm>, and each profile sits in that arm's scored run.  Both are overridable
# from the command line (--cache-root, --profile).
CACHE = {arm: ROOT / "runs/horizon/cache" / arm for arm in ARMS}
PROFILES = {
    "bookdit": ROOT / "runs/horizon/bookdit/profile_fixedset.csv",
    "lobs5": ROOT / "runs/horizon/lobs5/profile.csv",
    "cst": ROOT / "runs/horizon/cst/profile.csv",
}
KEEP18 = None            # filled from the cache's own stat list


def load_caches():
    out = {}
    for arm, d in CACHE.items():
        files = sorted(d.glob("block_*.npz"), key=lambda p: int(p.stem.split("_")[1]))
        if not files:
            raise SystemExit(f"no cache for {arm} under {d}")
        # z["horizon"] is the CUMULATIVE rollout length at that block (500, 1000, ... 16000).
        out[arm] = [(int(z["horizon"][0]), BlockCache(z))
                    for z in (np.load(f) for f in files)]
    return out


def block_horizons(caches):
    """Cumulative rollout length at each block. The cache stores the BLOCK length (500), not the
    cumulative horizon, so it is reconstructed here and cross-checked against the arms."""
    n = len(caches["bookdit"])
    for arm in ARMS:
        assert len(caches[arm]) == n, f"{arm} has {len(caches[arm])} blocks, expected {n}"
    return np.array([h for h, _ in caches["bookdit"]], dtype=float)


def population(caches, mode: str) -> list[str]:
    per_arm = {a: set.intersection(*[set(bc.rids) for _, bc in caches[a]]) for a in ARMS}
    if mode == "matched":
        pop = set.intersection(*per_arm.values())
    elif mode == "union":                       # bookdit-and-cst ids; lobs5 may cover a subset
        pop = per_arm["bookdit"] & per_arm["cst"]
    else:
        raise SystemExit(f"unknown population mode {mode}")
    for a in ARMS:
        print(f"    {a:8s} all-block set {len(per_arm[a]):3d}")
    return sorted(pop, key=int)


def index_maps(caches, pop):
    m = {}
    for arm in ARMS:
        rows = []
        for h, bc in caches[arm]:
            pos = {r: i for i, r in enumerate(bc.rids)}
            rows.append((h, bc, np.array([pos.get(r, -1) for r in pop])))
        m[arm] = rows
    return m


def curve(maps, arm, draw):
    out = []
    for _, bc, loc in maps[arm]:
        sel = loc[draw]
        out.append(bc.l1_18(sel[sel >= 0]))
    return np.array(out)


# ------------------------------------------------------------------------------- floor arm
def floor_curve(maps, draw, rng, R: int, arm: str = "bookdit", loc_override=None):
    """L1^18 between the pooled REAL scores of two disjoint halves of the drawn windows.

    Uses one arm's cache purely as the carrier of the real-side arrays; `--check-real` asserts the
    arms agree there, which they must (same window, same block, same real continuation)."""
    blocks = maps[arm]
    keep = [s for s in blocks[0][1].stats if s not in DROP18]
    out = np.zeros(len(blocks))
    for bi, (_, bc, loc) in enumerate(blocks):
        if loc_override is not None:
            loc = loc_override[bi]
        sel = loc[draw]
        sel = sel[sel >= 0]
        vals = []
        for _ in range(R):
            p = rng.permutation(len(sel))
            a, b = sel[p[:len(sel) // 2]], sel[p[len(sel) // 2:]]
            vals.append(np.mean([l1_of(bc.pooled(s, "real", a), bc.pooled(s, "real", b),
                                       s in DISCRETE) for s in keep]))
        out[bi] = float(np.mean(vals))
    return out


def floor_scaling(maps, draw, rng, arm="cst", fracs=(0.125, 0.25, 0.5)):
    """Floor level as a function of windows per side, at block 0 -- quantifies the level caveat."""
    _, bc, loc = maps[arm][0]
    sel = loc[draw]
    sel = sel[sel >= 0]
    keep = [s for s in bc.stats if s not in DROP18]
    rows = []
    for f in fracs:
        k = max(2, int(round(len(sel) * f)))
        vals = []
        for _ in range(60):
            p = rng.permutation(len(sel))
            a, b = sel[p[:k]], sel[p[k:2 * k]]
            if len(b) < k:
                continue
            vals.append(np.mean([l1_of(bc.pooled(s, "real", a), bc.pooled(s, "real", b),
                                       s in DISCRETE) for s in keep]))
        rows.append((k, float(np.mean(vals))))
    return rows


def slope_of(h, y):
    return float(np.polyfit(h / 1000.0, y, 1)[0])


def validate(caches) -> None:
    """Replay each arm on the window set its PUBLISHED profile used, and compare per statistic.

    The three profiles are not on the same kind of window set, which matters here:
      bookdit  profile_fixedset.csv is the intersection over blocks (rescore_horizon_fixedset.py),
               while the cache holds every window in the block tree -- so restrict to the
               intersection.
      lobs5    profile.csv is per-block: the official level-10 filter empties a different handful
               of windows at each block (kept ranges 89-94), so its published curve is NOT on a
               fixed set.  Replay each block on exactly the windows that block scored.
      cst      per-block too, but the filter never removes any (kept = 95 at all 32 blocks).
    """
    print("--- replay vs published profiles (published window set per block, no resampling) ---")
    worst = 0.0
    for arm in ARMS:
        if not PROFILES[arm].exists():
            raise SystemExit(f"--validate: no {arm} profile at {PROFILES[arm]} (see --profile)")
        prof = pd.read_csv(PROFILES[arm])
        fixed = set.intersection(*[set(bc.rids) for _, bc in caches[arm]]) \
            if arm == "bookdit" else None
        d18 = []
        for (h, bc), (_, row) in zip(caches[arm], prof.iterrows()):
            assert h == int(row.horizon), f"{arm}: block order mismatch {h} vs {row.horizon}"
            idx = np.array([i for i, r in enumerate(bc.rids)
                            if fixed is None or r in fixed])
            for s in bc.stats:
                worst = max(worst, abs(bc.l1(s, idx) - float(row[f"stat_{s}"])))
            pub = float(np.mean([row[f"stat_{s}"] for s in bc.stats if s not in DROP18]))
            d18.append(abs(bc.l1_18(idx) - pub))
        ns = [len(bc.rids) for _, bc in caches[arm]]
        print(f"  {arm:8s} windows/block {min(ns)}-{max(ns)}"
              + (f" (published fixed set {len(fixed)})" if fixed else "")
              + f"  blocks={len(prof):2d}  max|L1^18 replay - published| = {max(d18):.2e}")
    print(f"  worst per-statistic disagreement over all arms/blocks: {worst:.2e}")
    assert worst < 1e-10, "replay does not reproduce LOB-Bench"

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--population", choices=["matched", "union"], default="matched")
    ap.add_argument("-B", type=int, default=1000)
    ap.add_argument("-R", type=int, default=24, help="random half-splits per block for the floor")
    ap.add_argument("--boot-R", type=int, default=4,
                    help="half-splits per bootstrap replicate; must be >1")
    ap.add_argument("--floor-pop", choices=["matched", "carrier"], default="carrier",
                    help="windows the real-vs-real control is built from: `carrier` splits the "
                         "carrier arm's own all-block set 38/38; `matched` splits the 3-way "
                         "population")
    ap.add_argument("--seed", type=int, default=20260804)
    ap.add_argument("--check-real", action="store_true")
    ap.add_argument("--floor-scaling", action="store_true")
    ap.add_argument("--validate", action="store_true",
                    help="replay each arm on its published window set and compare per statistic")
    ap.add_argument("--cache-root", default=str(ROOT / "runs/horizon/cache"),
                    help="directory holding one horizon_score_cache.py --out per arm")
    ap.add_argument("--profile", action="append", default=[], metavar="ARM=PATH",
                    help="--validate only: override an arm's LOB-Bench profile csv")
    ap.add_argument("--out", default=str(ROOT / "runs/horizon/bands"))
    a = ap.parse_args()
    for arm in ARMS:
        CACHE[arm] = Path(a.cache_root) / arm
    for spec in a.profile:
        arm, _, path = spec.partition("=")
        if arm not in ARMS or not path:
            ap.error(f"--profile wants ARM=PATH with ARM in {ARMS}, got {spec!r}")
        PROFILES[arm] = Path(path)

    caches = load_caches()
    if a.validate:
        validate(caches)
        return
    hs = block_horizons(caches)
    print(f"[matched horizon bands] {len(hs)} blocks to h={int(hs[-1])}")
    pop = population(caches, a.population)
    print(f"    population ({a.population}) = {len(pop)} windows")
    maps = index_maps(caches, pop)

    if a.check_real:
        bad = 0
        for bi in range(len(hs)):
            for s in ("spread", "ofi"):
                ref = None
                for arm in ARMS:
                    _, bc, loc = maps[arm][bi]
                    sel = loc[np.arange(len(pop))]
                    v = bc.pooled(s, "real", sel[sel >= 0])
                    if ref is None:
                        ref = v
                    elif ref.shape != v.shape or not np.allclose(ref, v, equal_nan=True):
                        bad += 1
        print(f"    real-side agreement across arms: {'OK' if bad == 0 else f'{bad} MISMATCHES'}")

    # The control's own population.  See --floor-pop.
    if a.floor_pop == "carrier":
        fpop = sorted(set.intersection(*[set(bc.rids) for _, bc in caches["bookdit"]]), key=int)
    else:
        fpop = list(pop)
    floc = []
    for _, bc in caches["bookdit"]:
        pos = {r: i for i, r in enumerate(bc.rids)}
        floc.append(np.array([pos.get(r, -1) for r in fpop]))
    print(f"    control population ({a.floor_pop}) = {len(fpop)} windows "
          f"-> {len(fpop)//2}/{len(fpop)-len(fpop)//2} per side")

    rng = np.random.default_rng(a.seed)
    full = np.arange(len(pop))
    if a.floor_scaling:
        print("    floor level vs windows per side (block 0):")
        for k, v in floor_scaling(maps, full, rng):
            print(f"      {k:3d} windows/side  L1^18 = {v:.4f}")
        return

    point = {arm: curve(maps, arm, full) for arm in ARMS}
    point["real"] = floor_curve(maps, np.arange(len(fpop)),
                                np.random.default_rng(a.seed), a.R, loc_override=floc)
    keys = list(ARMS) + ["real"]
    pslope = {k: slope_of(hs, point[k]) for k in keys}

    t0 = time.time()
    sl = {k: np.zeros(a.B) for k in keys}
    cu = {k: np.zeros((a.B, len(hs))) for k in keys}
    for b in range(a.B):
        draw = rng.integers(0, len(pop), len(pop))          # paired across arms
        for arm in ARMS:
            y = curve(maps, arm, draw)
            cu[arm][b], sl[arm][b] = y, slope_of(hs, y)
        # unpaired when the control has its own population; same R as the point estimate's noise
        fdraw = rng.integers(0, len(fpop), len(fpop))
        y = floor_curve(maps, fdraw, rng, a.boot_R, loc_override=floc)
        cu["real"][b], sl["real"][b] = y, slope_of(hs, y)
        if (b + 1) % 50 == 0:
            print(f"      {b+1}/{a.B}  {time.time()-t0:.0f}s", flush=True)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"horizon": hs})
    for k in keys:
        df[f"{k}_point"] = point[k]
        df[f"{k}_se"] = cu[k].std(axis=0, ddof=1)
        df[f"{k}_bias"] = cu[k].mean(axis=0) - point[k]
        df[f"{k}_pct_lo"] = np.percentile(cu[k], 2.5, axis=0)
        df[f"{k}_pct_hi"] = np.percentile(cu[k], 97.5, axis=0)
    df.to_csv(out / "curve_bands_l1_18.csv", index=False)

    res = {"population_mode": a.population, "n_windows": len(pop), "B": a.B, "R": a.R,
           "boot_R": a.boot_R, "floor_pop_mode": a.floor_pop, "n_floor_windows": len(fpop),
           "floor_paired": a.floor_pop == "matched",
           "seed": a.seed, "window_ids": pop, "arms": {}, "contrasts": {}}
    for k in keys:
        s = sl[k]
        res["arms"][k] = {"slope_per_1k": pslope[k],
                          "lo": float(np.percentile(s, 2.5)), "hi": float(np.percentile(s, 97.5)),
                          "se": float(s.std(ddof=1)),
                          "first": float(point[k][0]), "last": float(point[k][-1])}
    # Contrasts against BookDiT (the model claim) AND against the real-vs-real control (the
    # stationarity claim: does this arm drift faster than two samples of real data do?).
    pairs = [(k, "bookdit") for k in ("lobs5", "cst", "real")] + \
            [(f"{k}_vs_real", k) for k in ("bookdit", "lobs5", "cst")]
    for name, ref in pairs:
        k = name.split("_vs_")[0]
        base = "bookdit" if not name.endswith("_vs_real") else "real"
        d = sl[k] - sl[base]
        res["contrasts"][name] = {
            "reference": base,
            "ratio": pslope[k] / pslope[base] if abs(pslope[base]) > 1e-12 else float("nan"),
            "diff_per_1k": pslope[k] - pslope[base],
            "diff_lo": float(np.percentile(d, 2.5)), "diff_hi": float(np.percentile(d, 97.5)),
            "p_diff_le0": float((d <= 0).mean()),
            # Fieller-type lower bound: the ratio's denominator can approach zero, so the
            # percentile ratio interval has no usable upper end; invert the difference test.
            "fieller_lo": float(np.percentile(sl[k], 2.5) / max(np.percentile(sl[base], 97.5),
                                                                1e-9))}
    (out / "bootstrap.json").write_text(json.dumps(res, indent=1))
    print(f"\n--- L1^18 drift slope per 1000 generated messages, {len(pop)} identical windows ---")
    for k in keys:
        s = res["arms"][k]
        print(f"  {k:8s} {s['first']:.4f} -> {s['last']:.4f}  slope {s['slope_per_1k']*1000:+.3f}e-3"
              f"  95% CI [{s['lo']*1000:+.3f}, {s['hi']*1000:+.3f}]e-3")
    for k, c in res["contrasts"].items():
        print(f"  {k:14s} - {c['reference']:7s} {c['diff_per_1k']*1000:+.3f}e-3 "
              f"[{c['diff_lo']*1000:+.3f}, {c['diff_hi']*1000:+.3f}]  ratio {c['ratio']:.2f}x  "
              f"P(<=0)={c['p_diff_le0']:.4f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
