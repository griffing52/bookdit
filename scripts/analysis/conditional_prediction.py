#!/usr/bin/env python
"""Conditional prediction test (train-on-X / test-on-Y transfer) for LOB generators.

Distance scores say that a generator's marginals differ from the market's; they do not say whether
its conditional dynamics differ.  This script fits a predictor of a future book quantity from a
preceding window of book states: if the generated trajectories carry the same conditional dynamics
as the real ones, a predictor fit on one transfers to the other.

Cells of the transfer matrix (rows = training source, cols = evaluation source):

    real -> real      the reference.  Every other cell is a gap from it.
    real -> gen       does the real predictive relationship hold inside generated data?
    gen  -> real      TSTR: is a model learned purely from synthetic data useful on the market?
    gen  -> gen       the generator's own internal self-consistency (an upper bound for its row).

Two designs, both reported:

  design A  "shared context".  X comes from the real 250-frame context every arm was conditioned
            on, so the feature matrix is identical in all cells and only the label changes: any
            difference in R^2 is a difference in the conditional law, with no covariate shift.

  design B  "self-contained".  X and y both come from inside one source's own 50-frame
            continuation (first W frames -> target h frames later), which is what a practitioner
            with only synthetic data would do, and which admits covariate shift.  Read together,
            the two separate "wrong labels given a real context" from "wrong context distribution".

TARGETS, at horizons h frames ahead (1 frame = 10 messages); see `targets()`:
    dmid    mid-price change from the last context frame, in ticks
    imb     level-1 order-book imbalance (bid_sz - ask_sz) / (bid_sz + ask_sz)
    dimb    the same imbalance over all ten levels ("depth imbalance")
    rvol    realised volatility: mean |mid change| per frame over the h frames, in ticks
    spread  bid-ask spread at frame h, in ticks
  For dmid we additionally report directional accuracy (sign of the prediction vs sign of the
  realised move, on the non-zero-move subset) against its own majority-class base rate.
  The paper's "R^2 carried to real" aggregate is the design-A X->real R^2 averaged over
  {spread, dimb, rvol} x the run's horizons; `condpred_report.py` prints it.

PREDICTOR.  Ridge on ~106 standardised summary features of the context window (per-channel
mean/sd/quantiles/last/acf + flow->price cross-lag + variance ratios), alpha chosen by an inner
day-grouped CV on the training fold only.

SPLIT.  GroupKFold over trading day: window-level splitting would let the predictor train on 09:31
and test on 09:32 of the same session, so the test fold is a held-out day.  Out-of-fold
predictions are pooled and scored once.

QUANTIZATION CONTROL (--snap).  BookDiT's raw frames are off the tick grid and some of its sizes
are negative, which alone makes real and generated frames distinguishable.  --snap rounds prices to
the 100 tick and clips sizes at 0; it is a no-op on real and on LOB-S5 (both on-grid by
construction), so comparing the two runs says whether a gap is dynamics or output hygiene.

LOB-S5 comparator.  Its matching-engine replay is one row per message; row 10j is taken as frame j
(the same stride-10 grid).  Those files top out at 439 rows, so h=50 (500 messages) does not exist
for LOB-S5; the three-way run is restricted to the windows where every arm has the frames natively.

INPUTS.  `--frames` is a `gen_bookdit_on_lobbench_windows.py --dump-frames` directory (one
`<date>_rid<id>.npz` per window, holding cond_book / real_book / gen_book).  The LOB-S5 arm reads
the LOBSTER-format orderbook CSVs of a LOB-S5 generation tree, `--l5-dir` (or `$BOOKDIT_L5_DIR`);
both `<dir>/data_gen/*.csv` and a flat `<dir>/*.csv` layout are accepted.

Usage
    python scripts/analysis/conditional_prediction.py --frames <dump dir> --sources real,bookdit
    python scripts/analysis/conditional_prediction.py --frames <dump dir> \
        --sources real,bookdit,lobs5 --l5-dir <LOB-S5 tree> --horizons 10,20
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from book_channels import _book_channels  # noqa: E402  (shared book-state channel definitions)

ROOT = Path(__file__).resolve().parents[2]


def _l5_csv_dir(l5_dir: Path) -> Path:
    """LOB-S5 orderbook CSVs: `<dir>/data_gen/` for a LOB-Bench-layout tree, else `<dir>/` itself
    (the flat layout `lobs5_gen_to_lobster.py --out` writes)."""
    return l5_dir / "data_gen" if (l5_dir / "data_gen").is_dir() else l5_dir

CHANNELS = ["dmid", "absdmid", "spread", "imb", "ofi", "sofi",
            "lasz", "lbsz", "ladep", "lbdep"]


# ------------------------------------------------------------------ featurisation
def _acf(x: np.ndarray, lag: int) -> float:
    x = x - x.mean()
    d = float((x * x).sum())
    return float((x[:-lag] * x[lag:]).sum() / d) if d > 0 and len(x) > lag else 0.0


def featurise(book: np.ndarray) -> np.ndarray:
    """Summary vector of a context window of book frames (book is (W, 40))."""
    ch = _book_channels(book)
    out = []
    for k in CHANNELS:
        # raw BookDiT frames contain negative sizes -> log1p returns NaN. Clean per element so a
        # single bad frame does not wipe out the whole summary statistic.
        x = np.nan_to_num(np.asarray(ch[k], dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        ch[k] = x
        out += [x.mean(), x.std(), *np.quantile(x, [0.1, 0.5, 0.9]),
                x[-1], x[-5:].mean(), _acf(x, 1), _acf(x, 2), _acf(x, 5)]
    s, d = ch["sofi"], ch["dmid"]
    for lag in (1, 5, 20):                       # flow -> price response
        out.append(float((s[:-lag] * d[lag:]).mean()) if len(s) > lag else 0.0)
    for k in (5, 10):                            # variance ratio of the price path
        if len(d) >= 4 * k:
            agg = d[:len(d) // k * k].reshape(-1, k).sum(1)
            v = d.var()
            out.append(float(agg.var() / (k * v)) if v > 0 else 1.0)
        else:
            out.append(1.0)
    out.append(float(d.sum()))                   # net drift over the window
    return np.nan_to_num(np.asarray(out, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)


TARGETS = ["dmid", "imb", "dimb", "rvol", "spread"]


def targets(book: np.ndarray, mid0: float, h: int) -> np.ndarray:
    """Book quantities h frames after the end of the context window.

    `dmid` and `imb` are the two specified targets.  Both are nearly unpredictable even
    real-on-real (mid is a martingale; L1 imbalance decorrelates within ~100 messages), which
    leaves every cell of the matrix near R^2 ~ 0.  `dimb` (the same imbalance read over all 10
    levels), `rvol` and `spread` are the graded companions: they are strongly predictable
    real-on-real, so a transfer gap has room to show up.
    """
    b = book[h - 1]
    mid = 0.5 * (float(b[0]) + float(b[2]))
    bs, as_ = float(b[3]), float(b[1])
    bd, ad = float(b[3::4][:10].sum()), float(b[1::4][:10].sum())
    m = 0.5 * (book[:h, 0] + book[:h, 2])
    rvol = float(np.abs(np.diff(m)).mean()) / 100.0 if h > 1 else 0.0
    return np.array([(mid - mid0) / 100.0,
                     (bs - as_) / max(bs + as_, 1.0),
                     (bd - ad) / max(bd + ad, 1.0),
                     rvol,
                     (float(b[0]) - float(b[2])) / 100.0])


def _snap(a: np.ndarray) -> np.ndarray:
    a = a.copy()
    a[:, 0::2] = np.round(a[:, 0::2] / 100.0) * 100.0
    a[:, 1::2] = np.maximum(np.round(a[:, 1::2]), 0.0)
    return a


# ------------------------------------------------------------------ per-window worker
def _one(job):
    npz, sources, snap, horizons, ctxW, selfW, l5_csv = job
    date, rid = npz.stem.split("_rid")
    try:
        z = np.load(npz)
        cond = z["cond_book"].astype(np.float64)
        books = {"real": z["real_book"].astype(np.float64),
                 "bookdit": z["gen_book"].astype(np.float64)}
        if "lobs5" in sources:
            f = l5_csv / f"GOOG_{date}_orderbook_real_id_{rid}_gen_id_0.csv"
            g = pd.read_csv(f, header=None).to_numpy(np.float64)
            books["lobs5"] = g[::10]                        # stride-10 grid, NO padding
        if snap:
            cond = _snap(cond)
            books = {k: _snap(v) for k, v in books.items()}
        hb = [h for h in horizons if selfW + h <= 50]        # design B lives inside 50 frames
        need = max(max(horizons), selfW + max(hb) if hb else 0)
        for s in sources:
            if len(books[s]) < need:
                return None
        mid0 = 0.5 * (cond[-1, 0] + cond[-1, 2])
        rec = {"date": date, "rid": rid,
               "XA": featurise(cond[-ctxW:]),
               "XB": {s: featurise(books[s][:selfW]) for s in sources},
               "yA": {}, "yB": {}}
        for s in sources:
            bb = books[s]
            m0b = 0.5 * (bb[selfW - 1, 0] + bb[selfW - 1, 2])
            for h in horizons:
                rec["yA"][(s, h)] = targets(bb, mid0, h)
            for h in hb:
                rec["yB"][(s, h)] = targets(bb[selfW:], m0b, h)
        return rec
    except Exception:
        return None


# ------------------------------------------------------------------ transfer matrix
def _fit_ridge(Xtr, ytr, gtr, alphas):
    """Ridge with alpha picked by inner day-grouped CV on the TRAINING fold only."""
    sc = StandardScaler().fit(Xtr)
    Z = sc.transform(Xtr)
    ng = len(set(gtr.tolist()))
    if ng >= 3:
        best, ba = -np.inf, alphas[len(alphas) // 2]
        for a in alphas:
            sc_ = []
            for itr, ite in GroupKFold(n_splits=min(3, ng)).split(Z, ytr, gtr):
                m = Ridge(alpha=a).fit(Z[itr], ytr[itr])
                sc_.append(r2_score(ytr[ite], m.predict(Z[ite])))
            if np.mean(sc_) > best:
                best, ba = np.mean(sc_), a
    else:
        ba = alphas[len(alphas) // 2]
    return sc, Ridge(alpha=ba).fit(Z, ytr), ba


def transfer_matrix(X: dict, Y: dict, days: np.ndarray, sources: list[str],
                    n_splits: int = 3, alphas=None, signed: bool = False) -> dict:
    """X[s] (n,d) features of source s; Y[s] (n,) labels of source s.  Pooled OOF predictions."""
    alphas = alphas if alphas is not None else np.logspace(-1, 7, 17)
    n = len(days)
    pred = {(a, b): np.zeros(n) for a in sources for b in sources}
    alpha_used = {a: [] for a in sources}
    for tr, te in GroupKFold(n_splits=n_splits).split(np.zeros(n), groups=days):
        for a in sources:
            sc, mdl, al = _fit_ridge(X[a][tr], Y[a][tr], days[tr], alphas)
            alpha_used[a].append(float(al))
            for b in sources:
                pred[(a, b)][te] = mdl.predict(sc.transform(X[b][te]))
    out = {"alpha": {a: alpha_used[a] for a in sources}, "cells": {}}
    for a in sources:
        for b in sources:
            yb, pb = Y[b], pred[(a, b)]
            nz = (yb != 0) if signed else np.zeros(len(yb), bool)
            acc = float((np.sign(pb[nz]) == np.sign(yb[nz])).mean()) if nz.sum() else float("nan")
            base = float(max((yb[nz] > 0).mean(), (yb[nz] < 0).mean())) if nz.sum() else float("nan")
            out["cells"][f"{a}->{b}"] = {
                "r2": float(r2_score(yb, pb)), "corr": float(np.corrcoef(yb, pb)[0, 1]),
                "sign_acc": acc, "sign_base": base, "n_nonzero": int(nz.sum()),
                "y_sd": float(yb.std()), "pred_sd": float(pb.std())}
    return out


# ------------------------------------------------------------------ main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", required=True,
                    help="a gen_bookdit_on_lobbench_windows.py --dump-frames directory")
    ap.add_argument("--l5-dir", default=os.environ.get("BOOKDIT_L5_DIR"),
                    help="LOB-S5 generation tree for the lobs5 source (default $BOOKDIT_L5_DIR)")
    ap.add_argument("--sources", default="real,bookdit")
    ap.add_argument("--horizons", default="10,30,50", help="frames ahead (1 frame = 10 messages)")
    ap.add_argument("--ctxW", type=int, default=100, help="design-A context frames (from cond)")
    ap.add_argument("--selfW", type=int, default=20, help="design-B context frames (from continuation)")
    ap.add_argument("--snap", action="store_true", help="quantization control: tick-round + clip")
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--out", default=str(ROOT / "runs/condpred"))
    a = ap.parse_args()

    sources = [s for s in a.sources.split(",") if s]
    horizons = [int(h) for h in a.horizons.split(",") if h]
    frames = Path(a.frames)
    files = sorted(frames.glob("*.npz"))
    if not files:
        raise SystemExit(f"no *.npz frame dumps under {frames}")
    l5_csv = None
    if "lobs5" in sources:
        if not a.l5_dir:
            raise SystemExit("--sources includes lobs5: pass --l5-dir (or set BOOKDIT_L5_DIR)")
        l5_csv = _l5_csv_dir(Path(a.l5_dir))
        if not any(l5_csv.glob("*orderbook*.csv")):
            raise SystemExit(f"no LOB-S5 orderbook CSVs under {l5_csv}")
    if a.limit:
        files = files[:: max(1, len(files) // a.limit)][:a.limit]
    jobs = [(f, sources, a.snap, horizons, a.ctxW, a.selfW, l5_csv) for f in files]
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        recs = [r for r in ex.map(_one, jobs, chunksize=32) if r is not None]
    if not recs:
        raise SystemExit(f"no usable windows among {len(files)} dumps in {frames}"
                         + (f" (LOB-S5 CSVs searched in {l5_csv})" if l5_csv else ""))
    days = np.array([r["date"] for r in recs])
    print(f"windows usable {len(recs)}/{len(files)}  days {len(set(days.tolist()))}  "
          f"sources {sources}  snap={a.snap}", flush=True)

    XA = np.stack([r["XA"] for r in recs])
    res = {"sources": sources, "horizons": horizons, "snap": bool(a.snap), "ctxW": a.ctxW,
           "selfW": a.selfW, "folds": a.folds, "n_windows": len(recs),
           "n_days": int(len(set(days.tolist()))), "n_files_seen": len(files),
           "feat_dim": int(XA.shape[1]), "rids": [r["rid"] for r in recs], "results": {}}
    for design in ("A", "B"):
        X = {s: XA for s in sources} if design == "A" else \
            {s: np.stack([r["XB"][s] for r in recs]) for s in sources}
        key = "yA" if design == "A" else "yB"
        hs = horizons if design == "A" else [h for h in horizons if a.selfW + h <= 50]
        for h in hs:
            for ti, tname in enumerate(TARGETS):
                Y = {s: np.array([r[key][(s, h)][ti] for r in recs]) for s in sources}
                m = transfer_matrix(X, Y, days, sources, a.folds, signed=(tname == "dmid"))
                res["results"][f"{design}|h{h}|{tname}"] = m
                ref = m["cells"][f"{sources[0]}->{sources[0]}"]["r2"]
                line = "  ".join(f"{k}={v['r2']:+.4f}" for k, v in m["cells"].items())
                print(f"[{design} h={h:2d} {tname:5s}] ref={ref:+.4f} | {line}", flush=True)

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    tag = a.tag or ("-".join(sources) + f"_h{'-'.join(map(str, horizons))}"
                    + ("_snap" if a.snap else "_raw"))
    p = out / f"condpred_{tag}.json"
    p.write_text(json.dumps(res, indent=2))
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
