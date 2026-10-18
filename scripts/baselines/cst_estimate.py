#!/usr/bin/env python
"""CST baseline, stage 2 (JAX env): estimate Cont-Stoikov-Talreja params from the prepared
per-day (book_df, message_df), using LOB-Bench's OWN param_estimation.estimate_params +
aggregate_params. Writes aggregated_params.pkl.

Run with the isolated JAX venv (jax/chex/scipy), not this repo's venv.
"""
from __future__ import annotations
import argparse, glob, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "lib"))
from _paths import lob_bench_dir                              # noqa: E402

_LB = lob_bench_dir(required=True)
sys.path.insert(0, str(_LB.parent))                          # lob_bench package
sys.path.insert(0, str(_LB / "cst_model"))
import param_estimation                                       # noqa: E402
import cst                                                     # noqa: E402
from cst import CSTParams                                      # noqa: E402
import jax                                                     # noqa: E402
import jax.numpy as jnp                                        # noqa: E402

MSG_COLS = ["time", "event_type", "order_id", "size", "price", "direction"]

# chunked re-implementation of param_estimation.estimate_params: identical math, but the
# per-depth mean-volume (Q_i) vmap over all book rows is streamed in chunks so a full trading
# day (millions of rows x num_ticks) fits in GPU memory instead of OOMing on one giant (N,500) array.
_INIT_BOOK_V = jax.jit(jax.vmap(cst.init_book, in_axes=(0, None)), static_argnums=(1,))


def _mean_size_at_depth(book_values, tick_size, num_ticks, chunk=200_000):
    cp = CSTParams(tick_size=tick_size, num_ticks=num_ticks)
    bvals = jnp.asarray(book_values)
    n = bvals.shape[0]
    sum_asks = jnp.zeros(num_ticks)
    sum_bids = jnp.zeros(num_ticks)
    for i in range(0, n, chunk):
        bj = _INIT_BOOK_V(bvals[i:i + chunk], cp)
        sum_asks = sum_asks + bj.asks.sum(axis=0)
        sum_bids = sum_bids + bj.bids.sum(axis=0)
    return (sum_asks / n + sum_bids / n) / 2


def estimate_params_chunked(book_data, message_data, tick_size, num_ticks):
    total_time = float(message_data.time.iloc[-1] - message_data.time.iloc[0])
    mo_count = message_data[message_data.event_type == 4].shape[0]
    lo_size = jnp.nan_to_num(message_data.loc[message_data.event_type == 1, "size"].mean(), nan=1.)
    mo_size = jnp.nan_to_num(message_data.loc[message_data.event_type == 4, "size"].mean(), nan=1.)
    co_size = jnp.nan_to_num(message_data.loc[message_data.event_type.isin((2, 3)), "size"].mean(), nan=1.)
    mean_spread = ((book_data.iloc[:, 0] - book_data.iloc[:, 2]) / tick_size).mean()
    mo_mu = mo_count / total_time * mo_size / lo_size

    lo_count, co_count = param_estimation._get_counts_by_depth(book_data, message_data, tick_size, num_ticks)
    lo_sol = param_estimation._fit_power_law(lo_count / total_time, plot_fit=False)
    mean_size_at_depth = _mean_size_at_depth(book_data.values, tick_size, num_ticks)

    co_theta = co_count / (mean_size_at_depth * total_time) * (co_size / lo_size)
    co_theta = co_theta.at[jnp.isnan(co_theta)].set(0)
    return {
        "total_time": total_time, "num_events": message_data.shape[0], "mean_spread": mean_spread,
        "lo_size": lo_size, "mo_size": mo_size, "co_size": co_size,
        "lo_lambda": lo_count / total_time, "mo_count": mo_count, "mo_mu": mo_mu,
        "lo_count": lo_count, "co_count": co_count, "q_i": mean_size_at_depth, "co_theta": co_theta,
        "tick_size": tick_size, "num_ticks": num_ticks, "lo_k": lo_sol[0], "lo_alpha": lo_sol[1],
    }


def load_day(npz_path: str):
    z = np.load(npz_path)
    mtime, mrest, book = z["mtime"], z["mrest"], z["book"]
    msg = pd.DataFrame({
        "time": mtime.astype(np.float64),
        "event_type": mrest[:, 0].astype(np.int64),
        "order_id": mrest[:, 1].astype(np.int64),
        "size": mrest[:, 2].astype(np.int64),
        "price": mrest[:, 3].astype(np.int64),
        "direction": mrest[:, 4].astype(np.int64),
    })[MSG_COLS]
    book_df = pd.DataFrame(book.astype(np.int64))             # 40 cols, positional
    return book_df, msg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--param-input", default="runs/lobbench/baseline_cst/prep/param_input")
    ap.add_argument("--out", default="runs/lobbench/baseline_cst/aggregated_params.pkl")
    ap.add_argument("--tick", type=int, default=100)
    ap.add_argument("--num-ticks", type=int, default=500)
    args = ap.parse_args()

    pin = ROOT / args.param_input if not Path(args.param_input).is_absolute() else Path(args.param_input)
    out = ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    files = sorted(glob.glob(str(pin / "*.npz")))
    assert files, f"no .npz in {pin}"
    per_day = []
    for f in files:
        book_df, msg = load_day(f)
        pd_ = estimate_params_chunked(book_df, msg, tick_size=args.tick, num_ticks=args.num_ticks)
        per_day.append(pd_)
        print(f"[est] {Path(f).stem}: n_events={int(pd_['num_events'])} "
              f"mo_mu={float(pd_['mo_mu']):.4f} lo_k={float(pd_['lo_k']):.3f} "
              f"lo_alpha={float(pd_['lo_alpha']):.3f} lo_size={float(pd_['lo_size']):.1f} "
              f"mean_spread={float(pd_['mean_spread']):.2f}", flush=True)

    aggr = param_estimation.aggregate_params(per_day, save_path=str(out))
    print(f"\n[CST params] aggregated over {len(files)} days -> {out}")
    print(f"  mo_mu={float(aggr['mo_mu']):.4f}  lo_k={float(aggr['lo_k']):.3f}  "
          f"lo_alpha={float(aggr['lo_alpha']):.3f}  lo_size={float(aggr['lo_size']):.1f}  "
          f"mean_spread={float(aggr['mean_spread']):.2f}")
    print(f"  lo_lambda[:5]={np.asarray(aggr['lo_lambda'])[:5]}")
    print(f"  co_theta[:5]={np.asarray(aggr['co_theta'])[:5]}")


if __name__ == "__main__":
    main()
