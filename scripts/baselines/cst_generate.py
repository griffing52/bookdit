#!/usr/bin/env python
"""CST baseline, stage 3 (JAX env): generate per-window continuations with the native CST
simulator (cst.make_step_book_scannable — pure JAX, no jaxlob/AlphaTrade needed), conditioned on
the real book state after each window's 500 cond events. Writes gen_{wi}.npy with columns
[time, event_type, size, price, direction] (the exact format score_lobs5_retrained.py ingests)
plus a copy of meta.json, so scoring reuses the LOB-S5 head-to-head harness verbatim.

Run with the isolated JAX venv, not this repo's venv.
"""
from __future__ import annotations
import argparse, json, shutil, sys
from pathlib import Path
import numpy as np
import jax, jax.numpy as jnp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "lib"))
from _paths import lob_bench_dir                              # noqa: E402

_LB = lob_bench_dir(required=True)
sys.path.insert(0, str(_LB.parent))
sys.path.insert(0, str(_LB / "cst_model"))
import cst                                                    # noqa: E402
from param_estimation import load_params                     # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", default="runs/lobbench/baseline_cst/aggregated_params.pkl")
    ap.add_argument("--prep", default="runs/lobbench/baseline_cst/prep")
    ap.add_argument("--out", default="runs/lobbench/baseline_cst/gen")
    ap.add_argument("--n-levels", type=int, default=10)
    ap.add_argument("--gen", type=int, default=0, help="override gen length (0 = use meta)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    prep = ROOT / args.prep if not Path(args.prep).is_absolute() else Path(args.prep)
    out = ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    meta = json.loads((prep / "meta.json").read_text())
    gen_n = args.gen if args.gen > 0 else int(meta["gen"])
    init_books = np.load(prep / "init_books.npy")            # (K,40) int
    init_times = np.load(prep / "init_times.npy")            # (K,) float
    K = len(init_books)

    aggr = load_params(str(ROOT / args.params if not Path(args.params).is_absolute() else Path(args.params)))
    params, lo_lambda, co_theta = cst.init_params(aggr)
    base_rates = cst.get_event_base_rates(params, cancel_rates=co_theta, lo_rates=lo_lambda)
    step = cst.make_step_book_scannable(args.n_levels)

    n_events = []
    for wi in range(K):
        book = cst.init_book(jnp.asarray(init_books[wi], dtype=jnp.int32), params, float(init_times[wi]))
        rng = jax.random.PRNGKey(args.seed + wi)
        _carry, (_l2, msgs) = jax.lax.scan(step, (book, base_rates, params, rng), length=gen_n)
        arr = np.stack([
            np.asarray(msgs.time, dtype=np.float64),
            np.asarray(msgs.event_type, dtype=np.float64),
            np.asarray(msgs.size, dtype=np.float64),
            np.asarray(msgs.price, dtype=np.float64),
            np.asarray(msgs.direction, dtype=np.float64),
        ], axis=1)
        np.save(out / f"gen_{wi}.npy", arr)
        n_events.append(len(arr))
    shutil.copy(prep / "meta.json", out / "meta.json")
    print(f"[cst-gen] K={K} gen_n={gen_n} events/window={np.mean(n_events):.0f} -> {out}")


if __name__ == "__main__":
    main()
