#!/usr/bin/env python
"""CST baseline, stage 1 (.venv): turn proc npys into (a) param-estimation inputs and
(b) per-window CST initial-book conditioning, using the SAME validated QueueBook/build_stream
that the LOB-S5 retrain scoring path uses (so books are row-aligned + LOBSTER-consistent).

Outputs under --out:
  param_input/{day}.npz   arrays: mtime(f8,N), mrest(i8,N x5 [etype,oid,size,price,dir]), book(i8,N x40)
  init_books.npy (K x40 i8), init_times.npy (K f8), meta.json {proc,cond,gen,warmup,model_starts}

The JAX-env stages (cst_estimate.py, cst_generate.py) consume these; scoring reuses
score_lobs5_retrained.py --gen-only, so CST plugs into the identical head-to-head harness as LOB-S5.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "lib"))
from _paths import bench_work, lob_bench_dir                  # noqa: E402
sys.path.insert(0, str(lob_bench_dir(required=True)))
from score_lobs5_retrained import proc_events, build_stream   # validated QueueBook path


def day_proc(proc_dir: Path, day: str) -> Path:
    return proc_dir / f"GOOG_{day}_message_proc.npy"


def build_day_arrays(proc, a, b):
    """proc rows [a,b) -> (mtime f8, mrest i8[N,5], book i8[N,40]) via build_stream."""
    events = proc_events(proc, a, b)
    msg_df, book_df = build_stream([], events)          # cold start; early unmatched reduces dropped
    m = msg_df.to_numpy()                                # cols: time,event_type,order_id,size,price,direction
    mtime = m[:, 0].astype(np.float64)
    mrest = m[:, [1, 2, 3, 4, 5]].astype(np.int64)
    book = book_df.to_numpy().astype(np.int64)
    return mtime, mrest, book


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proc-dir", default=None,
                    help="LOB-S5 preprocessing output (GOOG_<day>_message_proc.npy); "
                         "default $BOOKDIT_WORK/data_goog2024/proc")
    ap.add_argument("--train-days", nargs="+",
                    default=["20240123", "20240124", "20240125", "20240126", "20240129", "20240131"])
    ap.add_argument("--gen-day", default="20240131")
    ap.add_argument("--out", default="runs/lobbench/baseline_cst/prep")
    ap.add_argument("--cond", type=int, default=500)
    ap.add_argument("--gen", type=int, default=500)
    ap.add_argument("--warmup", type=int, default=8000)
    ap.add_argument("--K", type=int, default=40)
    ap.add_argument("--train-cap", type=int, default=0,
                    help="if >0, cap events per training day (0 = full day)")
    ap.add_argument("--starts-from", default=None,
                    help="meta.json (e.g. from gen_lobs5) to reuse proc+cond+gen+warmup+model_starts "
                         "for IDENTICAL windows; skips param_input (params fitted separately)")
    args = ap.parse_args()

    def proc_dir() -> Path:
        """--proc-dir, else the LOB-S5 preprocessing output under $BOOKDIT_WORK. Resolved on use,
        so --starts-from runs that carry their own `proc` need no work root at all."""
        return Path(args.proc_dir) if args.proc_dir else bench_work() / "data_goog2024/proc"

    out = ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)

    # ---- shared-window mode: condition CST on ANOTHER model's exact windows ----
    if args.starts_from:
        src = json.loads(Path(args.starts_from).read_text())
        gp = Path(src["proc"]) if "proc" in src else day_proc(proc_dir(), src["day"])
        proc = np.load(gp, allow_pickle=True).astype(np.int64)
        cond, gen, warmup = src["cond"], src["gen"], src.get("warmup", args.warmup)
        starts = [int(s) for s in src["model_starts"]]
        out.mkdir(parents=True, exist_ok=True)
        init_books, init_times = [], []
        for s in starts:
            warm = proc_events(proc, s - warmup, s)
            cond_ev = proc_events(proc, s, s + cond)
            _m, bdf = build_stream(warm, cond_ev)
            init_books.append(bdf.to_numpy()[-1].astype(np.int64))
            init_times.append(float(cond_ev[-1][0]))
        np.save(out / "init_books.npy", np.asarray(init_books, dtype=np.int64))
        np.save(out / "init_times.npy", np.asarray(init_times, dtype=np.float64))
        meta = {"proc": str(gp), "cond": cond, "gen": gen, "warmup": warmup,
                "model_starts": starts, "gen_day": src.get("day", "")}
        (out / "meta.json").write_text(json.dumps(meta, indent=2))
        print(f"[cst-shared] reused {len(starts)} starts from {args.starts_from} -> {out}")
        return

    (out / "param_input").mkdir(parents=True, exist_ok=True)

    # ---- (a) param-estimation inputs: full training days ----
    for day in args.train_days:
        p = day_proc(proc_dir(), day)
        proc = np.load(p, allow_pickle=True).astype(np.int64)
        b = len(proc) if args.train_cap <= 0 else min(len(proc), args.train_cap)
        mtime, mrest, book = build_day_arrays(proc, 0, b)
        np.savez(out / "param_input" / f"{day}.npz", mtime=mtime, mrest=mrest, book=book)
        print(f"[param] {day}: {len(mtime):,} rows (of {len(proc):,} proc) -> {day}.npz", flush=True)

    # ---- (b) per-window init books on the gen day ----
    gp = day_proc(proc_dir(), args.gen_day)
    proc = np.load(gp, allow_pickle=True).astype(np.int64)
    N = len(proc)
    lo, hi = args.warmup, N - (args.cond + args.gen) - 1
    starts = np.linspace(lo, hi, args.K).astype(np.int64).tolist()

    init_books, init_times = [], []
    for s in starts:
        warm = proc_events(proc, s - args.warmup, s)
        cond_ev = proc_events(proc, s, s + args.cond)
        _m, bdf = build_stream(warm, cond_ev)
        init_books.append(bdf.to_numpy()[-1].astype(np.int64))   # book AFTER the 500 cond events
        init_times.append(float(cond_ev[-1][0]))                  # last cond event time (seconds)
    np.save(out / "init_books.npy", np.asarray(init_books, dtype=np.int64))
    np.save(out / "init_times.npy", np.asarray(init_times, dtype=np.float64))

    meta = {"proc": str(gp), "cond": args.cond, "gen": args.gen,
            "warmup": args.warmup, "model_starts": starts, "gen_day": args.gen_day}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[cond] gen_day={args.gen_day} K={len(starts)} starts -> init_books.npy, meta.json")
    print(f"wrote prep to {out}")


if __name__ == "__main__":
    main()
