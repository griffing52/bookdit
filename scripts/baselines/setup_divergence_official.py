#!/usr/bin/env python
"""Build LOB-Bench-layout dirs for --div_only scoring on the OFFICIAL windows.

Every arm is scored on the SAME window set (paired) with the SAME real reference files, so any
difference in the divergence curve is a difference between the generators, not the protocol.

Arms and their default source trees (relative to $BOOKDIT_WORK/lobbench_repro/lob_bench/
sample_data/data_saved; override with --src ARM=DIR):
  bookdit    hf_bookdit_9d        BookDiT (your checkpoint) generated on LOB-Bench's GOOG cond
                                  windows; $BOOKDIT_DIVERGENCE_SRC overrides the default
  lobs5      hf_lobs5_9d_f10      LOB-S5's released samples, real side level-10 filtered the same way
  ditfloor   hf_bookditfloor_9d   IDM decode of the REAL frames (the decoder floor)
  realgen    hf_bookdit_9d        pseudo-model whose data_gen is the REAL continuation

Each arm is written to <out>/<arm>/GOOG (or <out>/<tag>_<arm>/GOOG with --tag); that directory
name is the model name run_divergence_official.sh scores and headline.csv reports.
"""
from __future__ import annotations
import argparse, glob, os, random, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "lib"))
from _paths import bench_work                                                   # noqa: E402

# Generated arms. BOOKDIT_DIVERGENCE_SRC (or --src bookdit=DIR) points the bookdit arm at a
# different generation dir without editing this table.
MODELS = {"bookdit": os.environ.get("BOOKDIT_DIVERGENCE_SRC", "hf_bookdit_9d"), "lobs5": "hf_lobs5_9d_f10",
          "ditfloor": "hf_bookditfloor_9d"}
# "realgen" is a pseudo-model: data_gen := the REAL continuation. Its divergence curve is the
# horizon-resolved floor -- it absorbs the finite-sample penalty of scoring a short sub-sequence
# against the pooled real distribution, which --div_error_bounds (real vs. real, unsplit) does not.
# It borrows the real/cond files of this tree.
REALGEN_SRC = "hf_bookdit_9d"


def src_root() -> Path:
    """Where the per-model generation trees live, under `$BOOKDIT_WORK`. Resolved on use."""
    return bench_work() / "lobbench_repro/lob_bench/sample_data/data_saved"


def window_ids(model_dir: Path) -> set[tuple[str, str]]:
    out = set()
    for f in glob.glob(str(model_dir / "GOOG" / "data_gen" / "*message*")):
        b = os.path.basename(f)
        out.add((b.split("_")[1], b.split("real_id_")[1].split("_gen")[0]))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=None,
                    help="default <repo>/runs/divergence/data, the tree "
                         "run_divergence_official.sh scores")
    ap.add_argument("--per-day", type=int, default=0, help="0 = all windows")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="",
                    help="optional prefix: arms are written as <tag>_<arm> instead of <arm>")
    ap.add_argument("--only", nargs="*", default=None,
                    help="build just these arms (names of MODELS, or 'realgen')")
    ap.add_argument("--mirror", default=None,
                    help="reuse the exact window set of an already-built arm dir under --out "
                         "(e.g. lobs5), so an arm built later is paired with the earlier ones")
    ap.add_argument("--src", nargs="*", default=[], metavar="ARM=DIR",
                    help="override an arm's source tree; DIR is absolute or relative to the "
                         "data_saved root")
    a = ap.parse_args()
    out_root = Path(a.out) if a.out else ROOT / "runs/divergence/data"
    srcs = dict(MODELS, realgen=REALGEN_SRC)
    for kv in a.src:
        k, _, v = kv.partition("=")
        if k not in srcs or not v:
            raise SystemExit(f"--src expects ARM=DIR with ARM in {sorted(srcs)}, got {kv!r}")
        srcs[k] = v

    if a.mirror:
        ids = window_ids(out_root / a.mirror)
        if not ids:
            raise SystemExit(f"--mirror {a.mirror}: no windows under {out_root / a.mirror}/GOOG/data_gen")
    else:
        ids = set.intersection(*(window_ids(src_root() / srcs[m]) for m in MODELS))
    by_day: dict[str, list[str]] = {}
    for d, rid in sorted(ids):
        by_day.setdefault(d, []).append(rid)
    rng = random.Random(a.seed)
    keep = []
    for d in sorted(by_day):
        rids = sorted(by_day[d])
        if a.per_day:
            rng.shuffle(rids)
            rids = sorted(rids[:a.per_day])
        keep += [(d, r) for r in rids]
    print(f"{len(keep)} windows over {len(by_day)} days")

    for name in (a.only or list(MODELS) + ["realgen"]):
        if name not in srcs:
            raise SystemExit(f"unknown arm {name!r}; choose from {sorted(srcs)}")
        src_dir = src_root() / srcs[name]               # an absolute DIR survives the join
        dst = out_root / (f"{a.tag}_{name}" if a.tag else name) / "GOOG"
        for sub in ("data_real", "data_gen", "data_cond"):
            (dst / sub).mkdir(parents=True, exist_ok=True)
        n = 0
        for d, rid in keep:
            ok = True
            links = []
            for sub in ("data_real", "data_cond", "data_gen"):
                suf = "_gen_id_0" if sub == "data_gen" else ""
                for kind in ("message", "orderbook"):
                    fn = f"GOOG_{d}_{kind}_real_id_{rid}{suf}.csv"
                    if name == "realgen" and sub == "data_gen":
                        s = src_dir / "GOOG" / "data_real" / f"GOOG_{d}_{kind}_real_id_{rid}.csv"
                    else:
                        s = src_dir / "GOOG" / sub / fn
                    if not s.exists():
                        raise SystemExit(f"missing {s}")
                    if s.stat().st_size < 2:          # empty CSV crashes Simple_Loader
                        ok = False
                        continue
                    links.append((dst / sub / fn, s.resolve()))
            if not ok:                                # keep the three folders in sync
                continue
            for t, s in links:
                if t.exists() or t.is_symlink():
                    t.unlink()
                t.symlink_to(s)
            n += 1
        print(f"  {name}: {n} windows -> {dst}")


if __name__ == "__main__":
    main()
