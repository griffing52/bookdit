#!/usr/bin/env python
"""Train the 500-context BookDiT world model on a streaming corpus.

Days are converted to the representation on the fly inside DataLoader workers, and each
worker holds a small rotating pool of days, so RAM is bounded by
``num_workers * pool_days * day_bytes`` rather than by the corpus size.  Channel mean/std
come from a precomputed exact streaming pass over the train split
(``scripts/lib/build_rep_norm_stats.py``). Validation uses a fixed non-overlapping grid over
the val split. Training runs for exactly ``--steps`` Adam steps (``train_to_budget``).

Provenance of the checkpoint reported in the paper (not shipped with this
repository). It was not written by this script but by an earlier run of the same recipe: the same model (``mini``: embed 128, depth 4,
953,769 parameters), the same flow-matching objective, optimizer, learning rate and
``train_to_budget`` loop, with 109,216 steps x batch 32 x 500 frames = 1,747,456,000 training
frames (a 1e16-FLOP budget at 6 x parameters x frames), seed 1, validation every 1,000 steps.
That run differed from this script in two ways:

* its loader streamed one whole day per worker in float32, where this one keeps a pool of
  ``--pool-days`` days per worker in float16;
* its channel mean/std were estimated from a random 200-day subsample of the train split
  (50,987,615 frames, sampling seed 0), where this script reads the exact full-split statistics.

Its mean/std are stored inside the checkpoint (``mean``/``std`` keys), as in every checkpoint this
script writes, so evaluation needs no stats file. A re-run of the command below therefore matches
the paper's model in recipe, not bit for bit.

Usage (the paper's recipe; the defaults below are the same values)
    python scripts/lib/train_bookdit.py \
        --manifest runs/crossasset/corpus_manifest_2024.csv \
        --norm-stats runs/crossasset/norm_midspread_dt.npz \
        --rep midspread_dt --size mini --steps 109216 --seed 1 --out <ckpt>
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset, get_worker_info

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "lib"))
sys.path.insert(0, str(ROOT / "src"))

from make_model_family import FAMILY, build_family_model                     # noqa: E402
from lob_representation import (book_to_midspread_dt,
                                book_to_midspread_dt_logvol,                  # noqa: E402
                                book_to_midspread_dt_logvol_trades,
                                book_to_chart_dt, CHART_ABLATIONS)

REPS = (("midspread_dt", "midspread_dt_logvol", "midspread_dt_logvol_trades")
        + tuple(CHART_ABLATIONS))


def day_rep(contig: Path, rep: str) -> np.ndarray | None:
    """(T,C) float32 representation for one contig day; None if its per-day trade file is missing."""
    d = torch.load(contig, map_location="cpu", weights_only=True)
    book = d["books"].float().numpy()
    tday = d["times"].double().numpy()
    if book.ndim != 2 or book.shape[1] % 4:
        raise ValueError(f"{contig.name}: books must have shape (T, 4*levels), got {book.shape}")
    if len(tday) != len(book):
        raise ValueError(f"{contig.name}: time/book frame mismatch: {len(tday)} vs {len(book)}")
    if rep == "midspread_dt":            # absolute sizes (the paper's model's rep)
        return book_to_midspread_dt(book, tday)
    if rep in CHART_ABLATIONS:           # chart ablation ladder; chart_full == midspread_dt
        return book_to_chart_dt(book, tday, CHART_ABLATIONS[rep])
    if rep == "midspread_dt_logvol":
        return book_to_midspread_dt_logvol(book, tday)
    if rep != "midspread_dt_logvol_trades":
        raise ValueError(f"unknown representation: {rep}")
    if "trades" not in d:
        return None
    trades = d["trades"].float().numpy()
    if trades.shape != (len(book), 4):
        raise ValueError(f"{contig.name}: trades must have shape ({len(book)}, 4), "
                         f"got {trades.shape}")
    return book_to_midspread_dt_logvol_trades(book, tday, trades)


def _portable(path: str) -> str:
    """Repo-relative when inside the checkout, so a shared checkpoint carries no home path."""
    p = Path(path).resolve()
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(path)


def manifest_rows(manifest: str, split: str) -> list[Path]:
    return [Path(r["tensor_path"]) for r in csv.DictReader(open(manifest))
            if r.get("split") == split and Path(r["tensor_path"]).exists()]


class StreamingRandomOffsetWindows(IterableDataset):
    """Infinite stream of standardized (seq_len, C) fp16 windows, bounded RAM.

    Sharded across DataLoader workers by day.  Each worker keeps `pool_days` days resident and
    yields a random-offset window from a random pool slot; a day leaves the pool once it has
    yielded its quota of `len(day) // seq_len` windows, which makes one pass over the day list
    equal to ~one epoch over the corpus frames.
    """

    def __init__(self, files, seq_len, seed, rep, mean, std, pool_days=4):
        self.files, self.seq_len, self.seed, self.rep = list(files), seq_len, seed, rep
        self.mean, self.std, self.pool_days = mean, std, pool_days

    def _load(self, f, rng):
        x = day_rep(f, self.rep)
        if x is None or len(x) < self.seq_len + 1:
            return None
        x = ((x - self.mean) / self.std).astype(np.float16)
        return [x, max(1, len(x) // self.seq_len)]

    def __iter__(self):
        wi = get_worker_info()
        wid, nw = (wi.id, wi.num_workers) if wi else (0, 1)
        files = self.files[wid::nw]
        rng = np.random.default_rng(self.seed * 100003 + wid)
        pool, order, ptr, fails = [], list(rng.permutation(len(files))), 0, 0
        while True:
            while len(pool) < self.pool_days:
                if ptr >= len(order):                        # epoch boundary -> reshuffle
                    order, ptr = list(rng.permutation(len(files))), 0
                f = files[order[ptr]]
                ptr += 1
                e = self._load(f, rng)
                if e is None:
                    fails += 1
                    if fails > 5 * len(files):
                        raise RuntimeError(f"worker {wid}: no loadable day in {fails} attempts "
                                           f"(missing per-day trade files?)")
                    continue
                fails = 0
                pool.append(e)
            k = int(rng.integers(len(pool)))
            x, quota = pool[k]
            s = int(rng.integers(0, len(x) - self.seq_len + 1))
            yield torch.from_numpy(x[s:s + self.seq_len].copy())
            pool[k][1] -= 1
            if pool[k][1] <= 0:
                pool.pop(k)


class GridWindows(Dataset):
    """Fixed non-overlapping validation windows."""

    def __init__(self, days, seq_len):
        self.items = [(di, i * seq_len) for di, d in enumerate(days)
                      for i in range(len(d) // seq_len)]
        self.days, self.seq_len = days, seq_len

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        di, s = self.items[i]
        return torch.from_numpy(self.days[di][s:s + self.seq_len].astype(np.float32))


@torch.no_grad()
def evaluate(model, loader, device, max_batches):
    model.eval()
    tot, n = 0.0, 0
    for i, x in enumerate(loader):
        if i >= max_batches:
            break
        out = model(x.to(device).float())
        tot += float(out["loss"]); n += 1
    return tot / max(1, n)


def loader_cycle(loader):
    """Yield batches from a streaming loader forever (iso-budget training)."""
    while True:
        yield from loader


def train_to_budget(model, train_loader, val_loader, *, device, lr, steps,
                    eval_every, eval_batches, tail_frac=0.15, grad_clip=1.0):
    """Train for exactly ``steps`` optimizer steps; report tail-mean TRAIN loss
    (smoothed over the last ``tail_frac``) plus shared-val loss for reference."""
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    tail_start = max(1, int(steps * (1.0 - tail_frac)))
    tail, best_val, last_val = [], float("inf"), float("nan")
    it = loader_cycle(train_loader)
    for step in range(1, steps + 1):
        x = next(it)
        model.train()
        opt.zero_grad(set_to_none=True)
        loss = model(x.to(device).float())["loss"]
        loss.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        lv = float(loss.detach())
        if step >= tail_start:
            tail.append(lv)
        if step % eval_every == 0 or step == steps:
            v = evaluate(model, val_loader, device, eval_batches)
            best_val = min(best_val, v)
            last_val = v
    train_loss = sum(tail) / max(1, len(tail))
    return {"train_loss": train_loss, "best_val_loss": best_val,
            "final_val_loss": last_val, "steps": steps}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default=str(ROOT / "runs/crossasset/corpus_manifest_2024.csv"))
    ap.add_argument("--norm-stats", default=str(ROOT / "runs/crossasset/norm_midspread_dt.npz"),
                    help="combined npz from build_rep_norm_stats.py for the same --rep (a "
                         "midspread_dt_logvol_trades file may also be used for --rep "
                         "midspread_dt_logvol, via its 41-channel prefix)")
    ap.add_argument("--rep", default="midspread_dt", choices=REPS)
    ap.add_argument("--size", default="mini", choices=sorted(FAMILY))
    # "wandit" is the arch name stored in the paper's checkpoint; both resolve to BookDiT.
    ap.add_argument("--arch", default="bookdit", choices=["bookdit", "wandit"])
    ap.add_argument("--seq-len", type=int, default=500)
    ap.add_argument("--steps", type=int, default=109216)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--eval-batches", type=int, default=40)
    ap.add_argument("--num-workers", type=int, default=6)
    ap.add_argument("--pool-days", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke", type=int, default=0, help="run this many steps and exit (no ckpt)")
    args = ap.parse_args()

    z = np.load(args.norm_stats)
    if "mean" not in z or "std" not in z or z["mean"].ndim != 1 or z["std"].ndim != 1:
        raise SystemExit(f"invalid norm stats {args.norm_stats}: expected 1-D mean and std")
    if len(z["mean"]) != len(z["std"]):
        raise SystemExit(f"invalid norm stats {args.norm_stats}: mean/std lengths differ")
    # A stats file for another rep would silently standardize with the wrong statistics (e.g.
    # log-size moments applied to absolute sizes), so refuse it. The one allowed mismatch is the
    # trades rep's 41-channel prefix, which is exactly midspread_dt_logvol's statistics.
    stats_rep = str(z["rep"]) if "rep" in z else None
    prefix_ok = stats_rep == "midspread_dt_logvol_trades" and args.rep == "midspread_dt_logvol"
    if stats_rep is not None and stats_rep != args.rep and not prefix_ok:
        raise SystemExit(f"norm stats {args.norm_stats} were built for --rep {stats_rep}, "
                         f"not {args.rep}; run build_rep_norm_stats.py --rep {args.rep}")

    torch.manual_seed(args.seed)
    t0 = time.time()
    train_files = manifest_rows(args.manifest, "train")
    val_files = manifest_rows(args.manifest, "val")
    C = None
    mean = std = None
    val_days = []
    expected = ({"midspread_dt": 1, "midspread_dt_logvol": 1, "midspread_dt_logvol_trades": 5}
                | {r: 1 for r in CHART_ABLATIONS})[args.rep]
    for f in val_files:
        x = day_rep(f, args.rep)
        if x is None:
            continue
        if C is None:
            C = x.shape[-1]
            if x.ndim != 2 or (C - expected) % 4:
                raise SystemExit(f"invalid inferred shape {x.shape} for {args.rep}")
            if len(z["mean"]) < C or (len(z["mean"]) != C and not prefix_ok
                                      and not (stats_rep is None
                                               and args.rep == "midspread_dt_logvol")):
                raise SystemExit(f"norm stats have {len(z['mean'])} channels but "
                                 f"{args.rep} requires {C}")
            mean = z["mean"][:C].astype(np.float32)
            std = z["std"][:C].astype(np.float32)
            if (not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std))
                    or np.any(std <= 0)):
                raise SystemExit(f"invalid norm stats {args.norm_stats}: non-finite mean/std "
                                 "or nonpositive std")
        elif x.ndim != 2 or x.shape[-1] != C:
            raise SystemExit(f"inconsistent val representation dimension for {f}: "
                             f"got {x.shape}, expected (*,{C})")
        val_days.append(((x - mean) / std).astype(np.float16))
    if not val_days:
        raise SystemExit("no val days")
    ev_val = sum(len(d) for d in val_days)
    print(f"rep={args.rep} C={C}  train: {len(train_files)} day files | "
          f"val: {len(val_days)} days, {ev_val/1e6:.1f}M events ({time.time()-t0:.0f}s)",
          flush=True)

    model = build_family_model(args.arch, torch.zeros(1, args.seq_len, C), FAMILY[args.size])
    n_enc = sum(p.numel() for p in model.parameters())   # checkpoint key name kept as "n_enc"
    steps = args.smoke or args.steps
    tokens = steps * args.batch_size * args.seq_len
    flop_budget = 6.0 * n_enc * tokens
    print(f"{args.arch} {args.size} seq_len={args.seq_len}: params={n_enc/1e6:.2f}M steps={steps} "
          f"tokens={tokens:.2e} C~{flop_budget:.1e}", flush=True)

    ds = StreamingRandomOffsetWindows(train_files, args.seq_len, args.seed, args.rep, mean, std,
                                      args.pool_days)
    train_loader = DataLoader(ds, batch_size=args.batch_size, num_workers=args.num_workers,
                              persistent_workers=args.num_workers > 0,
                              prefetch_factor=4 if args.num_workers > 0 else None)
    val_loader = DataLoader(GridWindows(val_days, args.seq_len), batch_size=args.batch_size,
                            shuffle=False, num_workers=0)
    res = train_to_budget(model, train_loader, val_loader, device=args.device, lr=args.lr,
                          steps=steps, eval_every=min(args.eval_every, steps),
                          eval_batches=args.eval_batches)
    wall = (time.time() - t0) / 60
    print(f"train_loss={res['train_loss']:.5f}  best_val={res['best_val_loss']:.5f}  "
          f"({wall:.1f} min)", flush=True)
    if args.smoke:
        return 0

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "rep": args.rep, "size": args.size,
                "arch": args.arch, "mean": torch.from_numpy(mean), "std": torch.from_numpy(std),
                "seq_len": args.seq_len, "n_enc": n_enc, "steps": res["steps"], "tokens": tokens,
                "flop_budget": flop_budget, "seed": args.seed,
                "manifest": _portable(args.manifest),
                "norm_stats": _portable(args.norm_stats), "wall_min": wall,
                "train_loss": res["train_loss"], "best_val_loss": res["best_val_loss"]}, out)
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
