#!/usr/bin/env python
"""Generate BookDiT continuations on the OFFICIAL LOB-Bench evaluation windows.

The LOB-Bench release (HuggingFace `peernagy/lob_bench`) ships each competitor's `data_gen` for a
fixed set of GOOG Jan-2023 windows; `scripts/baselines/build_lobbench_hf_real.py` rebuilds the
matching `data_real`/`data_cond`. This script puts BookDiT on the same windows.

BookDiT's generative unit is a book frame taken every `--stride` (=10) messages, not a single
message, so a 500-message LOB-Bench horizon is 50 generated frames and the observed context is the
250 frames (2,500 messages) preceding the window; see `docs/PROTOCOL.md` §3. The generated frames
are IDM-decoded back to LOBSTER messages; `--mode floor` decodes the REAL frames of the same
windows instead, which measures the representation and decode cost on its own.
"""
from __future__ import annotations

import argparse, re, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "lib"))
from _paths import lob_bench_dir, lobster_cache                               # noqa: E402
sys.path.insert(0, str(lob_bench_dir(required=True)))

from reconstruct_event_book import Book, frame_trade_aggregates               # noqa: E402
from idm_decoder import decode_book_sequence                                  # noqa: E402
from make_model_family import FAMILY, build_family_model                      # noqa: E402
from lob_rollout import flow_rollout                                          # noqa: E402
from lob_representation import (book_to_midspread_dt, midspread_dt_to_book,   # noqa: E402
                                midspread_dt_anchors, book_to_midspread_dt_logvol,
                                midspread_dt_logvol_to_book, levels_of, N_TRADE_CH,
                                book_to_midspread_dt_logvol_trades,
                                midspread_dt_logvol_trades_to_book,
                                book_to_chart_dt, chart_dt_to_book, CHART_ABLATIONS)

# the checkpoint's rep decides the encoding; hardcoding midspread_dt silently feeds a log-volume
# model out-of-distribution inputs.  The paper uses midspread_dt (the paper's checkpoint) and the
# chart ablations below; the log-volume and trade-channel reps are experimental extensions that no
# reported result uses.
#
# UNIFORM SIGNATURE.  The trades rep's encoder takes an extra (...,T,4) argument and its decoder
# returns a 3-tuple, so both are wrapped to the (book, times[, trades]) -> array and
# array -> (book, times, trades) shape used at the call sites; the book-only reps report zero
# trade columns.  `TRADE_REPS` says which reps consume/produce them.
TRADE_REPS = ("midspread_dt_logvol_trades",)
DT_ENC = {
    "midspread_dt": lambda b, t, tr=None: book_to_midspread_dt(b, t),
    "midspread_dt_logvol": lambda b, t, tr=None: book_to_midspread_dt_logvol(b, t),
    "midspread_dt_logvol_trades": book_to_midspread_dt_logvol_trades,
}
DT_DEC = {
    "midspread_dt": lambda h, m, t: (*midspread_dt_to_book(h, m, t), None),
    "midspread_dt_logvol": lambda h, m, t: (*midspread_dt_logvol_to_book(h, m, t), None),
    "midspread_dt_logvol_trades": midspread_dt_logvol_trades_to_book,
}

# CHART ABLATIONS (one structural guarantee removed per rep; see lob_representation.CHART_ABLATIONS).
# Registered through the same uniform signature as everything else so that generation, rollout and
# refinement code paths are byte-identical across the ladder and only the coordinate chart differs.
for _r, _parts in CHART_ABLATIONS.items():
    DT_ENC[_r] = (lambda p: lambda b, t, tr=None: book_to_chart_dt(b, t, p))(_parts)
    DT_DEC[_r] = (lambda p: lambda h, m, t: (*chart_dt_to_book(h, m, t, p), None))(_parts)

VERYDEEP_IDX = [4 * L + 1 for L in range(3, 10)] + [4 * L + 3 for L in range(3, 10)]


def refine(books, w=3):
    """Paper's optional refinement stage: local median filter over the DEEP resting-size channels
    (levels 4-10, sizes only), leaving prices and the dynamic near-touch untouched."""
    out = books.copy().astype(np.float64)
    sz = out[:, VERYDEEP_IDX]
    pad = w // 2
    szp = np.pad(sz, ((pad, pad), (0, 0)), mode="edge")
    sm = np.stack([np.median(szp[t:t + w], axis=0) for t in range(sz.shape[0])])
    out[:, VERYDEEP_IDX] = np.clip(np.round(sm), 0, None)
    return out

MKT_OPEN, MKT_CLOSE = 34200, 57600
KEEP_TYPES = (1, 2, 3, 4)


def csv_dir() -> Path:
    """LOBSTER message-CSV root, `$LOBSTER_CACHE/csv`.

    Resolved on first use, not at import: `horizon_official_protocol.py` imports `refine` from
    this module and must not require the LOBSTER cache to be configured just to do that."""
    return lobster_cache() / "csv"


def day_windows(day, date_str, cond_dir: Path, stride, n_ctx, n_gen, horizon_msgs, levels=10,
                with_trades=False):
    """For each official window of `day`: book frames for the observed context + the real
    continuation, on the model's stride-`stride` grid. Returns (keys, ctx_books, ctx_times,
    real_books, real_times, ctx_trades, real_trades); the two trade arrays are None unless
    `with_trades`.

    Grid note.  Frames step by `stride` over the filtered stream (types 1-4, market hours), which
    is not the corpus grid (`k % stride == 0` over the raw stream).  It matters here because
    the trade aggregates must span the raw messages between frames -- type-5 hidden executions are
    absent from the filtered stream entirely -- so they are computed from the raw indices `F[pos]`
    by `reconstruct_event_book.frame_trade_aggregates`, which is the same definition the corpus
    builder uses (verified frame-exact on GOOG 20240123)."""
    msg = pd.read_csv(f"{csv_dir()}/{day}/GOOG_message.csv", header=None, usecols=range(6),
                      names=["t", "type", "oid", "size", "price", "dir"])
    t_all = msg.t.to_numpy(np.float64); typ = msg.type.to_numpy(np.int64)
    keep = np.isin(typ, KEEP_TYPES) & (t_all >= MKT_OPEN) & (t_all <= MKT_CLOSE)
    F = np.flatnonzero(keep); tF = t_all[F]

    keys, starts = [], []
    for f in sorted(cond_dir.glob(f"*{date_str}*message*.csv")):
        rid = int(re.search(r"real_id_(\d+)\.csv", f.name).group(1))
        t_last = float(pd.read_csv(f, header=None, usecols=[0]).iloc[-1, 0])
        p = int(np.searchsorted(tF, t_last)) + 1        # first message of the continuation
        if p - stride * n_ctx < 0 or p + horizon_msgs > len(F):
            continue
        keys.append((date_str, rid)); starts.append(p)

    # frame positions (in the filtered stream) for context and for the real continuation
    ctx_pos = [np.array([p - stride * (n_ctx - k) for k in range(n_ctx)]) for p in starts]
    gen_pos = [np.array([p + stride * k for k in range(n_gen)]) for p in starts]
    needed = set()
    for a, b in zip(ctx_pos, gen_pos):
        needed.update(F[a].tolist()); needed.update(F[b].tolist())

    bk = Book(); snaps = {}
    size = msg["size"].to_numpy(np.int64); px = msg.price.to_numpy(np.int64)
    dr = msg["dir"].to_numpy(np.int64)
    for k in range(len(t_all)):
        bk.apply(typ[k], size[k], px[k], dr[k])
        if k in needed:
            snaps[k] = bk.snapshot(n=levels)

    grab = lambda pos: (np.stack([snaps[k] for k in F[pos]]), t_all[F[pos]])   # (n, 4*levels)
    ctx_b, ctx_t, gen_b, gen_t, ctx_q, gen_q = [], [], [], [], [], []
    for a, b in zip(ctx_pos, gen_pos):
        cb, ct = grab(a); gb, gt = grab(b)
        ctx_b.append(cb); ctx_t.append(ct); gen_b.append(gb); gen_t.append(gt)
        if with_trades:
            # RAW indices, so the interval sweep sees type-5 hidden executions too
            ctx_q.append(frame_trade_aggregates(typ, size, dr, F[a]))
            gen_q.append(frame_trade_aggregates(typ, size, dr, F[b]))
    return (keys, np.stack(ctx_b), np.stack(ctx_t), np.stack(gen_b), np.stack(gen_t),
            np.stack(ctx_q) if with_trades else None,
            np.stack(gen_q) if with_trades else None)


@torch.no_grad()
def generate(model, ctx_b, ctx_t, mean, std, n_gen, gen_steps, device, seed, rollout_fn,
             rep="midspread_dt", ctx_tr=None):
    """One flow-inpainting pass over a (2*n_ctx)-frame window: context observed, suffix generated.

    Returns (gen_books, gen_times, gen_trades); gen_trades is None for the book-only reps.
    `ctx_tr` is the (K, n_ctx, 4) observed trade aggregate, required by the trades reps."""
    K, n_ctx, n_cols = ctx_b.shape          # 40 cols at 10 levels, 80 at 20, 120 at 30
    T = 2 * n_ctx
    b = np.zeros((K, T, n_cols)); b[:, :n_ctx] = ctx_b
    t = np.zeros((K, T)); t[:, :n_ctx] = ctx_t
    tr = None
    if rep in TRADE_REPS:
        if ctx_tr is None:
            raise ValueError(f"rep {rep!r} needs conditioning trade aggregates; pass ctx_tr")
        tr = np.zeros((K, T, N_TRADE_CH)); tr[:, :n_ctx] = ctx_tr
    a_mid, a_t = midspread_dt_anchors(b, t)
    x = torch.tensor((DT_ENC[rep](b, t, tr) - mean) / std, dtype=torch.float32, device=device)
    g, _ = rollout_fn(model, x, gen_steps, observe_frac=0.5, seed=seed)
    gb, gt, gq = DT_DEC[rep](g.cpu().numpy() * std + mean, a_mid, a_t)
    sl = slice(n_ctx, n_ctx + n_gen)
    return gb[:, sl], gt[:, sl], (gq[:, sl] if gq is not None else None)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True,
                    help="BookDiT checkpoint (from scripts/lib/train_bookdit.py)")
    ap.add_argument("--cond-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--days", nargs="+", required=True)
    ap.add_argument("--mode", choices=["model", "floor"], default="model",
                    help="floor: IDM-decode the REAL frames instead of generating (representation tax)")
    ap.add_argument("--stride", type=int, default=10, help="messages per book frame (model grid)")
    ap.add_argument("--levels", type=int, default=10, help="book levels the representation carries")
    ap.add_argument("--horizon-msgs", type=int, default=500)
    ap.add_argument("--refine", action="store_true",
                    help="apply the paper's deep-size median-filter refinement before decoding")
    ap.add_argument("--cold-queue", action="store_true",
                    help="decode the horizon in isolation (no order history); biases lifecycle stats")
    ap.add_argument("--dump-frames", default=None, metavar="DIR",
                    help="Write the NATIVE book frames (pre-IDM) as one .npz per window and skip "
                         "decoding entirely.  The frames are what the model actually outputs; "
                         "everything else on disk is a replay of IDM-decoded messages, which puts "
                         "the inverse map in BookDiT's path and nothing else's.  With this dump a "
                         "state-space comparison never calls the IDM and needs no decode floor.")
    ap.add_argument("--decode-levels", type=int, default=0, metavar="L",
                    help="Book levels handed to the IDM. 0 (default) = ALL generated levels, which "
                         "is what the 10- vs 20-level comparison (paper, 'State representation') "
                         "used: the decoder infers depth from width, seeds its queue from every "
                         "level and diffs every level, while still emitting a 40-column book. "
                         "Setting 10 restricts it to the top ten and DISCARDS that information -- "
                         "it does not buy comparability with the paper's 10-level checkpoint, "
                         "whose decode is identical either way.")
    ap.add_argument("--no-trade-decode", action="store_true",
                    help="For a trades checkpoint: still GENERATE the four trade channels but do "
                         "not feed them to the IDM, so the decode is the book-only rule. This is "
                         "the arm that separates 'did the extra channels change the book?' from "
                         "'does the trade-aware decode pay off?'; without it the two changes are "
                         "confounded.")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--gen-steps", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=True)
    mean, std = ck["mean"].numpy(), ck["std"].numpy()
    t_win = int(ck.get("seq_len", 100)); n_ctx = t_win // 2
    ck_levels = levels_of(ck["rep"], ck["mean"].numel())   # rep-aware (midspread_dt: 41 -> 10)
    needs_trades = ck["rep"] in TRADE_REPS
    if a.mode == "model" and ck_levels != a.levels:
        print(f"[levels] checkpoint carries {ck_levels} levels; overriding --levels {a.levels}", flush=True)
        a.levels = ck_levels
    n_gen = a.horizon_msgs // a.stride
    model = rfn = None
    if a.mode == "model":
        model = build_family_model(ck.get("arch", "wandit"),
                                   torch.zeros(1, t_win, ck["mean"].numel()),
                                   FAMILY[ck["size"]]).to(a.device)
        model.load_state_dict(ck["model_state_dict"]); model.eval()
        rfn = flow_rollout

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    written, t0 = 0, time.time()
    for day in a.days:
        ds = f"{day[:4]}-{day[4:6]}-{day[6:]}"
        keys, cb, ct, rb, rt, cq, rq = day_windows(day, ds, Path(a.cond_dir), a.stride, n_ctx,
                                                   n_gen, a.horizon_msgs, levels=a.levels,
                                                   with_trades=needs_trades)
        print(f"[{ds}] {len(keys)} windows | ctx {n_ctx} frames = {n_ctx*a.stride} msgs, "
              f"gen {n_gen} frames = {n_gen*a.stride} msgs", flush=True)
        for s in range(0, len(keys), a.batch):
            e = min(s + a.batch, len(keys))
            if a.mode == "model":
                gb, gt, gq = generate(model, cb[s:e], ct[s:e], mean, std, n_gen, a.gen_steps,
                                      a.device, a.seed + 1 + s, rfn, rep=ck["rep"],
                                      ctx_tr=cq[s:e] if needs_trades else None)
            else:
                gb, gt = rb[s:e], rt[s:e]
                gq = rq[s:e] if needs_trades else None
            if a.refine:
                # Refinement is DEFINED on levels 4-10 sizes (VERYDEEP_IDX). At 20/30 levels we
                # refine that same 40-column prefix rather than extrapolating the index set, so the
                # operation stays identical to the published arm; the deeper levels are dropped by
                # LOB-Bench's level-10 filter anyway.
                if gb.shape[-1] < 40:
                    raise ValueError(f"refinement needs >=40 book columns, got {gb.shape[-1]}")
                gb = gb.copy()
                gb[..., :40] = np.stack([refine(gb[i, :, :40]) for i in range(len(gb))])
            if a.dump_frames:
                # native frames only: context, real continuation, generated continuation
                fd = Path(a.dump_frames); fd.mkdir(parents=True, exist_ok=True)
                for i in range(e - s):
                    date, rid = keys[s + i]
                    extra = {} if gq is None else {
                        "cond_trades": cq[s + i].astype(np.float64),
                        "real_trades": rq[s + i].astype(np.float64),
                        "gen_trades": gq[i].astype(np.float64)}
                    np.savez_compressed(fd / f"{date}_rid{rid}.npz",
                                        cond_book=cb[s + i].astype(np.int64),
                                        cond_time=ct[s + i].astype(np.float64),
                                        real_book=rb[s + i].astype(np.int64),
                                        real_time=rt[s + i].astype(np.float64),
                                        gen_book=gb[i].astype(np.int64),
                                        gen_time=gt[i].astype(np.float64), **extra)
                    written += 1
                print(f"  dumped {written} windows (frames, no decode)", flush=True)
                continue
            for i in range(e - s):
                date, rid = keys[s + i]
                # Decode depth: see --decode-levels. Default 0 keeps every generated level, the
                # convention the depth ablation established.
                dc = 4 * a.decode_levels if a.decode_levels else cb.shape[-1]
                d_cb, d_gb = cb[s + i][:, :dc], gb[i][:, :dc]
                # gen_trades drive execution-vs-cancel labelling when the decoder supports it
                use_q = gq is not None and not a.no_trade_decode
                q_gen = gq[i] if use_q else None
                q_cond = cq[s + i] if use_q else None
                if a.cold_queue:
                    mg, bg = decode_book_sequence(d_gb, frame_times=gt[i], frame_trades=q_gen)
                else:
                    # Seed the IDM's FIFO queue by decoding the conditioning window first, so orders
                    # resting before the horizon have real lifetimes; then keep only the horizon's
                    # messages. Without this, no order can outlive the 500-message window and the
                    # lifecycle statistics (time-to-cancel, cancel levels/depths) are badly biased.
                    t_split = ct[s + i][-1]
                    mg, bg = decode_book_sequence(
                        np.concatenate([d_cb, d_gb], axis=0),
                        frame_times=np.concatenate([ct[s + i], gt[i]]),
                        frame_trades=(np.concatenate([q_cond, q_gen], axis=0) if use_q else None))
                    keep = (mg["time"].to_numpy() > t_split)
                    mg, bg = mg[keep].reset_index(drop=True), bg[keep].reset_index(drop=True)
                if len(mg) < 8:
                    continue
                mg[["time", "event_type", "order_id", "size", "price", "direction"]].to_csv(
                    out / f"GOOG_{date}_message_real_id_{rid}_gen_id_0.csv", header=False, index=False)
                pd.DataFrame(bg).to_csv(out / f"GOOG_{date}_orderbook_real_id_{rid}_gen_id_0.csv",
                                        header=False, index=False)
                written += 1
        print(f"  [{ds}] cumulative {written} written ({time.time()-t0:.0f}s)", flush=True)
    print(f"DONE: {written} windows -> {out}")


if __name__ == "__main__":
    main()
