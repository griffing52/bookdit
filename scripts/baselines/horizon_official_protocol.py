#!/usr/bin/env python
"""Long-horizon stability profile on the official protocol.

Generated blocks are scored against the real LOBSTER message stream, the same reference the
official-window comparison uses.  Conditioning windows are LOB-Bench's own; the real
continuation is read forward from the raw message file for as many blocks as the horizon needs.
"""
from __future__ import annotations
import argparse, re, shutil, sys
from pathlib import Path
import numpy as np, pandas as pd, torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "lib"))
from _paths import lob_bench_dir                                              # noqa: E402
sys.path.insert(0, str(lob_bench_dir(required=True)))
from reconstruct_event_book import Book, frame_trade_aggregates                # noqa: E402
from lob_representation import levels_of, N_TRADE_CH, midspread_dt_anchors     # noqa: E402
import sys as _sys; _sys.path.insert(0, str(ROOT / "scripts" / "baselines"))
from gen_bookdit_on_lobbench_windows import (csv_dir, refine as _refine,       # noqa: E402
                                             TRADE_REPS, DT_ENC, DT_DEC)
from idm_decoder import decode_book_sequence                                   # noqa: E402
from make_model_family import FAMILY, build_family_model                       # noqa: E402
from lob_rollout import flow_rollout                                           # noqa: E402
import data_loading, scoring                                                   # noqa: E402
from run_bench import DEFAULT_SCORING_CONFIG, DEFAULT_METRICS                  # noqa: E402

MKT_OPEN, MKT_CLOSE = 34200, 57600
KEEP = (1, 2, 3, 4)
CLOCK_STATS = {"log_inter_arrival_time", "log_time_to_cancel", "vol_per_min"}
BOOK10_STATS = (
    "spread", "orderbook_imbalance", "ask_volume_touch", "bid_volume_touch",
    "ask_volume", "bid_volume", "ofi", "ofi_up", "ofi_stay", "ofi_down",
)


def load_day(day, date_str, cond_dir, stride, n_ctx, horizon, limit, levels=10, cond_messages=500):
    """Context frames on the model grid + the real message stream for `horizon` messages forward."""
    msg = pd.read_csv(f"{csv_dir()}/{day}/GOOG_message.csv", header=None, usecols=range(6),
                      names=["t", "type", "oid", "size", "price", "dir"])
    t_all = msg.t.to_numpy(np.float64); typ = msg.type.to_numpy(np.int64)
    keep = np.isin(typ, KEEP) & (t_all >= MKT_OPEN) & (t_all <= MKT_CLOSE)
    F = np.flatnonzero(keep); tF = t_all[F]

    keys, starts = [], []
    for f in sorted(Path(cond_dir).glob(f"*{date_str}*message*.csv")):
        rid = int(re.search(r"real_id_(\d+)\.csv", f.name).group(1))
        t_last = float(pd.read_csv(f, header=None, usecols=[0]).iloc[-1, 0])
        p = int(np.searchsorted(tF, t_last)) + 1
        if p - stride * n_ctx < 0 or p + horizon >= len(F):
            continue
        keys.append((date_str, rid)); starts.append(p)
    if limit: keys, starts = keys[:limit], starts[:limit]

    ctx_pos = [np.array([p - stride * (n_ctx - k) for k in range(n_ctx)]) for p in starts]
    need = set()
    for a, p in zip(ctx_pos, starts):
        need.update(F[a].tolist()); need.update(F[p:p + horizon].tolist())
        need.update(F[p - cond_messages:p].tolist())

    bk = Book(); snaps = {}
    size = msg["size"].to_numpy(np.int64); px = msg.price.to_numpy(np.int64); dr = msg["dir"].to_numpy(np.int64)
    for k in range(len(t_all)):
        bk.apply(typ[k], size[k], px[k], dr[k])
        if k in need: snaps[k] = bk.snapshot(n=levels)
    return keys, starts, F, msg, snaps, ctx_pos


@torch.no_grad()
def rollout_chunked_dt(model, real_books, real_times, mean, std, cond_events, gen_steps,
                       device, seed=1, t_win=500, rollout_fn=flow_rollout, rep="midspread_dt",
                       real_trades=None, return_trades=False, obs_frames=None):
    """Chunked autoregressive rollout threading a per-event TIME array. The model input is the
    41-col midspread_dt rep (book + log1p(Δt)); the observed prefix pins BOTH book and Δt to the
    real path, and the generated suffix produces book AND timing.

    For a trades rep the four per-frame trade-aggregate channels are threaded the SAME way, in a
    `trseq` array parallel to `tseq`: chunk i's generated trade counts become part of chunk i+1's
    observed prefix, so the rollout stays self-consistent instead of re-reading real trades after
    the conditioning window. `real_trades` (K,T,4) supplies the observed prefix and is required for
    those reps.

    Returns (books (K,T,4L), times (K,T)), or that plus trades (K,T,4) when `return_trades`."""
    K, T_tot, _ = real_books.shape
    # `obs_frames` is how many already-resolved frames each chunk re-observes; the remaining
    # t_win - obs_frames are generated and the rollout advances by that many.  The default,
    # t_win // 2, is the published half-window hop (observe_frac 0.5, advance 250, slice
    # [250:500]).  Raising it shortens the hop and so increases the number of autoregressive
    # block boundaries per generated message.
    step = t_win // 2 if obs_frames is None else int(obs_frames)
    if not 0 < step < t_win:
        raise ValueError(f"obs_frames must be in (0, {t_win}), got {step}")
    seq, tseq = real_books.copy(), real_times.copy()
    seq[:, cond_events:] = 0.0
    tseq[:, cond_events:] = 0.0                         # regenerated; anchor comes from observed frame 0
    trseq = None
    if rep in TRADE_REPS:
        if real_trades is None:
            raise ValueError(f"rep {rep!r} needs real_trades (K,T,4) for the observed prefix")
        trseq = np.asarray(real_trades, dtype=np.float64).copy()
        if trseq.shape != (K, T_tot, N_TRADE_CH):
            raise ValueError(f"real_trades must be {(K, T_tot, N_TRADE_CH)}, got {trseq.shape}")
        trseq[:, cond_events:] = 0.0
    pos, ci = cond_events, 0
    while pos < T_tot:
        # The first chunk can only re-observe as many frames as the conditioning prefix actually
        # holds.  With the default step (t_win//2 == cond_events in the published protocol) this is
        # exactly `step` and the clamp is inert; with obs_frames > cond_events, `pos - step` would
        # otherwise go NEGATIVE and silently wrap to the end of the array.
        start = max(0, pos - step)
        obs_here = pos - start
        bwin = seq[:, start: start + t_win]                     # (K,t_win,4L)
        twin = tseq[:, start: start + t_win]                    # (K,t_win)
        qwin = None if trseq is None else trseq[:, start: start + t_win]
        a_mid, a_t = midspread_dt_anchors(bwin, twin)           # window frame-0 mid & time (observed)
        x = torch.tensor((DT_ENC[rep](bwin, twin, qwin) - mean) / std,
                         dtype=torch.float32, device=device)
        g, _ = rollout_fn(model, x, gen_steps, observe_frac=obs_here / t_win, seed=seed + ci)
        gbook, gtimes, gtr = DT_DEC[rep](g.cpu().numpy() * std + mean, a_mid, a_t)
        n_new = min(t_win - obs_here, T_tot - pos)
        seq[:, pos:pos + n_new] = gbook[:, obs_here:obs_here + n_new]
        tseq[:, pos:pos + n_new] = gtimes[:, obs_here:obs_here + n_new]
        if trseq is not None:
            trseq[:, pos:pos + n_new] = gtr[:, obs_here:obs_here + n_new]
        pos += n_new
        ci += 1
    return (seq, tseq, trseq) if return_trades else (seq, tseq)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True,
                    help="BookDiT checkpoint (from scripts/lib/train_bookdit.py)")
    ap.add_argument("--cond-dir", required=True)
    ap.add_argument("--day", default="20230103")
    ap.add_argument("--refine", action="store_true",
                    help="apply the paper's deep-size median-filter refinement before decoding, "
                         "so this profile matches Table 1's headline arm")
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--levels", type=int, default=None)
    ap.add_argument("--horizon", type=int, default=16000, help="generated messages")
    ap.add_argument("--block", type=int, default=500, help="messages per scored block")
    ap.add_argument("--K", type=int, default=64)
    ap.add_argument("--obs-frames", type=int, default=None,
                    help="frames each rollout chunk re-observes (default t_win//2 = the "
                         "published half-window hop). Larger values shorten the hop and so "
                         "increase the number of autoregressive block boundaries per message.")
    ap.add_argument("--gen-steps", type=int, default=100)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=str(ROOT / "runs/horizon/bookdit"))
    ap.add_argument("--dump-rollout", default=None, help="optional native-array diagnostic .npz")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    ds = f"{a.day[:4]}-{a.day[4:6]}-{a.day[6:]}"

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=True)
    mean, std = ck["mean"].numpy(), ck["std"].numpy()
    ck_levels = levels_of(ck["rep"], len(mean))    # rep-aware (midspread_dt: 41 channels -> 10 levels)
    needs_trades = ck["rep"] in TRADE_REPS
    if a.levels is None:
        a.levels = ck_levels
    elif a.levels != ck_levels:
        raise ValueError(f"checkpoint has {ck_levels} levels, requested {a.levels}")
    t_win = int(ck.get("seq_len", 500)); n_ctx = t_win // 2
    n_gen = a.horizon // a.stride
    model = build_family_model(ck.get("arch", "wandit"),
                               torch.zeros(1, t_win, ck["mean"].numel()),
                               FAMILY[ck["size"]]).to(a.device)
    model.load_state_dict(ck["model_state_dict"]); model.eval()
    rfn = flow_rollout

    keys, starts, F, msg, snaps, ctx_pos = load_day(
        a.day, ds, a.cond_dir, a.stride, n_ctx, a.horizon, a.K, levels=a.levels, cond_messages=a.block)
    K = len(keys)
    print(f"{K} official windows | ctx {n_ctx} frames ({n_ctx*a.stride} msgs) | horizon {a.horizon} msgs "
          f"= {n_gen} frames", flush=True)

    cb = np.stack([np.stack([snaps[k] for k in F[p]]) for p in ctx_pos])
    ct = np.stack([msg.t.to_numpy()[F[p]] for p in ctx_pos])
    # The model was trained on t_win frames.  Even a shorter requested horizon must retain the
    # full latent suffix; truncating the first inpainting window changes (and can destroy) samples.
    padded_n_gen = max(n_gen, t_win // 2)
    T = n_ctx + padded_n_gen
    b = np.zeros((K, T, 4 * a.levels)); b[:, :n_ctx] = cb
    t = np.zeros((K, T)); t[:, :n_ctx] = ct
    q = None
    if needs_trades:
        # Same definition as the corpus builder, computed over RAW message indices so type-5
        # hidden executions are counted (they are absent from the filtered stream F).
        typ = msg.type.to_numpy(np.int64); sz = msg["size"].to_numpy(np.int64)
        dr = msg["dir"].to_numpy(np.int64)
        cq = np.stack([frame_trade_aggregates(typ, sz, dr, F[p]) for p in ctx_pos])
        q = np.zeros((K, T, N_TRADE_CH)); q[:, :n_ctx] = cq
    with torch.no_grad():
        seq, tseq, trseq = rollout_chunked_dt(model, b, t, mean, std, n_ctx, a.gen_steps, a.device,
                                              seed=1, t_win=t_win, rollout_fn=rfn, rep=ck["rep"],
                                              real_trades=q, return_trades=True,
                                              obs_frames=a.obs_frames)
    if a.dump_rollout:
        np.savez_compressed(a.dump_rollout, seq=seq, times=tseq,
                            keys=np.asarray(keys, dtype=object), n_ctx=n_ctx, n_gen=n_gen)
    if a.refine:
        # The headline arm is BookDiT + refinement (Table 1's 'BookDiT + refinement' row).  Applied to the
        # generated frames BEFORE decoding, exactly as gen_bookdit_on_lobbench_windows does.
        # VERYDEEP_IDX is defined on levels 4-10 sizes. At 20/30 levels we refine that same
        # 40-column prefix rather than extrapolating the index set, so the operation stays
        # identical to the published arm; deeper levels are dropped by the level-10 filter anyway.
        if seq.shape[-1] < 40:
            raise ValueError(f"refinement needs >=40 book columns, got {seq.shape[-1]}")
        seq[:, n_ctx:, :40] = np.stack([_refine(seq[i, n_ctx:, :40]) for i in range(len(seq))])
        print("applied refinement to the generated frames", flush=True)
    print(f"rolled out {n_gen} frames x {K} windows", flush=True)

    cols = ["t", "type", "oid", "size", "price", "dir"]
    fpb = a.block // a.stride                                   # frames per scored block
    rows = []
    for bi in range(n_gen // fpb):
        base = out / f"block_{bi}"
        if base.exists(): shutil.rmtree(base)
        for sub in ("data_real", "data_gen", "data_cond"): (base / sub).mkdir(parents=True)
        kept = 0
        for i, (date, rid) in enumerate(keys):
            g0, g1 = n_ctx + bi * fpb, n_ctx + (bi + 1) * fpb
            # Warm the IDM queue with observed context and prior generated frames, matching the
            # validated 500-message scorer instead of decoding each block from an empty queue.
            # Decode at TEN levels: a deeper book would seed the IDM queue with 3x the resting
            # liquidity and emit messages at levels 11-30 that the level-10 filter then removes,
            # giving a different message mix from the published arm.
            mg, bg = decode_book_sequence(
                seq[i, :g1, :40], frame_times=tseq[i, :g1],
                frame_trades=(trseq[i, :g1] if trseq is not None else None),
                deadband_from=n_ctx)   # never smooth the observed prefix
            t_lo = tseq[i, g0 - 1]
            t_hi = tseq[i, g1 - 1]
            in_block = ((mg["time"].to_numpy() > t_lo) &
                        (mg["time"].to_numpy() <= t_hi))
            mg = mg.loc[in_block].reset_index(drop=True)
            bg = bg.loc[in_block].reset_index(drop=True)
            if len(mg) < 8: continue
            if i == 0:
                print(f"    block {bi}: first decoded window has {len(mg)} messages", flush=True)
            p = starts[i]; r0, r1 = p + bi * a.block, p + (bi + 1) * a.block   # REAL messages
            ridx = F[r0:r1]
            mr = msg.iloc[ridx][cols]; br = np.stack([snaps[k] for k in ridx]).astype(np.int64)
            cidx = F[p - a.block:p]
            mc = msg.iloc[cidx][cols]; bc = np.stack([snaps[k] for k in cidx]).astype(np.int64) \
                if all(k in snaps for k in cidx) else None
            if bc is None: continue
            for sub, m_, bk_ in (("data_real", mr, br), ("data_cond", mc, bc)):
                m_.to_csv(base / sub / f"GOOG_{date}_message_real_id_{rid}.csv", header=False, index=False)
                pd.DataFrame(bk_).to_csv(base / sub / f"GOOG_{date}_orderbook_real_id_{rid}.csv",
                                         header=False, index=False)
            mg[cols[1:]].insert(0, "time", mg["time"]) if False else None
            mg[["time", "event_type", "order_id", "size", "price", "direction"]].to_csv(
                base / "data_gen" / f"GOOG_{date}_message_real_id_{rid}_gen_id_0.csv", header=False, index=False)
            pd.DataFrame(bg).to_csv(base / "data_gen" / f"GOOG_{date}_orderbook_real_id_{rid}_gen_id_0.csv",
                                    header=False, index=False)
            kept += 1
        # level-10 filter, matching the official protocol
        for sub in ("data_real", "data_gen", "data_cond"):
            tmp = base / f"{sub}_f10"; tmp.mkdir(exist_ok=True)
            data_loading.cut_data_to_lvl(str(base / sub) + "/", str(tmp) + "/", 10)
            n_msg = len(list((base / sub).glob("*message*.csv")))
            print(f"    block {bi} {sub}: {n_msg} files after level-10 filter", flush=True)
            shutil.rmtree(base / sub); tmp.rename(base / sub)
        # the level filter can empty a file; an empty CSV crashes Simple_Loader's time parser
        for sub in ("data_real", "data_gen", "data_cond"):
            for f in (base / sub).glob("*.csv"):
                if f.stat().st_size < 2: f.unlink()
        gen_ids = {f.name.split("_real_id_")[1].split("_gen")[0] for f in (base / "data_gen").glob("*message*")}
        print(f"    block {bi}: {len(gen_ids)}/{kept} generated windows retained", flush=True)
        for sub in ("data_real", "data_cond"):          # keep the three folders in sync
            for f in (base / sub).glob("*.csv"):
                if f.name.split("_real_id_")[1].split(".")[0] not in gen_ids: f.unlink()
        loader = data_loading.Simple_Loader(str(base / "data_real"), str(base / "data_gen"),
                                            str(base / "data_cond"))
        for s in loader: s.materialize()
        sc, _, _ = scoring.run_benchmark(loader, DEFAULT_SCORING_CONFIG, default_metric=DEFAULT_METRICS)
        vals = {name: float(result["l1"][0]) for name, result in sc.items()}
        l1_21 = float(np.mean(list(vals.values())))
        l1_18 = float(np.mean([value for name, value in vals.items() if name not in CLOCK_STATS]))
        book10 = float(np.mean([vals[name] for name in BOOK10_STATS]))
        n_actual = len(gen_ids)
        rows.append({
            "horizon": (bi + 1) * a.block,
            "l1": l1_21, "l1_21": l1_21, "l1_18": l1_18, "book_10": book10,
            "n_windows": n_actual, "n_generated": kept, "levels": a.levels,
        })
        print(f"  h={(bi+1)*a.block:6d}  L1^21={l1_21:.4f}  L1^18={l1_18:.4f}  Book10={book10:.4f}  ({n_actual}/{kept} windows after/before filter)", flush=True)
        pd.DataFrame(rows).to_csv(out / "profile.csv", index=False)
    print(f"DONE -> {out/'profile.csv'}")


if __name__ == "__main__":
    main()
