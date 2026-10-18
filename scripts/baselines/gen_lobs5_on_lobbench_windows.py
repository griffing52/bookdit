#!/usr/bin/env python
"""Autoregressive message sampling from the retrained LOB-S5 on the official LOB-Bench windows.

Takes explicit window starts (aligned to LOB-Bench's own evaluation windows by timestamp) rather
than choosing them at random, so the compute-matched retrain, the released LOB-S5 samples and
BookDiT are all scored on identical windows. The same driver produces the 500-message continuations
behind Table 1 (--gen 500) and the 16,000-message rollouts behind the long-horizon figure
(--gen 16000).

The AlphaTrade simulator is bypassed: generation is pure token sampling, plus a minimal book (best
bid/ask only) that converts LOB-S5's book-relative price tokens into absolute LOBSTER prices.
Generated messages are written per window; the main venv then rebuilds order ids and books
(lobs5_gen_to_lobster.py) and scores them with LOB-Bench.

--starts-json is a list of {"rid": <LOB-Bench real_id>, "start": <row of --proc where the scored
continuation begins>}; the conditioning window is the --cond rows before "start".

Runs in LOB-S5's own JAX environment, with $LOBS5_DIR pointing at the patched clone
(scripts/data/setup_libs.sh --with-lobs5). Relative paths are resolved against the directory the
script is launched from.
"""
import os, sys, json, time, argparse
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from _paths import lobs5_dir

# $LOBS5_DIR is the patched LOB-S5 clone (patches/lobs5-jax-compat.patch); LOB-S5 resolves its
# own configs relative to the cwd, so we chdir into it before importing anything from `lob`.
# The launch directory is kept so relative CLI paths still mean what the caller meant.
LAUNCH_DIR = os.getcwd()
LOBS5_DIR = str(lobs5_dir())
sys.path.insert(0, LOBS5_DIR)
os.chdir(LOBS5_DIR)

import jax, jax.numpy as jnp
from lob.init_train import load_args_from_checkpoint, init_train_state, load_checkpoint
from lob.encoding import Vocab, Message_Tokenizer, encode_msgs
import lob.encoding as encoding
import lob.validation_helpers as valh

MSG_LEN = Message_Tokenizer.MSG_LEN  # 22
# field-token slices inside one message
TIME_S_START, TIME_S_END = valh.get_idx_from_field('time_s')     # (9,11)
TIME_NS_START, TIME_NS_END = valh.get_idx_from_field('time_ns')  # (11,14)
DT_S_START, DT_S_END = valh.get_idx_from_field('delta_t_s')      # (5,6)
DT_NS_START, DT_NS_END = valh.get_idx_from_field('delta_t_ns')   # (6,9)
TICK = 100


def add_times(a_s, a_ns, b_s, b_ns):
    a_ns = b_ns + a_ns
    extra = a_ns // 1000000000
    a_ns = a_ns % 1000000000
    return a_s + b_s + extra, a_ns


class MiniBook:
    """Minimal aggregate book: tracks best bid/ask for the p_ref (mid) reference only."""
    __slots__ = ("bsz", "asz")
    def __init__(self):
        self.bsz = {}   # price -> size (bids, dir=+1)
        self.asz = {}   # price -> size (asks, dir=-1)
    def apply(self, typ, size, price, direction):
        if typ not in (1, 2, 3, 4):
            return
        szmap = self.bsz if direction == 1 else self.asz
        if typ == 1:
            szmap[price] = szmap.get(price, 0) + size
        else:
            if price in szmap:
                szmap[price] -= size
                if szmap[price] <= 0:
                    del szmap[price]
    def best_bid(self):
        return max(self.bsz) if self.bsz else -1
    def best_ask(self):
        return min(self.asz) if self.asz else -1
    def p_ref(self):
        bid, ask = self.best_bid(), self.best_ask()
        if ask > 0 and bid > 0:
            mid = (ask + bid) / 2
        elif ask < 0 and bid < 0:
            return None
        elif ask < 0:
            mid = bid + TICK
        else:
            mid = ask - TICK
        return int((mid // TICK) * TICK)


def load_model(ckpt_dir, step=0):
    args = load_args_from_checkpoint(ckpt_dir, step=step)
    args.num_devices = 1
    args.bsz = 1   # inference is batch-1; avoid the 32x one-hot init OOM
    v = Vocab(); n_classes = len(v)
    seq_len = args.msg_seq_len * MSG_LEN
    state, model_cls = init_train_state(args, n_classes=n_classes, seq_len=seq_len,
                                        book_dim=0, book_seq_len=0)
    ckpt = load_checkpoint(state, ckpt_dir, vars(args), step=step)
    state = ckpt['model']
    # checkpoint was saved under jax.pmap(1 device) -> strip the leading device axis
    import flax.jax_utils as jax_utils
    lead = jax.tree_util.tree_leaves(state.params)[0].shape
    if len(lead) >= 1 and lead[0] == 1:
        state = jax_utils.unreplicate(state)
        print(f"[model] unreplicated pmap axis (was {lead})", flush=True)
    model = model_cls(training=False, step_rescale=1)
    n_params = sum(x.size for x in jax.tree_util.tree_leaves(state.params))
    print(f"[model] restored; params={n_params} batchnorm={args.batchnorm} mode={args.mode} "
          f"seq_len={seq_len} n_classes={n_classes}", flush=True)
    return state, model, args.batchnorm, v, seq_len


def sample_window(state, model, batchnorm, enc, valid_mask_array, vocab_len,
                  cond_tokens_flat, book, rng, n_gen, sample_top_n):
    """cond_tokens_flat: (seq_len,) tokenized last msg_seq_len cond msgs. book: MiniBook seeded to post-cond.
    Returns list of generated LOBSTER tuples (time, event_type, size, abs_price, direction)."""
    m_seq = jnp.asarray(cond_tokens_flat, dtype=jnp.int32)
    last = m_seq.shape[0] - MSG_LEN
    out = []
    for _ in range(n_gen):
        t_init_s, t_init_ns = encoding.decode_time(m_seq[last + TIME_S_START: last + TIME_NS_END], enc)
        m_seq = valh.append_hid_msg(m_seq)
        for mask_i in range(MSG_LEN):
            if mask_i == TIME_S_START:
                dts = encoding.combine_field(encoding.decode(m_seq[last + DT_S_START:last + DT_S_END], *enc['time']), 3)
                dtns = encoding.combine_field(encoding.decode(m_seq[last + DT_NS_START:last + DT_NS_END], *enc['time']), 3)
                ts, tns = add_times(t_init_s, t_init_ns, dts, dtns)
                ts_tok = encoding.encode(encoding.split_field(ts, 2, 3), *enc['time'])
                tns_tok = encoding.encode(encoding.split_field(tns, 3, 3), *enc['time'])
                m_seq = m_seq.at[last + TIME_S_START: last + TIME_NS_END].set(jnp.hstack([ts_tok, tns_tok]))
            if TIME_S_START <= mask_i < TIME_NS_END:
                continue
            valid_mask = valid_mask_array[mask_i]
            m_seq, _ = valh.mask_last_msg_in_seq(m_seq, mask_i)
            inp = (jax.nn.one_hot(jnp.expand_dims(m_seq, 0), vocab_len).astype(float),)
            its = (jnp.ones((1, m_seq.shape[0])),)
            logits = valh.predict(inp, its, state, model, batchnorm)
            logits = valh.filter_valid_pred(logits, valid_mask)
            rng, rng_ = jax.random.split(rng)
            m_seq = valh.fill_predicted_toks(m_seq, logits, sample_top_n, jnp.array([rng_]))
        dec = np.asarray(encoding.decode_msg(m_seq[-MSG_LEN:], enc))
        etype, dir01, prel, size = int(dec[1]), int(dec[2]), int(dec[4]), int(dec[5])
        time_s, time_ns = int(dec[8]), int(dec[9])
        pref = book.p_ref()
        if pref is None or etype not in (1, 2, 3, 4) or size <= 0:
            continue
        abs_price = pref + prel * TICK
        direction = 2 * dir01 - 1
        book.apply(etype, size, abs_price, direction)
        out.append((time_s + time_ns / 1e9, etype, size, abs_price, direction))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True,
                    help="LOB-S5 checkpoint directory written by run_train.py "
                         "(checkpoints/<run name>_<run id> inside $LOBS5_DIR)")
    ap.add_argument("--step", type=int, default=5,
                    help="checkpoint step; run_train.py saves step = epoch index from 0, so 5 is "
                         "the end of the sixth (final) epoch, the checkpoint every reported "
                         "number uses")
    ap.add_argument("--proc", required=True,
                    help="LOB-S5 preprocessed message array for --day (lob/preproc.py output)")
    ap.add_argument("--day", required=True,
                    help="trading day of --proc, YYYYMMDD (recorded in meta.json)")
    ap.add_argument("--starts-json", dest="starts_json", required=True,
                    help='[{"rid": ..., "start": ...}, ...]; see the module docstring')
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--cond", type=int, default=500)
    ap.add_argument("--gen", type=int, default=500)
    ap.add_argument("--ctx", type=int, default=100)       # msg_seq_len context
    ap.add_argument("--warmup", type=int, default=8000)   # msgs before cond to seed mini-book
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--top_n", type=int, default=-1)      # -1 = full-distribution sampling
    ap.add_argument("--out", required=True, help="directory for the per-window gen_<rid>.npy")
    args = ap.parse_args()
    for k in ("ckpt", "proc", "starts_json", "out"):
        setattr(args, k, os.path.join(LAUNCH_DIR, os.path.expanduser(getattr(args, k))))

    os.makedirs(args.out, exist_ok=True)
    state, model, batchnorm, v, seq_len = load_model(args.ckpt, args.step)
    enc = v.ENCODING
    vocab_len = len(v)
    valid_mask_array = valh.syntax_validation_matrix(v)

    proc = np.load(args.proc, allow_pickle=True).astype(np.int64)  # (N,14)
    N = len(proc)
    starts_meta = json.load(open(args.starts_json))
    if args.limit:
        starts_meta = starts_meta[:args.limit]
    # entry["start"] is the first message of the HORIZON; the conditioning window precedes it
    windows = [(int(e["rid"]), int(e["start"]) - args.cond) for e in starts_meta]
    windows = [(rid, s0) for rid, s0 in windows if s0 - args.warmup >= 0 and s0 + args.cond + args.gen < N]
    print(f"[data] N={N} official windows: {len(windows)} (cond={args.cond} gen={args.gen} warmup={args.warmup})", flush=True)

    # sanity: tokenize a proc row and decode back
    tk = encode_msgs(jnp.asarray(proc[windows[0][1]:windows[0][1] + 2]), enc)
    dtest = np.asarray(encoding.decode_msg(tk[0], enc))
    _sp = windows[0][1]
    print(f"[sanity] proc[s0] price_rel={proc[_sp,4]} size={proc[_sp,5]} -> retok/decoded price_rel={int(dtest[4])} size={int(dtest[5])} etype={int(dtest[1])}", flush=True)

    meta = dict(day=args.day, cond=args.cond, gen=args.gen, ctx=args.ctx, warmup=args.warmup,
                windows=[[rid, s0] for rid, s0 in windows], proc=args.proc, top_n=args.top_n)
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f)

    key = jax.random.PRNGKey(args.seed + 1)
    for wi, (rid, s) in enumerate(windows):
        t0 = time.time()
        # seed mini-book with warmup..cond-end real messages
        book = MiniBook()
        seed_rows = proc[s - args.warmup: s + args.cond]   # order_id,etype,dir,price_abs,price_rel,size,...
        for r in seed_rows:
            etype, dir01, pabs, size = int(r[1]), int(r[2]), int(r[3]), int(r[5])
            book.apply(etype, size, pabs, 2 * dir01 - 1)
        # context tokens = last ctx cond msgs
        ctx_rows = proc[s + args.cond - args.ctx: s + args.cond]
        cond_tokens = np.asarray(encode_msgs(jnp.asarray(ctx_rows), enc)).reshape(-1)
        key, sub = jax.random.split(key)
        gen = sample_window(state, model, batchnorm, enc, valid_mask_array, vocab_len,
                            cond_tokens, book, sub, args.gen, args.top_n)
        np.save(os.path.join(args.out, f"gen_{rid}.npy"), np.asarray(gen, dtype=np.float64))
        et = [g[1] for g in gen]
        from collections import Counter
        print(f"[win {wi} rid={rid}] s={s} gen={len(gen)}/{args.gen} types={dict(Counter(et))} "
              f"px[min/max]={min((g[3] for g in gen), default=0)}/{max((g[3] for g in gen), default=0)} "
              f"{time.time()-t0:.1f}s", flush=True)
    print("[done] generation complete", flush=True)


if __name__ == "__main__":
    main()
