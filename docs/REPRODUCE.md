# Running the pipeline

How each stage of the pipeline behind the paper is run, in order: corpus, training, generation,
scoring, the analyses and the figures. `docs/PROTOCOL.md` says what the reported numbers mean and
which are comparable; `artifacts/README.md` describes the metadata files that ship (corpus
manifest, LOB-S5 window starts). No results or weights ship: every table and figure is regenerated
by the commands below, which write under `runs/` (and `$BOOKDIT_WORK` for generation trees).
Commands are plain `python`, run from the repository root; prefix with `uv run --no-sync` if you
used `uv`.
Settings that were not recorded for the published runs are marked as such; `<...>` is always a
value you supply.

## Paper item → commands

| Paper item | Section here | Output |
|---|---|---|
| Table 1, BookDiT without refinement | §3.4, §3.6 | `runs/lobbench/tab1_nomed/` (`nomed`) |
| Table 1, decode floor | §3.4, §3.5 | `runs/lobbench/tab1/` (`floor`) |
| Table 1, BookDiT + refinement | §3.4, §3.6 | `runs/lobbench/tab1_refine/` (`abs50`) |
| Table 1, LOB-S5 released and compute-matched, LOB-Bench baseline | §3.7 | `runs/lobbench/msg/`; window starts in `artifacts/tab1_baselines/lobs5_starts/` |
| Table 1, data-matched; six seeds; same-set check | §3.6 scoring; training of the extra seeds not included | none |
| Table 1, Painting the Market; Table 2 | §3.7 | `runs/lobbench/painting_vs_bookdit/` |
| Table 1, RWKV-4/6 and real-vs-real floor | not included | none |
| Table 3 | §3.8 | `runs/lobbench/book_native/`, `runs/lobbench/book_time/`, `runs/condpred/` |
| Table 4 | §3.7, chart ablation | `runs/lobbench/chart_ablation/` |
| §5 blockwise stability; Figure 2 (drift panel) | §3.9, §4 | `runs/horizon/bands/`, `runs/figures/horizon_bands.pdf`; LOB-S5 window starts in `artifacts/fig_horizon/` |
| §5 divergence; Figure 3 | §3.9, §4 | `runs/divergence/`, `runs/figures/divergence_drift.pdf` |
| Table 5 | not included | none |

## 1. Setup and data

### 1.1 Data layout

The raw archives must be obtained from LOBSTER
(<https://lobsterdata.com>) under your own subscription and its terms. The corpus build needs the
2024 archives of the 21 names in `artifacts/corpus/corpus_manifest_2024.csv`; the evaluation needs
GOOG message files for the nine January-2023 days of §1.3. The only market-format files in the
repository are the two synthetic days in `quickstart/toy_data/`.

`build_corpus.py` globs `$LOBSTER_RAW_ROOT/*/[A-Z]*.7z`, so the per-day directory level is
required, and every reader expects one message CSV per ticker-day at the path below (the corpus
build extracts it there itself; for the 2023 evaluation days, place each day's GOOG message file
there under that name):

```
$LOBSTER_RAW_ROOT/<YYYYMMDD>/<TICKER>.7z                      # raw archives
$LOBSTER_CACHE/csv/<YYYYMMDD>/<TICKER>_message.csv            # message CSVs
$LOBSTER_CONTIG_ROOT/<TICKER>_<YYYYMMDD>_event10_contig.pt    # corpus output
```

**What the message files must be.** The corpus build and every real-side reader replay the message
file alone, from an empty book (§3.1). That is exact only for **full-depth** message files (all
price levels) that cover the **whole session from the 04:00 pre-market open**, one per ticker-day,
as exported by an institutional LOBSTER server. The files must have the LOBSTER message columns:
time (seconds after midnight), event type, order id, size, price (in 1/10000 dollar) and
direction. The files behind the paper have a seventh trailing column (`null`, or a
market-participant id on some rows), and their first row can be a type-7 halt record from before
04:00. Both are harmless. `load_msgs` reads only columns 1, 2, 4, 5 and 6 (time, type, size, price,
direction), so the order id and any extra columns are ignored. `Book.apply` leaves the visible book
unchanged for types 5 (hidden execution), 6 (cross) and 7 (halt).

A standard LOBSTER download does not meet this. It covers 09:30–16:00 at level *N* and carries only
the events inside the requested levels. Orders resting deeper than level *N*, or entered before the
open, then have no submission message, so a replay from an empty book never contains them and
drifts from the true book. Checked on two standard level-10 days (WMT 2019-01-03 and AIG
2019-01-02), `--validate` matched LOBSTER's own orderbook file on 0.45% and 0.10% of rows over the
whole file, and on 0.18% and 0.05% in market hours. Before a full build, check one day of your data
with `reconstruct_event_book.py --validate` (§3.1). It needs that day's LOBSTER orderbook file and
compares the replay with it message by message.

### 1.2 Install and the pinned clones

`README.md` has the `uv sync` install; the `dev` extra is pytest and nothing else. Python 3.11 or
newer.

```bash
bash scripts/data/setup_libs.sh                # libs/lob_bench at its pinned SHA (scoring only)
bash scripts/data/setup_libs.sh --with-lobs5   # also clone LOB-S5 and apply the patch
```

| Clone | URL | SHA |
|---|---|---|
| `lob_bench` | `https://github.com/peernagy/lob_bench` | `276baf3039bf7630b555891b9d04ad49a92b3356` |
| `LOBS5` | `https://github.com/peernagy/LOBS5` | `aa36029e9d348a017d00efc37c6b6dda89fab150` + `patches/lobs5-jax-compat.patch` |

`patches/lobs5-jax-compat.patch` holds JAX/Flax/NumPy-2 compatibility fixes for LOB-S5 at that
commit, plus a message-only collate fix; it changes no model, loss, tokenizer or training default.
`--latest` clones unpinned heads, on which the patch may not apply.

`libs/` is gitignored and training imports nothing from it, so this can wait until you score; a
`ModuleNotFoundError` naming `data_loading`, `scoring` or `run_bench` means `setup_libs.sh` has
not run.

**JAX environment.** `scripts/baselines/cst_{estimate,generate}.py`,
`gen_lobs5_on_lobbench_windows.py` and LOB-S5's own training run in a separate environment: a CUDA
jaxlib and a CUDA torch in one venv contend for the GPU allocator. The `.[baselines]` extra (jax,
jaxlib, flax, chex) is not enough for LOB-S5, which also imports optax, orbax-checkpoint, torch,
einops, pandas and wandb. The environment of the published LOB-S5 retrain was Python 3.13.5,
jax/jaxlib 0.11.0 with jax-cuda12-plugin/pjrt 0.11.0, flax 0.12.7, optax 0.2.8, orbax-checkpoint
0.12.1, chex 0.1.92, numpy 2.5.1, pandas 3.0.3, torch 2.13.0+cpu, einops 0.8.2, wandb 0.28.1, with
every `site-packages/nvidia/*/lib` directory on `LD_LIBRARY_PATH` and
`XLA_PYTHON_CLIENT_PREALLOCATE=false`.

### 1.3 `.env`

```bash
cp .env.example .env      # then fill in absolute paths
```

| variable | accessor | needed by |
|---|---|---|
| `LOBSTER_RAW_ROOT` | `lobster_raw_root()` | corpus build only (the `.7z` archives) |
| `LOBSTER_CACHE` | `lobster_cache()` | everything that reads real books: corpus build, window build, generation, scoring |
| `LOBSTER_CONTIG_ROOT` | `contig_root()` | training and every rollout driver (`*_event10_contig.pt`); defaults to `<LOBSTER_CACHE>/tensors_event_contig` |
| `LOBSTER_SCRATCH` | `scratch_dir()` | corpus-build temporaries; optional, falls back to the system temp dir |
| `BOOKDIT_WORK` | `bench_work()` | every LOB-Bench lane: the generation and scoring trees live here (tens of GB) |
| `LOB_BENCH_DIR` | `lob_bench_dir()` | scoring; defaults to `<repo>/libs/lob_bench`, where `setup_libs.sh` writes it |
| `LOBS5_DIR` | `lobs5_dir()` | the LOB-S5 baseline arm only |

Paths must be absolute: the parser (`src/bookdit/utils/env.py`) does not expand `~`, `$HOME` or
`${VAR}`, and an unset root raises `SystemExit` naming the variable. Shorthands used throughout:

```bash
DS=$BOOKDIT_WORK/lobbench_repro/lob_bench/sample_data/data_saved
DAYS="20230103 20230104 20230105 20230106 20230109 20230110 20230111 20230112 20230113"
COND=$DS/hf_lobs5_9d_f10/GOOG/data_cond     # built in §3.2
```

## 2. What runs from a checkout alone

The unit suite, a pico-size training run on the synthetic books and inspection of the resulting
checkpoint's metadata run on CPU in seconds;
[`../quickstart/README.md`](../quickstart/README.md) walks them one at a time.

## 3. The pipeline

The steps are in order; each names what it writes, and the later ones assume the earlier ones ran.

### 3.1 Build the corpus

| Property | Value |
|---|---|
| ticker-days | 5,142 |
| names | 21 (listed in the manifest) |
| year | 2024 (2024-01-02 … 2024-12-31) |
| stride | 10: one book frame per ten raw messages |
| levels | 10 (40 columns: `[ask_px, ask_sz, bid_px, bid_sz] × 10`) |
| split | 5,087 train / 44 val / 11 test |
| manifest | `artifacts/corpus/corpus_manifest_2024.csv` |

The split is in code (`split_for` in `scripts/data/build_corpus.py`): **test** is GOOG 2024-11-12 …
2024-11-26; **val** is GOOG 2024-11-08 and 2024-11-11 plus 2024-12-30 and 2024-12-31 for every
name; **train** is the rest. It governs training and model selection only; evaluation is on
LOB-Bench's GOOG January-2023 windows (§3.2). One torch dict per ticker-day, from
`reconstruct_event_book.emit_contiguous`:

```python
{
  "books":      torch.int32,   # (T, 40)  prices in 1/10000 dollar, sizes in shares
  "times":      torch.float64, # (T,)     seconds after midnight, aligned to books
  "trades":     torch.int32,   # (T, 4)   per-frame (buy_vol, sell_vol, n_trades, hidden_vol)
  "has_trades": True,
}
```

Frames sit on the `k % stride == 0` grid over the raw message stream, restricted to market hours
`34200 ≤ t ≤ 57600`; the market-hours test runs first, so frame 0 is the first stride-multiple
inside the session, not raw index 0. `trades[0]` is all-zero, and days with fewer than 1,000
snapshots are skipped. The trainer derives the 41-channel `midspread_dt` representation on the fly
(`scripts/lib/lob_representation.py:book_to_midspread_dt`).

**Input.** One full-depth, full-session (from 04:00) message file per ticker-day, with the LOBSTER
message columns (§1.1). The extra trailing column is ignored, and pre-open rows are replayed so that
the book is complete at 09:30. Only frames inside market hours are kept.

**Step 1: check the replay on one day** (`scripts/lib/reconstruct_event_book.py`):

```bash
python scripts/lib/reconstruct_event_book.py --day 20240304 --ticker GOOG --validate
python scripts/lib/reconstruct_event_book.py --day 20240304 --ticker GOOG \
    --emit-contiguous --out runs/corpus_check        # one day by hand; --stride defaults to 10
```

`--validate` replays the message file and compares it, message by message, with LOBSTER's
orderbook file at `$LOBSTER_CACHE/csv/<day>/<TICKER>_orderbook.csv` (or `--orderbook PATH`; the two
files must come from the same LOBSTER request and have the same row count). It prints the
exact-match rate over every message and over market hours, and the first mismatch. Match rates
near 1 mean the file is usable. Rates under 1% (what a standard level-10 download gives; §1.1)
mean the replay misses resting orders, and the corpus built from that file would be wrong.

**Step 2: build every day, in parallel, with the manifest.** Resumable; deletes only the CSVs it
extracted itself.

```bash
python scripts/data/build_corpus.py --stride 10 --workers 16 \
    --manifest-out runs/crossasset/corpus_manifest_2024.csv
```

`--limit N` builds only the first N archives. Point `LOBSTER_SCRATCH` at a fast local disk; 7z
extraction goes through it. The manifest has six columns, `ticker,day,year,split,tensor_path,status`.
A fresh build writes absolute `tensor_path`s; the shipped manifest holds `${LOBSTER_CACHE}/...`
placeholders instead and documents the day list and split, so train from the one you build.

**Step 3: per-channel normalisation statistics.** A sharded streaming pass with exact float64 sums
over the training split:

```bash
for s in $(seq 0 11); do
  python scripts/lib/build_rep_norm_stats.py \
      --manifest runs/crossasset/corpus_manifest_2024.csv --split train --rep midspread_dt \
      --shard "$s" --nshards 12 --out runs/crossasset/stats_p$s.npz &
done; wait
python scripts/lib/build_rep_norm_stats.py --combine 'runs/crossasset/stats_p*.npz' \
    --out runs/crossasset/norm_midspread_dt.npz
```

Per-shard files are partial sums, which the trainer rejects until they are combined. The combined
file carries `mean`, `std`, `n`, `n_days`, `n_parts`, `rep`, and its path is the trainer's
`--norm-stats` default. The paper's checkpoint instead used `mean`/`std` estimated on a random
200-day subsample of the training split (§3.3).

### 3.2 Rebuild the benchmark's real and conditioning sides

The HuggingFace release (`peernagy/lob_bench`) ships only `data_gen` for the GOOG January-2023
windows; `data_real` and `data_cond` are rebuilt from your LOBSTER messages:

```bash
python scripts/baselines/build_lobbench_hf_real.py --days $DAYS \
    --gen_dir <LOB-Bench HF release: LOB-S5 data_gen> \
    --out $DS/hf_lobs5_9d_f10
# -> $DS/hf_lobs5_9d_f10/GOOG/{data_real,data_cond}
ln -sfn <LOB-Bench HF release: LOB-S5 data_gen> $DS/hf_lobs5_9d_f10/GOOG/data_gen
```

Alignment is by timestamp (`build_day` in `build_lobbench_hf_real.py`): released filenames say
nothing about where in the day a window starts, so each generated file's first timestamp is located
by `np.searchsorted` at position `p` in the day's filtered stream `F` (visible-book event types
`(1, 2, 3, 4)` inside market hours). Windows with `p - 500 < 0` or `p + 500 > len(F)` are rejected;
`data_cond = F[p-500 : p]` and `data_real = F[p : p+500]` are written as messages and books.
`--gen_dir` fixes the window set, so every arm is aligned against one shared tree, which
`score_official.py` resolves as `$DS/hf_lobs5_9d_f10/GOOG` (its `ref_dir()`); keep that directory
name. The `data_gen` link makes the same directory the released-LOB-S5 arm for §3.7 and §3.9.

`data_cond` is a 500-message prefix, the prompt the message models received. BookDiT re-derives its
own 250-frame (2,500-message) context from the same filtered stream (`day_windows` in
`gen_bookdit_on_lobbench_windows.py`; [`PROTOCOL.md` §3](PROTOCOL.md#3-the-conditioning-arithmetic)).
A misaligned rebuild will not produce a near-zero real-vs-real floor (Table 1).

### 3.3 Train BookDiT

The trainer converts days to the representation inside the DataLoader workers, so the corpus never
has to fit in RAM. The defaults are the recipe of the checkpoint reported in the paper:

```bash
python scripts/lib/train_bookdit.py \
    --manifest runs/crossasset/corpus_manifest_2024.csv \
    --norm-stats runs/crossasset/norm_midspread_dt.npz \
    --rep midspread_dt --size mini --seq-len 500 \
    --steps 109216 --batch-size 32 --lr 5e-4 --seed 1 \
    --out runs/crossasset/bookdit_mini.pt
```

The checkpoint reported in the paper is seed 1 of three seeds (0, 1, 2) of this recipe, picked by
LOB-Bench L1 on two held-out GOOG validation days (2024-11-08, 2024-11-11). Six seeds scored on a
common window set give L1¹⁸ 0.222 ± 0.020 against 0.193 for the paper's one (Table 1, six seeds;
README, "Seeds and checkpoint selection"). Expect a re-run at another seed to land in that range,
not at the paper checkpoint's scores.

The checkpoint carries the representation, architecture, normalisation statistics and context
length, so downstream scorers need no config. Below, `$CKPT` is the checkpoint you trained (here
`runs/crossasset/bookdit_mini.pt`). No weights ship with this repository; the paper's checkpoint is
not distributed. Every BookDiT evaluation number in the paper comes from
that checkpoint, which was produced by an earlier run of this recipe with a different streaming
loader and subsampled normalisation statistics; a re-run matches the recipe, not the weights.
Training is a multi-day run on a single GPU.

### 3.4 Generate on the official windows

Four runs of one script, differing in a flag and in `--out`:

```bash
COMMON="--ckpt $CKPT --cond-dir $COND --days $DAYS \
        --stride 10 --levels 10 --horizon-msgs 500 --gen-steps 100 --seed 0 --batch 64"

# refined frames, dumped before decoding (Table 1 "+ refinement", decoded in §3.6)
python scripts/baselines/gen_bookdit_on_lobbench_windows.py $COMMON --refine \
    --dump-frames $BOOKDIT_WORK/refine_frames --out $BOOKDIT_WORK/refine_unused
# unrefined frames, dumped before decoding (Table 1 "without refinement", decoded in §3.6)
python scripts/baselines/gen_bookdit_on_lobbench_windows.py $COMMON \
    --dump-frames $BOOKDIT_WORK/norefine_frames --out $BOOKDIT_WORK/norefine_unused
# book-state decode floor: the real frames through the same decoder
python scripts/baselines/gen_bookdit_on_lobbench_windows.py $COMMON --mode floor \
    --out $BOOKDIT_WORK/tab1/floor
# unrefined, decoded directly with the decoder's built-in column-space band: the submitted
# version's "without refinement" row and the BookDiT
# stand-in of the divergence lane (§3.9); not a camera-ready Table 1 row
python scripts/baselines/gen_bookdit_on_lobbench_windows.py $COMMON --out $BOOKDIT_WORK/tab1/raw
```

The two dumps differ only in `--refine`, which is applied after sampling. With the same `--seed`
and `--batch` both runs sample the same generated frames (up to GPU nondeterminism), so the
"+ refinement" and "without refinement" rows differ only by the median filter.

The driver inpaints 250 frames per window, keeps the first 50, and decodes them with the standard
decoder ([`PROTOCOL.md` §3–4](PROTOCOL.md#3-the-conditioning-arithmetic)), writing LOBSTER-format
message and orderbook CSVs under `--out`. `--mode floor` decodes the real continuation instead and
still reads `--ckpt` for the representation and level count. `--dump-frames <dir>` writes the
pre-decode frames (one `.npz` per window) and skips decoding, so its `--out` stays empty. `--batch`
is part of the random draw (each batch is seeded from its start index), so hold it fixed with
`--seed` to repeat a generation.

### 3.5 Score

```bash
python scripts/baselines/score_official.py \
    --gen-root $BOOKDIT_WORK/tab1 --arms raw floor --no-intersect \
    --out runs/lobbench/tab1
```

The scorer applies the level-ten filter to every arm, runs LOB-Bench, and writes `aggregates.csv`,
`per_stat_l1.csv` and `keys.json` into `--out`, leaving the
filtered copies in `<out>/_work/_f10_<arm>/` for §3.8. Row keys are the arm directory names.
`--no-intersect` scores each arm on its own
surviving windows; without it every arm is restricted to the intersection, which is what a
head-to-head wants.

### 3.6 The two BookDiT rows: deadband, decode, score

Both Table 1 BookDiT rows decode their dumped frames (§3.4) the same way. The θ = 50-share deadband
is applied before decoding, in the decoder's quantized (side, price) space, and the decoder's
built-in band is switched off so the band is applied once. `apply_deadband.py` forces
`IDM_DEADBAND_ABS=0` itself; setting it explicitly keeps the intent visible. The gap profile stays
on (`IDM_GAP_PROFILE=1`, the default).

```bash
# + refinement
IDM_DEADBAND_ABS=0 python scripts/baselines/apply_deadband.py \
    --frames $BOOKDIT_WORK/refine_frames --days all --settings raw abs50 \
    --out-root $BOOKDIT_WORK/tab1_refine
python scripts/baselines/score_official.py \
    --gen-root $BOOKDIT_WORK/tab1_refine --arms raw abs50 --no-intersect \
    --out runs/lobbench/tab1_refine
# without refinement: same deadband and decoder on the unrefined frames
IDM_DEADBAND_ABS=0 python scripts/baselines/apply_deadband.py \
    --frames $BOOKDIT_WORK/norefine_frames --days all --settings abs50 \
    --out-root $BOOKDIT_WORK/tab1_norefine
mkdir -p $BOOKDIT_WORK/tab1_nomed
ln -sfn $BOOKDIT_WORK/tab1_norefine/abs50 $BOOKDIT_WORK/tab1_nomed/nomed
python scripts/baselines/score_official.py \
    --gen-root $BOOKDIT_WORK/tab1_nomed --arms nomed --no-intersect \
    --out runs/lobbench/tab1_nomed
```

`abs50` of the first run is the "+ refinement" row and `nomed` is the "without refinement" row,
both on 3,127 windows in the paper's run. The arm names are `apply_deadband.py` settings, not
refinement settings: `raw` in the first run means *no deadband* (refined frames, decoder band off),
not *no refinement*. `$BOOKDIT_WORK/tab1_refine/abs50` is also the BookDiT arm the Painting
comparison (§3.7) scores against.

The data-matched row, the six-seed rows and the same-set check
put the paper's checkpoint, five more seeds of its configuration, three data-matched checkpoints
and the LOB-S5 retrain through the refined path above (`--refine`, then `apply_deadband.py
--settings abs50`). They are scored together in the default intersect mode, which restricts every
arm to the 3,111 windows all of them keep. The extra seeds, the data-matched checkpoints and their
training configuration are not included in this release.

### 3.7 The other arms

**LOB-S5 and the LOB-Bench baseline** are message models and never touch the IDM decoder. The
released LOB-S5 samples and the parametric baseline come from the LOB-Bench HuggingFace release;
the compute-matched retrain is trained and sampled in the patched LOB-S5 clone, in the JAX
environment of §1.2.

*Proc arrays (manual step, not reproducible from this repository alone).* LOB-S5 reads one
preprocessed array per day, `GOOG_<YYYYMMDD>_message_proc.npy`, written by LOB-S5's own
preprocessing (`python lob/preproc.py --data_dir <raw csv dir> --save_dir <proc dir>`). Its input is
a LOBSTER-named message/orderbook CSV pair per day. The export that turned the full-depth message
files (§1.1) into those pairs is not included, and its exact preprocessing flags were not recorded.
The run read message and orderbook files, so it was not `--messages_only`. Whatever you use, it
must produce the following, or the shipped window starts will not line up:

- **Message CSV.** The rows of the day's message file with `34200 ≤ t ≤ 57600` (09:30–16:00
  inclusive), in the six standard LOBSTER columns. For 2023-01-04 the archived export wrote
  4,120,394 rows, which equals that count in our file.
- **Orderbook CSV.** Row-aligned with it: the visible book after each message, for example the
  §3.1 replay of the full-session file at those rows. LOB-S5's preprocessing reads both files.
  How the published export built this file was not recorded.
- **Proc array.** After preprocessing, one row per message of types 1–4 in that range, minus one
  row. Columns 1, 2, 3, 5, 8, 9 are event type, direction (0/1), absolute price, size, seconds and
  nanoseconds (`proc_events` in `lobs5_gen_to_lobster.py`). The archived runs logged these row
  counts (`N=`), each one less than that day's count of type-1–4 market-hours messages:

| day | proc rows | day | proc rows | day | proc rows |
|---|---:|---|---:|---|---:|
| 20230103 | 2,643,224 | 20230106 | 2,991,538 | 20230111 | 2,426,887 |
| 20230104 | 4,107,509 | 20230109 | 2,496,411 | 20230112 | 3,934,750 |
| 20230105 | 2,858,829 | 20230110 | 3,740,675 | 20230113 | 2,638,965 |

A `start` in the shipped start files indexes rows of this array, so check your row counts against
the table before sampling. The retrain's training days, 20240123, 20240124, 20240125, 20240126,
20240129 and 20240131 (23.2M messages), went through the same preprocessing. Rebuilding these
arrays has not been verified.

*Training.* In the JAX environment, with the working directory `$LOBS5_DIR`:

```bash
WANDB_MODE=offline python run_train.py --USE_WANDB False --dir_name <proc dir> \
    --use_book_data False --masking causal --msg_seq_len 100 --n_layers 4 --n_message_layers 2 \
    --d_model 64 --ssm_size_base 64 --blocks 8 --bsz 32 --epochs 6 --num_devices 1 \
    --n_data_workers 0 --jax_seed 1
```

Every other setting is a `run_train.py` default (`ssm_lr_base` 1e-3, `lr_factor` 1, `weight_decay`
0.05, cosine annealing, `warmup_end` 1, `p_dropout` 0, batchnorm, prenorm, `mode pool`, `C_init
trunc_standard_normal`, `discretization zoh`, `activation half_glu1`). The published run had
1,600,043 trainable parameters; validation loss by epoch was 2.16604, 1.83804, 1.76485, 1.74984,
1.72737, 1.74075, with validation/test accuracy 0.7026 / 0.7223 after epoch 6. Checkpoints are
written to `$LOBS5_DIR/checkpoints/<run name>_<run id>/checkpoint_<0-based epoch>`; sampling uses
`--step 5`, the end of epoch 6 (the sampler's default).

*Sampling on the official windows (Table 1).* `--starts-json` lists `{"rid", "start"}` pairs:
`rid` is the LOB-Bench `real_id`, and `start` is the row of that day's proc array at which the
scored continuation begins. The per-day start files of the published run ship in
`artifacts/tab1_baselines/lobs5_starts/` (3,130 windows over the nine days). No script here builds
them: they align LOB-Bench windows to proc rows by timestamp. `--cond 500 --gen 500 --ctx 100
--warmup 8000 --top_n -1 --seed 0` are the defaults and were used. One call per day:

```bash
# JAX env, LOBS5_DIR exported, any working directory
for d in $DAYS; do
  python scripts/baselines/gen_lobs5_on_lobbench_windows.py \
      --ckpt $LOBS5_DIR/checkpoints/<run name>_<run id> --step 5 \
      --proc <proc dir>/GOOG_${d}_message_proc.npy --day $d \
      --starts-json artifacts/tab1_baselines/lobs5_starts/lobs5_starts_${d}.json \
      --out $BOOKDIT_WORK/lobs5_gen/$d
done
# main venv: rebuild order ids and books, all days into one arm directory
for d in $DAYS; do
  python scripts/baselines/lobs5_gen_to_lobster.py --gen-dir $BOOKDIT_WORK/lobs5_gen/$d \
      --proc <proc dir>/GOOG_${d}_message_proc.npy --date ${d:0:4}-${d:4:2}-${d:6:2} \
      --out $BOOKDIT_WORK/msg/lobs5_retrain
done
ln -sfn $DS/hf_lobs5_9d_f10/GOOG/data_gen $BOOKDIT_WORK/msg/lobs5_released
ln -sfn <LOB-Bench HF release: baseline data_gen> $BOOKDIT_WORK/msg/baseline
python scripts/baselines/score_official.py --gen-root $BOOKDIT_WORK/msg \
    --arms lobs5_retrain lobs5_released baseline --out runs/lobbench/msg
```

The published run did not make one call per day. For eight of the days it started one process
per day on these files, eight in parallel. Those processes did not finish their days, so it
finished the remaining windows in further processes on subsets of the same starts: 28 processes
in all. Each process seeds its sampler once from `--seed` (`PRNGKey(seed + 1)`, then split per
window), so the one-call-per-day loop above samples the same windows with the same starts but
does not reproduce the published draws exactly. The subset files are not shipped.

`lobs5_gen_to_lobster.py` replays the sampled messages through the crossing-aware `MatchingBook`
(defined in `rescore_matching.py`), warmed on the real messages before each window: a naive replay
would rest a crossing limit order instead of executing it, producing crossed books. The three
rows in the paper share one 3,124-window set; the default intersect mode restricts the arms to
their common windows.

**Painting the Market.** `scripts/baselines/run_painting.sh` is the whole recipe for our
reproduction of arXiv:2509.05107: it trains with `train_painting_market.py` (one pass over the GOOG
training split, 11,622 steps at effective batch 32), samples one 100-state block per window over the
nine-day source tree with `eval_painting_official.py --windows 0 --blocks 1 --sample-steps 50 --seed
20260801` (DDIM, η = 0), and scores the result against BookDiT's refined arm in intersect mode:

```bash
bash scripts/baselines/run_painting.sh                  # BookDiT arm: $BOOKDIT_WORK/tab1_refine/abs50
# -> runs/lobbench/painting_vs_bookdit/{aggregates.csv,per_stat_l1.csv}
```

847 of 3,136 windows survive the level-ten filter and 844 are shared with BookDiT. Table 2's spread
and OFI columns are the `spread` row and the mean of the four `ofi*` rows of `per_stat_l1.csv`. The
evaluator draws its sampling noise from a seeded generator; whether the published draw used the same
seeding is not recorded, so it may not repeat bit for bit.

**Cont–Stoikov–Talreja.** `scripts/baselines/cst_prep.py` (main venv; reads LOB-S5 proc arrays) →
`cst_estimate.py` → `cst_generate.py` (both JAX env), chained through
`runs/lobbench/baseline_cst/{prep,aggregated_params.pkl,gen}` by default. CST appears only in the
long-horizon lane. The archived driver of the published 16,000-message run did the following. It
estimated parameters on 2023-01-03, the only corpus day before the generation day, and generated
on 2023-01-04:

```bash
python scripts/baselines/cst_prep.py --proc-dir <proc dir> --train-days 20230103 --gen-day 20230104 \
    --cond 500 --gen 16000 --warmup 8000 --K 40 --out <cst>/prep_params
python scripts/baselines/cst_estimate.py --param-input <cst>/prep_params/param_input \
    --out <cst>/params_2023.pkl                                        # JAX env
python scripts/baselines/cst_prep.py --starts-from <starts-from json> --out <cst>/prep_official
python scripts/baselines/cst_generate.py --params <cst>/params_2023.pkl --prep <cst>/prep_official \
    --gen 16000 --seed 0 --out <cst>/gen                                # JAX env
```

The `--starts-from` file of that run (164 window starts on 2023-01-04, with `cond 500`, `gen
16000`, `warmup 8000`) is not shipped. Its 164 windows contain the 107 of the LOB-S5 shards in
`artifacts/fig_horizon/`.

**The coordinate-chart ablation (Table 4).** Six representations, the `CHART_ABLATIONS` names in
`lob_representation.py`, each trained with its own normalisation pass (§3.1 step 3 with that `--rep`):

```bash
python scripts/lib/train_bookdit.py \
    --manifest runs/crossasset/corpus_manifest_2024.csv \
    --norm-stats <stats for that rep> --rep chart_full --size mini \
    --steps <steps> --seed <seed> --out runs/crossasset/chart_ablation/chart_full.pt
# generate each arm as in §3.4 (--out $BOOKDIT_WORK/chart/<arm>, and once more with
# --dump-frames $BOOKDIT_WORK/chart_frames/<arm>), then score all six together
python scripts/baselines/score_official.py --gen-root $BOOKDIT_WORK/chart \
    --arms chart_full chart_absmid chart_linspread chart_linlevels chart_rawtouch chart_raw \
    --out runs/lobbench/chart_ablation
python scripts/analysis/chart_validity.py --root $BOOKDIT_WORK/chart_frames \
    --arms chart_full chart_absmid chart_linspread chart_linlevels chart_rawtouch chart_raw \
    --key gen_book --out runs/lobbench/chart_ablation/validity.csv
```

The step count, seed and refinement setting of the six published runs were not recorded.
`chart_full` is the same chart as `midspread_dt` but a separately trained model, not the paper's
checkpoint. The paper's LOB-Bench columns are on the 2,570-window
intersection; the validity rates in `validity.csv` are read off the dumped frames before any decode
(`--key gen_book`), over all 3,132 generated windows: `all_*` over every emitted frame, `pop_*` over
frames with all ten levels populated on both sides.

### 3.8 Analyses that do not use the benchmark (Table 3)

**Book-native and dynamics rows.** These work on the book itself, with no decoder on either side.
`NAME=DIR` arguments take LOBSTER-format directories: the level-ten-filtered copies the scorer
leaves in `<score out>/_work/_f10_<arm>/` (runnable alone with `score_official.py --filter-only`),
and `$DS/hf_lobs5_9d_f10/GOOG/data_real` for `real`.

```bash
python scripts/analysis/book_native_facts.py \
    --arms real=<dir> bookdit=<dir> lobs5_released=<dir> painting=<dir> \
    --ref real --out runs/lobbench/book_native
python scripts/analysis/book_time_facts.py \
    --arms real=<dir> bookdit=<dir> lobs5_released=<dir> \
    --ref real --grid both --points 64 --out runs/lobbench/book_time
```

The published run restricts every arm to the single day 2023-01-03 (270 windows; 269 survive the
normalised-time grid of `book_time_facts.py`), so point the arm directories at that day's files.
Whether its BookDiT arm was the refined or the unrefined generation is not recorded. Both outputs
report every arm twice: take the `sample=matched` rows of `book_native_facts.py` and the `grid=norm`
rows of `book_time_facts.py` (VR(5) is `vr_q5`); `book_time_facts.py` also writes
`event_clock.csv` and `protocol.json`, which the paper does not report. `book_native_facts.py` also takes a model's own
pre-decode frames through `--npz-arms NAME=DIR` pointed at a `--dump-frames` directory. Table 3's
depth-profile drift and deep-volume drift rows are not produced by any script in this release.

**Conditional prediction (transfer rows).** Design A holds the feature matrix identical across cells
and changes only the label. `--frames` is a §3.4 `--dump-frames` directory over the nine days;
whether the published run used the refined or the unrefined dump is not recorded. The LOB-S5 source
is a LOBSTER-format generation tree, `--l5-dir` (or `$BOOKDIT_L5_DIR`):

```bash
python scripts/analysis/conditional_prediction.py --frames <dump dir> \
    --l5-dir $BOOKDIT_WORK/msg/lobs5_released --sources real,bookdit,lobs5 \
    --horizons 10,20 --ctxW 100 --selfW 20 --folds 3 --tag released_h1020 --out runs/condpred
python scripts/analysis/conditional_prediction.py --frames <dump dir> \
    --l5-dir $BOOKDIT_WORK/msg/lobs5_retrain --sources real,bookdit,lobs5 \
    --horizons 10,20 --ctxW 100 --selfW 20 --folds 3 --tag retrain_h1020 --out runs/condpred
python scripts/analysis/condpred_report.py runs/condpred/condpred_released_h1020.json
```

`condpred_report.py` prints the "R² carried to real" row: design A, train on X and test on real,
averaged over the targets `spread`, `dimb` (depth imbalance) *and* `rvol` (realized volatility) and
over horizons 10 and 20 (`--aggregate-only runs/condpred/condpred_*_h1020.json` prints it for both
runs). In the paper it is +0.440 / +0.291 / +0.242 for real / BookDiT / released LOB-S5 and −0.397
for the retrain. The retrain run keeps only the 374
windows on which every source has the frames natively (`n_windows`), against 3,126 for the released
run.

### 3.9 The long-horizon and divergence lanes

**Long-horizon drift (§5, Figure 2 drift panel).** Standard decoder, `--refine`, day 2023-01-04, 32
blocks of 500 messages out to 16,000, bootstrapped over the windows present in every block of every
arm. BookDiT (runs for hours on one GPU):

```bash
python scripts/baselines/horizon_official_protocol.py \
    --ckpt $CKPT --cond-dir $COND --day 20230104 \
    --stride 10 --horizon 16000 --block 500 --K 200 --gen-steps 100 --refine \
    --out runs/horizon/bookdit
python scripts/baselines/rescore_horizon_fixedset.py --run runs/horizon/bookdit --block 500
python scripts/analysis/horizon_score_cache.py --arm bookdit --run runs/horizon/bookdit
```

LOB-S5 retrain, 16,000 messages, four shards on one GPU (about 1,020–1,120 s per window and 27
windows per shard in the published run, so about 7.7 h). The four start files of the published run
ship as `artifacts/fig_horizon/lobs5_starts_20230104_shard{0,1,2,3}.json` (107 windows; format in
§3.7). The generation loop below is the archived driver's, with paths replaced:

```bash
# JAX env
for k in 0 1 2 3; do
  python scripts/baselines/gen_lobs5_on_lobbench_windows.py \
      --ckpt $LOBS5_DIR/checkpoints/<run name>_<run id> --step 5 \
      --proc <proc dir>/GOOG_20230104_message_proc.npy \
      --starts-json artifacts/fig_horizon/lobs5_starts_20230104_shard$k.json \
      --day 20230104 --cond 500 --gen 16000 --warmup 8000 --seed $((100+k)) \
      --out $BOOKDIT_WORK/lobs5_h16k/shard$k &
done; wait
# main venv: adapt the shards, restricted to the windows BookDiT keeps at every block
python scripts/baselines/build_horizon_scoring_dirs.py \
    --lobs5-shards $BOOKDIT_WORK/lobs5_h16k/shard{0,1,2,3} \
    --restrict-to-run runs/horizon/bookdit --out-lobs5 $BOOKDIT_WORK/lobs5_h16k/scoring
python scripts/analysis/horizon_score_cache.py --arm lobs5 --gen-dir $BOOKDIT_WORK/lobs5_h16k/scoring
python scripts/baselines/score_lobs5_horizon.py --gen-dir $BOOKDIT_WORK/lobs5_h16k/scoring
```

The commands after generation (the adapter, cache and scoring calls above) are the scripts'
documented usage. The published run's own invocations of them were not recorded, so their flags are
a reconstruction. CST goes through the same adapter (`--cst-dir <cst gen> --cst-prep <cst prep>
--starts-json <starts> --out-cst <cst scoring dir>`), then `horizon_score_cache.py --arm cst
--gen-dir <cst scoring dir>` and `score_lobs5_horizon.py --gen-dir <cst scoring dir> --out
runs/horizon/cst`. Its generation commands are the archived ones in §3.7; the CST and LOB-S5
16,000-message commands beyond those archived drivers were not recorded. Then the fixed-window-set
bands and the real-vs-real control:

```bash
python scripts/analysis/horizon_matched_bands.py --population matched -B 1000 -R 24 \
    --floor-pop carrier --seed 20260804 --out runs/horizon/bands
python scripts/analysis/horizon_matched_bands.py --validate      # replay vs LOB-Bench profiles
```

`horizon_matched_bands.py` reads `<--cache-root>/<arm>` (default `runs/horizon/cache`), which is
where `horizon_score_cache.py` writes by default. `--validate` checks the local L1 replay against
each arm's LOB-Bench profile (`runs/horizon/bookdit/profile_fixedset.csv`,
`runs/horizon/{lobs5,cst}/profile.csv`; override with `--profile ARM=PATH`). The bands step writes
`bootstrap.json` and `curve_bands_l1_18.csv` to `--out`. The original LOB-S5 and CST generations are not archived, so these arms
have to be regenerated.

**Divergence (§5, Figure 3).** LOB-Bench's own `--div_only` mode, BookDiT against the released
LOB-S5 samples. `setup_divergence_official.py` reads each arm from a LOB-Bench-layout tree
`$DS/<tree>/GOOG/{data_real,data_cond,data_gen}`: `hf_lobs5_9d_f10` (§3.2) for `lobs5` and for the
`realgen` control's real side, `hf_bookdit_9d` for `bookdit` and `hf_bookditfloor_9d` for
`ditfloor` (or `--src ARM=DIR`). Build the two BookDiT trees from §3.4 output:

```bash
for t in hf_bookdit_9d:raw hf_bookditfloor_9d:floor; do
  tree=${t%%:*}; arm=${t##*:}; mkdir -p $DS/$tree/GOOG
  for s in data_real data_cond; do ln -sfn $DS/hf_lobs5_9d_f10/GOOG/$s $DS/$tree/GOOG/$s; done
  ln -sfn $BOOKDIT_WORK/tab1/$arm $DS/$tree/GOOG/data_gen
done
# 1. symlink farms for bookdit, lobs5, ditfloor, realgen on one window set (900 = 100 per day)
python scripts/baselines/setup_divergence_official.py --per-day 100 --seed 0
# 2. score each arm at each divergence horizon
for A in bookdit lobs5 realgen ditfloor; do
  bash scripts/baselines/run_divergence_official.sh $A 100 50
done
# 3. pickles -> curves -> depth-matched slopes -> bands
python scripts/baselines/summarize_divergence_official.py
python scripts/baselines/slopes_divergence_official.py --min-full 0.98
python scripts/figures/build_divergence_bands.py
```

Everything is written under `runs/divergence/`; step 3 writes `headline.csv` and `curve_bands.csv`
there and prints the control-adjusted ratio (3.12× at H = 100 and 3.08× at H = 50 in the
paper). `slopes_divergence_official.py --match` defaults to `bookdit lobs5 realgen`, which
fits those arms over a common depth; `ditfloor` is fitted over its own depth. `--min-full 0.98`
keeps a horizon group only if that fraction of windows still supplies a full sub-sequence. Five of
the 21 statistics (`ofi{,_up,_stay,_down}`, `vol_per_min`) are undefined on sub-sequences this short
and are excluded. The arms do not share an L1 level, so only the slope is like-for-like and the
figure plots each arm relative to its own first block.

The published 900-window set was drawn with `--per-day 100 --seed 0`, but the draw depends on which
windows every source tree keeps. The refinement setting, batch size (part of the draw) and decoder
revision of the published BookDiT and decoder-floor arms are not recorded in this release, and their
generation trees are not shipped, so a rerun reproduces the protocol rather than the exact slopes. The loop above therefore makes a choice the published run does not record: it wires the
unrefined one-shot generation (`tab1/raw`) as the BookDiT arm and the current decoder's floor
(`tab1/floor`) as `ditfloor`. Treat both as stand-ins; the floor in particular will not match
the paper's exactly.

## 4. The figures

Both result figures are drawn by `scripts/figures/fig_divergence_drift.py` from the §3.9 outputs:

```bash
python scripts/figures/fig_divergence_drift.py --only horizon-matched   # horizon_bands.pdf (Figure 2 drift panel)
python scripts/figures/fig_divergence_drift.py --only divergence        # divergence_drift.pdf (Figure 3)
```

Each writes a `.pdf` and a `.png` to `--outdir` (default `runs/figures`) and prints the numbers it
draws. The inputs default to `runs/horizon/bands` (`--matched-src`; `bootstrap.json`,
`curve_bands_l1_18.csv`) and `runs/divergence` (`--div-src`; `headline.csv`, `curve_bands.csv`).
Paper Figure 2's drift panel is drawn from the same two bands files as `horizon_bands.pdf`; its
left-hand pipeline panels are an illustration, as is Figure 1, and neither has a script here.

## 5. Reading a result

Most reported cells are one value in one `aggregates.csv` or JSON your run wrote under `runs/`. To
read the two §3.6 refined arms:

```bash
python -c "
import pandas as pd
d = pd.read_csv('runs/lobbench/tab1_refine/aggregates.csv')
print(d[['arm','n_windows','l1_21','l1_18','book_10']].round(3).to_string(index=False))"
```

The paper's checkpoint gives:

```
  arm  n_windows  l1_21  l1_18  book_10
  raw       3127  0.266  0.215    0.183
abs50       3127  0.249  0.193    0.187
```

A checkpoint you train yourself will differ (§3.3). The `raw` row is `apply_deadband.py`'s
deadband-off setting (§3.6). Table 1's "without refinement" row is the `nomed` row of
`runs/lobbench/tab1_nomed/aggregates.csv`; the `raw` row of `runs/lobbench/tab1` is the submitted
version's unrefined row (decoder's built-in band), kept for comparison.
