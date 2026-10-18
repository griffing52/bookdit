#!/usr/bin/env bash
# Painting the Market (arXiv:2509.05107) reproduction: the Painting arm of Tables 1 and 2.
#
#   1. train   one pass over the GOOG training split (train_painting_market.py defaults:
#              11,622 steps at effective batch 32, base 128, 156-state normalisation history,
#              0.95 clip quantile)
#   2. sample  one 100-state block per window over every window of the nine-day official source
#              tree (hf_lobs5_9d_f10), DDIM with 50 steps, decoded by the IDM decoder; also writes
#              book10.json (the ten book statistics over all generated windows)
#   3. score   the generated tree against BookDiT's refined arm on the official lane, restricted
#              to the windows both arms keep after the level-ten filter (847 of Painting's 3,136
#              windows survive it; 844 of those are shared with BookDiT)
#
# Usage: BOOKDIT_WORK=<work root> bash scripts/baselines/run_painting.sh [BOOKDIT_ARM_DIR]
#   BOOKDIT_ARM_DIR  the refined BookDiT generation tree to compare against (default
#                    $BOOKDIT_WORK/tab1_refine/abs50, the Table 1 'BookDiT + refinement' arm
#                    written by docs/REPRODUCE.md section 3.6)
# Env: PYTHON (default .venv/bin/python, else python3); LOB_BENCH_DIR is honoured by the scorer.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORK="${BOOKDIT_WORK:?set BOOKDIT_WORK to the LOB-Bench work root}"
cd "$ROOT"

PY="${PYTHON:-$([ -x "$ROOT/.venv/bin/python" ] && echo "$ROOT/.venv/bin/python" || echo python3)}"
RUN_ROOT="runs/baselines/painting_market"
CKPT="$RUN_ROOT/painting_goog_base128_s0.pt"
SOURCE="$WORK/lobbench_repro/lob_bench/sample_data/data_saved/hf_lobs5_9d_f10/GOOG"
EVAL="runs/lobbench/painting"
BOOKDIT_ARM="${1:-$WORK/tab1_refine/abs50}"
GEN_ROOT="$WORK/painting_vs_bookdit"

[ -d "$SOURCE/data_cond" ] || { echo "missing $SOURCE/data_cond (build_lobbench_hf_real.py)"; exit 1; }
mkdir -p "$RUN_ROOT"

"$PY" -u scripts/baselines/train_painting_market.py \
  --manifest runs/crossasset/corpus_manifest_2024.csv \
  --out "$CKPT" --ticker GOOG --steps 11622 --batch-size 1 --accum 32 --base 128 \
  --workers 2 --save-every 1000 --norm-history 156 --clip-quantile 0.95 --seed 0

"$PY" -u scripts/baselines/eval_painting_official.py \
  --checkpoint "$CKPT" --source "$SOURCE" --out "$EVAL" \
  --windows 0 --sample-steps 50 --blocks 1 --seed 20260801

# score_official.py takes one directory per arm under --gen-root; the arm names become row keys.
[ -d "$BOOKDIT_ARM" ] || { echo "missing BookDiT arm $BOOKDIT_ARM (see docs/REPRODUCE.md 3.6)"; exit 1; }
mkdir -p "$GEN_ROOT"
ln -sfn "$(cd "$EVAL/model/data_gen" && pwd)" "$GEN_ROOT/painting"
ln -sfn "$(cd "$BOOKDIT_ARM" && pwd)" "$GEN_ROOT/bookdit"
"$PY" -u scripts/baselines/score_official.py \
  --gen-root "$GEN_ROOT" --arms painting bookdit --out runs/lobbench/painting_vs_bookdit

echo "Painting pipeline complete: runs/lobbench/painting_vs_bookdit/aggregates.csv"
