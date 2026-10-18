#!/usr/bin/env bash
# LOB-Bench's own divergence mode on the OFFICIAL windows.
# usage: run_divergence_official.sh <arm> <horizon> [<horizon> ...]
#   <arm> is a directory name under runs/divergence/data (setup_divergence_official.py builds
#   them: bookdit, lobs5, realgen, ditfloor); it is also the model name in the score pickles.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY=${PYTHON:-$([ -x "$ROOT/.venv/bin/python" ] && echo "$ROOT/.venv/bin/python" || echo python3)}
OUT=$ROOT/runs/divergence
[ $# -ge 2 ] || { echo "usage: $0 <arm> <horizon> [<horizon> ...]" >&2; exit 2; }
ARM=$1; shift
[ -d "$OUT/data/$ARM/GOOG" ] || { echo "missing $OUT/data/$ARM/GOOG; run setup_divergence_official.py first" >&2; exit 1; }
mkdir -p "$OUT/results/scores" "$OUT/logs"
cd "${LOB_BENCH_DIR:-$ROOT/libs/lob_bench}" || exit 1
for H in "$@"; do
  echo "=== $ARM H=$H  $(date -Is)"
  $PY -u run_bench.py --stock GOOG --model_name "$ARM" --time_period '' \
      --data_dir "$OUT/data" --save_dir "$OUT/results" \
      --div_only --div_error_bounds --divergence_horizon "$H" \
      > "$OUT/logs/${ARM}_H${H}.log" 2>&1
  echo "   rc=$? $(date -Is)"
done
echo "DONE $ARM"
