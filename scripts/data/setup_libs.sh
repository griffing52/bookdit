#!/usr/bin/env bash
# Clone the third-party libraries the scoring lane needs into libs/ (git clones, not submodules,
# and gitignored). Training imports nothing from libs/, so this is only needed to score.
# lob_bench is pinned to the commit every published result was produced against.
#
# Usage:
#   bash scripts/data/setup_libs.sh                 # clone lob_bench if missing (idempotent)
#   bash scripts/data/setup_libs.sh --force         # re-checkout the pinned SHA even if present
#   bash scripts/data/setup_libs.sh --latest        # clone at default branch HEAD (unpinned)
#   bash scripts/data/setup_libs.sh --with-lobs5    # additionally clone + patch LOB-S5 (JAX baseline)
#
# LOB-S5 (MIT, Copyright (c) 2022 Linderman Lab) is cloned at its pinned SHA and patched with
# patches/lobs5-jax-compat.patch, which only ports it to current JAX / Flax / NumPy 2 (plus a
# message-only collate fix); it changes no model, loss or training setting.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LIBS="$ROOT/libs"

LOB_BENCH_URL="https://github.com/peernagy/lob_bench"
LOB_BENCH_SHA="276baf3039bf7630b555891b9d04ad49a92b3356"
LOB_BENCH_DEST="${LOB_BENCH_DIR:-$LIBS/lob_bench}"

LOBS5_URL="https://github.com/peernagy/LOBS5"
LOBS5_SHA="aa36029e9d348a017d00efc37c6b6dda89fab150"
LOBS5_DEST="${LOBS5_DIR:-$LIBS/LOBS5}"
LOBS5_PATCH="$ROOT/patches/lobs5-jax-compat.patch"

FORCE=0
PINNED=1
WITH_LOBS5=0
for arg in "$@"; do
  case "$arg" in
    --force)      FORCE=1 ;;
    --latest)     PINNED=0 ;;
    --with-lobs5) WITH_LOBS5=1 ;;
    -h|--help) sed -n '2,14p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown flag: $arg" >&2; exit 2 ;;
  esac
done

# clone_pinned <name> <url> <sha> <dest> -> clones if absent, re-checks-out the SHA under --force.
clone_pinned() {
  local name="$1" url="$2" sha="$3" dest="$4"

  if [ -d "$dest/.git" ]; then
    if [ "$FORCE" -eq 1 ] && [ "$PINNED" -eq 1 ]; then
      echo "==> $name: re-checking out $sha in $dest"
      git -C "$dest" fetch --quiet origin "$sha" 2>/dev/null || git -C "$dest" fetch --quiet origin
      git -C "$dest" checkout --quiet "$sha"
    else
      echo "==> $name: already present at $(git -C "$dest" rev-parse --short HEAD) in $dest (nothing to do)"
    fi
    return 0
  fi

  echo "==> $name: cloning $url -> $dest"
  mkdir -p "$(dirname "$dest")"
  git clone --quiet "$url" "$dest"
  if [ "$PINNED" -eq 1 ]; then
    git -C "$dest" checkout --quiet "$sha"
    echo "    pinned to $sha"
  else
    echo "    left at $(git -C "$dest" rev-parse --short HEAD) (default branch HEAD, unpinned)"
  fi
}

clone_pinned lob_bench "$LOB_BENCH_URL" "$LOB_BENCH_SHA" "$LOB_BENCH_DEST"

if [ "$WITH_LOBS5" -eq 1 ]; then
  if [ ! -f "$LOBS5_PATCH" ]; then
    echo "error: $LOBS5_PATCH is missing (it ships with this repo; re-checkout patches/)." >&2
    exit 1
  fi
  if [ "$PINNED" -eq 0 ]; then
    echo "    WARNING: --latest leaves LOB-S5 unpinned; the patch targets $LOBS5_SHA and may not apply." >&2
  fi
  clone_pinned LOBS5 "$LOBS5_URL" "$LOBS5_SHA" "$LOBS5_DEST"
  # JAX/Flax/NumPy-2 compatibility fixes for the pinned SHA; the training configuration is passed
  # on the run_train.py command line (docs/REPRODUCE.md), not baked into the patch.
  # The reverse check first, so re-running the script is a no-op rather than an error.
  if git -C "$LOBS5_DEST" apply --check --reverse "$LOBS5_PATCH" 2>/dev/null; then
    echo "    patch already applied: $(basename "$LOBS5_PATCH")"
  else
    git -C "$LOBS5_DEST" apply "$LOBS5_PATCH"
    echo "    applied $(basename "$LOBS5_PATCH")"
  fi
  echo "    NOTE: LOB-S5 needs its OWN JAX environment -- do not install it into this repo's venv."
  echo "    NOTE: export LOBS5_DIR=$LOBS5_DEST so the baseline drivers find it."
fi

echo
echo "libs/ state:"
printf '  %-10s %s\n' lob_bench "$(git -C "$LOB_BENCH_DEST" rev-parse HEAD 2>/dev/null || echo MISSING) ($LOB_BENCH_DEST)"
if [ "$WITH_LOBS5" -eq 1 ] || [ -d "$LOBS5_DEST/.git" ]; then
  printf '  %-10s %s\n' LOBS5 "$(git -C "$LOBS5_DEST" rev-parse HEAD 2>/dev/null || echo MISSING) ($LOBS5_DEST)"
else
  printf '  %-10s %s\n' LOBS5 "not cloned (re-run with --with-lobs5 if you need the baseline)"
fi
echo
echo "Verify the load-bearing one:"
echo "  uv run --no-sync python -c \"import sys; sys.path.insert(0,'${LOB_BENCH_DEST}'); import scoring, eval; print('lob_bench ok')\""
