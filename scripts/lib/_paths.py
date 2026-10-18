"""Machine-specific data roots, resolved from the environment — never hardcoded.

Every data root the drivers need comes from an environment variable, loaded from
`.env` (see `.env.example`; the parser is `src/bookdit/utils/env.py:8`, which does
**not** expand `~` or `${HOME}` — use absolute paths).

Resolution is **fail-loud**: an unset root raises `SystemExit` with the variable
name and what it means, rather than silently falling back to some previous owner's
home directory and producing zero windows.

This module lives in `scripts/lib/`. Other `scripts/lib/*.py` import it directly
(their own directory is `sys.path[0]`). A script in a sibling folder — e.g.
`scripts/data/build_corpus.py` — must put `scripts/lib` on the path first::

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))

    from _paths import lobster_cache, lobster_raw_root, contig_root

    RAW_DIR = lobster_cache() / "tensors_event"

Alongside the data roots there are three TOOL locations — `bench_work()`
(`$BOOKDIT_WORK`, where generation/scoring trees are written), `lobs5_dir()`
(`$LOBS5_DIR`) and `lob_bench_dir()` (`$LOB_BENCH_DIR`, which falls back to
`<repo>/libs/lob_bench`). The repo root itself is NOT an environment variable:
`repo_root()` resolves it from `__file__`.

Call these at USE time, not at import time: a module that resolves a root while
being imported makes every importer depend on a configured `.env`.
"""
from __future__ import annotations

import os
from pathlib import Path

_DOC = {
    "LOBSTER_CACHE": "root of the materialised LOBSTER cache (csv/, features/, tensors_event/, ...)",
    "LOBSTER_RAW_ROOT": "directory holding the raw LOBSTER .7z archives",
    "LOBSTER_CONTIG_ROOT": "the event-level per-day corpus (*_event10_contig.pt)",
    "LOBSTER_SCRATCH": "fast scratch directory for corpus-build temporaries",
    "BOOKDIT_WORK": "working root for LOB-Bench generation/scoring trees (cond/real/gen + scores)",
    "LOBS5_DIR": "the patched LOB-S5 clone (its own JAX environment; see scripts/data/setup_libs.sh)",
    "LOB_BENCH_DIR": "the pinned lob_bench clone (default: <repo>/libs/lob_bench)",
}


def _load_dotenv_once() -> None:
    if getattr(_load_dotenv_once, "_done", False):
        return
    try:
        from bookdit.utils.env import load_dotenv
    except Exception:  # package not importable (e.g. bare script use) — skip
        _load_dotenv_once._done = True  # type: ignore[attr-defined]
        return
    load_dotenv(repo_root() / ".env")
    _load_dotenv_once._done = True  # type: ignore[attr-defined]


def repo_root() -> Path:
    """Root of this checkout, resolved from `__file__` — never from the environment."""
    return Path(__file__).resolve().parents[2]


def env_path(name: str, *, default: str | None = None, required: bool = True) -> Path | None:
    """Return the Path in ``$name``, or exit with an actionable message."""
    _load_dotenv_once()
    val = os.environ.get(name) or default
    if not val or val.startswith("/ABSOLUTE/PATH/TO"):
        if not required:
            return None
        raise SystemExit(
            f"{name} is not set (or is still the .env.example placeholder).\n"
            f"  It should be: {_DOC.get(name, 'a machine-specific data root')}\n"
            f"  Fix: copy .env.example to .env and set {name} to an absolute path."
        )
    return Path(val).expanduser()


def lobster_cache(required: bool = True) -> Path:
    """Root of the materialised LOBSTER cache. `$LOBSTER_CACHE`."""
    return env_path("LOBSTER_CACHE", required=required)  # type: ignore[return-value]


def lobster_raw_root(required: bool = True) -> Path:
    """Directory of raw LOBSTER `.7z` archives. `$LOBSTER_RAW_ROOT`."""
    return env_path("LOBSTER_RAW_ROOT", required=required)  # type: ignore[return-value]


def contig_root(required: bool = True) -> Path:
    """Event-level per-day corpus root. `$LOBSTER_CONTIG_ROOT`, defaulting under the cache."""
    explicit = env_path("LOBSTER_CONTIG_ROOT", required=False)
    if explicit is not None:
        return explicit
    return lobster_cache(required=required) / "tensors_event_contig"


def scratch_dir(required: bool = False) -> Path | None:
    """Fast scratch for large temporaries. `$LOBSTER_SCRATCH`; None -> system temp."""
    return env_path("LOBSTER_SCRATCH", required=required)


def bench_work(required: bool = True) -> Path:
    """Working root for LOB-Bench generation/scoring trees. `$BOOKDIT_WORK`."""
    return env_path("BOOKDIT_WORK", required=required)  # type: ignore[return-value]


def lobs5_dir(required: bool = True) -> Path:
    """The patched LOB-S5 clone (separate JAX env). `$LOBS5_DIR`."""
    return env_path("LOBS5_DIR", required=required)  # type: ignore[return-value]


def lob_bench_dir(required: bool = False) -> Path:
    """The pinned lob_bench clone. `$LOB_BENCH_DIR`, defaulting to `<repo>/libs/lob_bench`."""
    explicit = env_path("LOB_BENCH_DIR", required=False)
    d = explicit if explicit is not None else repo_root() / "libs" / "lob_bench"
    if required and not d.is_dir():
        raise SystemExit(
            f"lob_bench is not present at {d}.\n"
            f"  It should be: {_DOC['LOB_BENCH_DIR']}\n"
            f"  Fix: bash scripts/data/setup_libs.sh (or set LOB_BENCH_DIR to an existing clone)."
        )
    return d
