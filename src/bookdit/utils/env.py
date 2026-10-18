from __future__ import annotations

from pathlib import Path
import os
import shlex


def load_dotenv(path: str | Path = ".env", override: bool = False) -> None:
    env_path = Path(path)
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        if not override and key in os.environ:
            continue
        value = value.strip()
        try:
            parts = shlex.split(value, comments=False, posix=True)
            value = parts[0] if len(parts) == 1 else value
        except ValueError:
            value = value.strip('"').strip("'")
        os.environ[key] = value
