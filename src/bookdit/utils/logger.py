from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from .reproducibility import ensure_directory


class ExperimentLogger:
    def __init__(self, run_dir: str | Path):
        self.run_dir = ensure_directory(run_dir)
        self.checkpoint_dir = ensure_directory(self.run_dir / "checkpoints")
        self.metrics_path = self.run_dir / "metrics.yaml"
        self.config_path = self.run_dir / "config.yaml"

    def save_config(self, config: Any) -> None:
        payload = asdict(config) if hasattr(config, "__dataclass_fields__") else config
        with self.config_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(payload, handle, sort_keys=False)

    def save_metrics(self, metrics: dict[str, Any]) -> None:
        with self.metrics_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(metrics, handle, sort_keys=False)

    def save_text(self, name: str, contents: str) -> Path:
        path = self.run_dir / name
        path.write_text(contents, encoding="utf-8")
        return path
