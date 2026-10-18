from __future__ import annotations

from pathlib import Path

import numpy as np

from bookdit.training import run_training
from bookdit.config import ExperimentConfig


def _write_day(tmp_path: Path, day: str, offset: float) -> None:
    n = 120
    book = np.column_stack(
        [
            np.linspace(100.0 + offset, 120.0 + offset, n),
            np.linspace(10.0, 20.0, n),
            np.linspace(99.5 + offset, 119.5 + offset, n),
            np.linspace(12.0, 22.0, n),
        ]
    )
    np.savetxt(tmp_path / f"TOY_{day}_orderbook.csv", book, delimiter=",")


def test_run_training_end_to_end(tmp_path: Path):
    """config -> loaders -> model -> Trainer runs and writes a checkpoint."""
    data_root = tmp_path / "data"
    data_root.mkdir()
    _write_day(data_root, "2020-01-02", 0.0)
    _write_day(data_root, "2020-01-03", 1.0)

    config = ExperimentConfig()
    config.data.root = str(data_root)
    config.data.window_size = 20
    config.data.stride = 10
    config.model.name = "wandit"
    config.model.embed_dim = 16
    config.model.depth = 1
    config.model.num_heads = 4
    config.train.epochs = 1
    config.train.batch_size = 2
    config.train.num_workers = 0
    config.train.device = "cpu"
    config.output_dir = str(tmp_path / "runs")
    config.run_name = "smoke"

    summary = run_training(config)

    assert summary["best_epoch"] == 1
    assert (tmp_path / "runs" / "smoke" / "checkpoints" / "last.pt").exists()
