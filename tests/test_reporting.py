from __future__ import annotations

from pathlib import Path

from bookdit.evaluation.reporting import save_training_curves


def test_reporting_outputs(tmp_path: Path):
    history = [
        {"epoch": 1, "train_loss": 1.0, "val_loss": 0.9},
        {"epoch": 2, "train_loss": 0.8, "val_loss": 0.7},
    ]
    chart_path = save_training_curves(history, tmp_path / "plots" / "training_curves.png")
    assert chart_path.exists()
