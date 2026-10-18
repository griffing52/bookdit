from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


def save_training_curves(history: list[dict[str, Any]], output_path: str | Path) -> Path:
    if not history:
        raise ValueError("Cannot plot empty training history")

    frame = pd.DataFrame(history)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Prefer the finest-grained x-axis available so in-progress runs render a
    # smooth curve rather than 1-2 epoch points.
    x_key = "step" if "step" in frame.columns else "epoch"
    x_label = "Step" if x_key == "step" else "Epoch"

    figure, axis = plt.subplots(figsize=(8, 5), dpi=160)
    if "train_loss" in frame.columns:
        train = frame.dropna(subset=["train_loss"])
        axis.plot(train[x_key], train["train_loss"], label="train_loss", linewidth=2)
    if "val_loss" in frame.columns:
        val = frame.dropna(subset=["val_loss"])
        axis.plot(val[x_key], val["val_loss"], label="val_loss", linewidth=2)
    axis.set_xlabel(x_label)
    axis.set_ylabel("Loss")
    axis.set_title("Training and validation loss")
    axis.grid(True, alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output_path)
    plt.close(figure)
    return output_path


def history_to_frame(history: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(history)
    if frame.empty:
        return frame
    ordered_columns = [column for column in ["epoch", "train_loss", "val_loss"] if column in frame.columns]
    remaining_columns = [column for column in frame.columns if column not in ordered_columns]
    return frame[ordered_columns + remaining_columns]
