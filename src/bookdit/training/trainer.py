from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ..evaluation.reporting import history_to_frame, save_training_curves
from ..utils.logger import ExperimentLogger
from ..utils.reproducibility import build_run_name, ensure_directory, get_device, set_seed


@dataclass(slots=True)
class EpochResult:
    loss: float


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-3,
        device: str | None = None,
        output_dir: str | Path = "runs",
        run_name: str | None = None,
        seed: int = 42,
        grad_clip: float | None = None,
        log_every_steps: int | None = None,
        eval_max_batches: int | None = 50,
    ) -> None:
        set_seed(seed)
        self.device = get_device(device)
        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.grad_clip = grad_clip
        self.log_every_steps = log_every_steps
        # Cap the batches used for the lightweight in-epoch validation so logging
        # stays cheap; the full validation still runs at each epoch boundary.
        self.eval_max_batches = eval_max_batches
        self.run_dir = ensure_directory(Path(output_dir) / (run_name or build_run_name("run")))
        self.logger = ExperimentLogger(self.run_dir)
        self.history: list[dict[str, Any]] = []
        # Fine-grained, continuously-flushed records (one row per log interval).
        self.progress: list[dict[str, Any]] = []
        self.global_step = 0
        self._best_val_loss = float("inf")
        self._best_epoch = 0

    def _step(self, batch, training: bool = True):
        # A batch is the window tensor alone: the flow objective needs no target.
        x = batch.to(self.device).float()

        if training:
            self.optimizer.zero_grad(set_to_none=True)

        output = self.model(x)
        if not (isinstance(output, dict) and "loss" in output):
            raise ValueError("model must return a dictionary containing 'loss'")
        loss = output["loss"]

        if training:
            loss.backward()
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()

        return loss

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        epochs: int = 1,
        log_every_steps: int | None = None,
    ):
        log_every_steps = log_every_steps if log_every_steps is not None else self.log_every_steps
        try:
            steps_per_epoch: int | None = len(train_loader)
        except TypeError:                 # an IterableDataset without __len__
            steps_per_epoch = None
        # If the loader exposes a length and no explicit cadence was given, log
        # roughly five times per epoch so progress is visible on long runs.
        if log_every_steps is None and steps_per_epoch is not None and steps_per_epoch >= 5:
            log_every_steps = max(1, steps_per_epoch // 5)

        for epoch in range(epochs):
            self.model.train()
            train_losses: list[float] = []
            interval_losses: list[float] = []
            progress_bar = tqdm(train_loader, desc=f"epoch {epoch + 1}/{epochs}", leave=False)
            for batch in progress_bar:
                loss = self._step(batch, training=True)
                loss_value = float(loss.detach().cpu())
                train_losses.append(loss_value)
                interval_losses.append(loss_value)
                self.global_step += 1
                progress_bar.set_postfix(loss=f"{loss_value:.4f}")

                if log_every_steps and self.global_step % log_every_steps == 0:
                    interval = len(interval_losses)
                    self._log_progress(
                        train_loss=float(sum(interval_losses) / max(1, interval)),
                        val_loader=val_loader,
                        epoch=epoch,
                        steps_per_epoch=steps_per_epoch,
                        seen=len(train_losses),
                    )
                    interval_losses = []

            epoch_result = {
                "epoch": epoch + 1,
                "train_loss": float(sum(train_losses) / max(1, len(train_losses))),
            }
            if val_loader is not None:
                val_result = self.evaluate(val_loader)
                epoch_result["val_loss"] = val_result.loss
                if val_result.loss < self._best_val_loss:
                    self._best_val_loss = val_result.loss
                    self._best_epoch = epoch + 1
                    self.save_checkpoint("best.pt", epoch_result)

            self.save_checkpoint("last.pt", epoch_result)
            self.history.append(epoch_result)
            self._flush(epoch_done=True)

        summary = self._build_summary()
        self.logger.save_metrics({"history": self.history, "progress": self.progress, "summary": summary})
        self.logger.save_text("summary.txt", self.format_summary(summary))
        return summary

    def _log_progress(
        self,
        train_loss: float,
        val_loader: DataLoader | None,
        epoch: int,
        steps_per_epoch: int | None,
        seen: int,
    ) -> None:
        # Fractional epoch for the x-axis; 0 when the loader has no length.
        fraction = (seen / steps_per_epoch) if steps_per_epoch else 0.0
        record: dict[str, Any] = {
            "step": self.global_step,
            "epoch": round(epoch + min(fraction, 0.999), 4),
            "train_loss": train_loss,
        }

        if val_loader is not None:
            val_result = self.evaluate(val_loader, max_batches=self.eval_max_batches)
            record["val_loss"] = val_result.loss
            if val_result.loss < self._best_val_loss:
                self._best_val_loss = val_result.loss
                self._best_epoch = epoch + 1
                self.save_checkpoint("best.pt", record)
            self.model.train()

        self.progress.append(record)
        self.save_checkpoint("last.pt", record)
        self._flush(epoch_done=False)

    def _flush(self, epoch_done: bool) -> None:
        """Persist history/progress/plots so a run can be inspected mid-flight."""
        if self.history:
            history_to_frame(self.history).to_csv(self.run_dir / "history.csv", index=False)
        if self.progress:
            history_to_frame(self.progress).to_csv(self.run_dir / "progress.csv", index=False)
        records = self.progress if len(self.progress) >= 2 else self.history
        if records:
            save_training_curves(records, self.run_dir / "plots" / "training_curves.png")
        self.logger.save_metrics(
            {
                "history": self.history,
                "progress": self.progress,
                "summary": self._build_summary(allow_partial=True),
            }
        )

    @torch.no_grad()
    def evaluate(self, loader: DataLoader, max_batches: int | None = None) -> EpochResult:
        self.model.eval()
        losses: list[float] = []
        for batch_idx, batch in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            losses.append(float(self._step(batch, training=False).detach().cpu()))
        return EpochResult(loss=float(sum(losses) / max(1, len(losses))))

    def save_checkpoint(self, filename: str, metrics: dict[str, Any]) -> Path:
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
            "global_step": self.global_step,
        }
        path = self.logger.checkpoint_dir / filename
        torch.save(checkpoint, path)
        return path

    def _build_summary(self, allow_partial: bool = False) -> dict[str, Any]:
        if not self.history:
            if allow_partial:
                final_row = self.progress[-1] if self.progress else {}
            else:
                raise ValueError("Cannot build a summary without training history")
        else:
            final_row = self.history[-1]

        return {
            "best_epoch": self._best_epoch,
            "best_val_loss": self._best_val_loss if self._best_val_loss != float("inf") else None,
            "final_train_loss": final_row.get("train_loss"),
            "final_val_loss": final_row.get("val_loss"),
        }

    @staticmethod
    def format_summary(summary: dict[str, Any]) -> str:
        lines = ["Training summary"]
        for key, value in summary.items():
            lines.append(f"- {key}: {value}")
        return "\n".join(lines)
