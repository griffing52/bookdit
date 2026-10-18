from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class DataConfig:
    root: str = "data"
    window_size: int = 100
    normalization: str = "zscore"
    max_samples: int | None = None
    val_fraction: float = 0.1
    split_strategy: str = "chronological"
    stride: int = 1
    shuffle_buffer: int = 0


@dataclass(slots=True)
class ModelConfig:
    """The backbone name plus the five shape kwargs `BookDiT` reads.

    The defaults are the paper's model's `mini` shape (embed 128, depth 4, 4 heads); the
    network has no dropout, so there is no dropout setting.
    """

    name: str = "bookdit"
    embed_dim: int = 128
    depth: int = 4
    num_heads: int = 4
    mlp_ratio: int = 4


@dataclass(slots=True)
class TrainConfig:
    epochs: int = 3
    batch_size: int = 8
    lr: float = 1e-3
    seed: int = 42
    num_workers: int = 0
    device: str | None = None
    log_every_steps: int | None = None
    grad_clip: float | None = None


@dataclass(slots=True)
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    output_dir: str = "runs"
    run_name: str | None = None
