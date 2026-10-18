"""Config -> loaders -> model -> Trainer, for LOBSTER-format CSV day pairs.

:func:`run_training` builds streaming loaders from an :class:`~bookdit.config.ExperimentConfig`,
instantiates the backbone named by ``config.model.name`` and fits it. The loader
(:mod:`bookdit.data.lobster`) feeds the raw orderbook columns with one global price z-score and
one global volume z-score; it does not build the paper's 41-channel ``midspread_dt`` chart. The
paper's training path is ``scripts/lib/train_bookdit.py``.
"""
from __future__ import annotations

from torch.utils.data import DataLoader

from ..data import (
    LobsterStreamDataset,
    compute_norm_stats,
    discover_lobster_pairs,
    split_lobster_pairs,
)
from ..models import build_model
from ..utils.reproducibility import build_run_name, set_seed
from .trainer import Trainer


def _build_loaders(config):
    pairs = discover_lobster_pairs(config.data.root)
    train_pairs, val_pairs = split_lobster_pairs(
        pairs,
        val_fraction=config.data.val_fraction,
        strategy=config.data.split_strategy,
    )

    common = dict(
        root=config.data.root,
        window_size=config.data.window_size,
        normalization=config.data.normalization,
        max_samples=config.data.max_samples,
        stride=config.data.stride,
    )

    # Normalization stats are computed on the training pairs only and reused
    # for validation to avoid look-ahead leakage.
    norm_stats = None
    if config.data.normalization.lower() in {"zscore", "standard", "standardize"}:
        norm_stats = compute_norm_stats(train_pairs)
    train_dataset = LobsterStreamDataset(
        pairs=train_pairs,
        norm_stats=norm_stats,
        shuffle_buffer=config.data.shuffle_buffer,
        seed=config.train.seed,
        **common,
    )
    val_dataset = LobsterStreamDataset(
        pairs=val_pairs,
        norm_stats=norm_stats,
        shuffle_buffer=0,
        seed=config.train.seed,
        **common,
    )
    train_loader = DataLoader(train_dataset, batch_size=config.train.batch_size, shuffle=False, num_workers=config.train.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=config.train.batch_size, shuffle=False, num_workers=config.train.num_workers)
    return train_dataset, train_loader, val_loader


def run_training(config):
    """Train a model from an in-memory config and return the run summary."""
    # Trainer seeds itself, but only after the model exists; seeding here as well makes the
    # weight init (and so the whole run) reproducible at a given seed and device.
    set_seed(config.train.seed)
    _, train_loader, val_loader = _build_loaders(config)
    sample = next(iter(train_loader))
    if sample.dim() == 3:
        sample = sample.unsqueeze(0)
    model = build_model(
        config.model.name,
        input_dim=sample.shape[-1],
        seq_len=sample.shape[-2],
        embed_dim=config.model.embed_dim,
        depth=config.model.depth,
        num_heads=config.model.num_heads,
        mlp_ratio=config.model.mlp_ratio,
    )
    trainer = Trainer(
        model=model,
        lr=config.train.lr,
        device=config.train.device,
        output_dir=config.output_dir,
        run_name=config.run_name or build_run_name(config.model.name),
        seed=config.train.seed,
        log_every_steps=config.train.log_every_steps,
        grad_clip=config.train.grad_clip,
    )
    trainer.logger.save_config(config)
    return trainer.fit(train_loader, val_loader=val_loader, epochs=config.train.epochs)
