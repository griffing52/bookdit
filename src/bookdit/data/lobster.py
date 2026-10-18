from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch

# LOBSTER stores empty book levels with these sentinel prices (and zero volume).
# Any |price| at/above this magnitude is treated as a missing level.
PLACEHOLDER_THRESHOLD = 1e8


@dataclass(slots=True)
class LobsterFilePair:
    prefix: str
    book_path: Path
    message_path: Path | None = None
    ticker: str | None = None
    date: str | None = None
    session: str | None = None


@dataclass(slots=True)
class NormStats:
    """Global price/volume normalization statistics.

    Price columns and volume columns are standardized *separately* using statistics
    pooled across a reference set of days (here, the training split) rather than
    per-window. This preserves cross-window level/trend structure that
    per-window z-scoring destroys.
    """

    price_mean: float
    price_std: float
    volume_mean: float
    volume_std: float

    def as_dict(self) -> dict[str, float]:
        return {
            "price_mean": self.price_mean,
            "price_std": self.price_std,
            "volume_mean": self.volume_mean,
            "volume_std": self.volume_std,
        }


def _load_matrix(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        matrix = np.load(path)
    elif suffix == ".npz":
        archive = np.load(path)
        first_key = archive.files[0]
        matrix = archive[first_key]
    elif suffix in {".csv", ".txt", ".tsv"}:
        separator = {".csv": ",", ".tsv": "\t"}.get(suffix, None)
        matrix = pd.read_csv(path, header=None, sep=separator).to_numpy()
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")
    return np.asarray(matrix, dtype=np.float32)


def _strip_dataset_suffix(stem: str) -> str:
    match = re.search(r"_(book|orderbook|message)(?:[_\.-].*)?$", stem, flags=re.IGNORECASE)
    if match is None:
        return stem
    return stem[: match.start()]


def _parse_pair_metadata(prefix: str) -> tuple[str | None, str | None, str | None]:
    parts = prefix.split("_")
    if len(parts) < 2:
        return None, None, None
    ticker = parts[0]
    date = parts[1] if re.match(r"\d{4}-\d{2}-\d{2}", parts[1]) else None
    session = None
    if len(parts) >= 4 and parts[2].isdigit() and parts[3].isdigit():
        session = f"{parts[2]}_{parts[3]}"
    return ticker, date, session


def _expand_roots(root: str | Path | list) -> list[Path]:
    """Expand a root spec into concrete base paths.

    Accepts a single path, a list of paths, a comma-separated string, and/or
    glob patterns (e.g. ``data/*_2019-*`` to select every directory for 2019).
    """
    if isinstance(root, (list, tuple)):
        items = [str(r) for r in root]
    else:
        text = str(root)
        items = [part.strip() for part in text.split(",")] if "," in text else [text]

    expanded: list[Path] = []
    for item in items:
        if any(ch in item for ch in "*?[]"):
            expanded.extend(sorted(Path().glob(item)))
        else:
            expanded.append(Path(item))
    return expanded


def discover_lobster_pairs(root: str | Path | list) -> list[LobsterFilePair]:
    roots = _expand_roots(root)
    book_files: dict[str, Path] = {}
    message_files: dict[str, Path] = {}
    saw_existing_root = False

    for root_path in roots:
        if not root_path.exists():
            continue
        saw_existing_root = True
        candidates = [root_path] if root_path.is_file() else root_path.rglob("*")
        for path in candidates:
            if not path.is_file():
                continue
            stem = path.stem.lower()
            if "orderbook" in stem or re.search(r"(^|_)book(_|$)", stem):
                book_files[_strip_dataset_suffix(path.stem)] = path
            elif "message" in stem:
                message_files[_strip_dataset_suffix(path.stem)] = path

    if not saw_existing_root:
        raise FileNotFoundError(f"No existing LOBSTER root matched: {root}")

    pairs: list[LobsterFilePair] = []
    for prefix, book_path in sorted(book_files.items()):
        ticker, date, session = _parse_pair_metadata(prefix)
        pairs.append(
            LobsterFilePair(
                prefix=prefix,
                book_path=book_path,
                message_path=message_files.get(prefix),
                ticker=ticker,
                date=date,
                session=session,
            )
        )

    if not pairs:
        raise FileNotFoundError(f"No LOBSTER book files found under {root}")
    return pairs


def split_lobster_pairs(
    pairs: list[LobsterFilePair],
    val_fraction: float = 0.1,
    strategy: str = "chronological",
) -> tuple[list[LobsterFilePair], list[LobsterFilePair]]:
    """Hold the latest ``val_fraction`` of pairs out for validation.

    Pairs are ordered by date first (then ticker), so with several tickers the validation set
    is the most recent days across all of them, not the alphabetically last ticker.
    """
    if not pairs:
        raise ValueError("Cannot split an empty pair list")
    if strategy.lower() != "chronological":
        raise ValueError(f"Unsupported split strategy: {strategy}")

    ordered = sorted(
        pairs,
        key=lambda pair: (
            pair.date or "",
            pair.ticker or "",
            pair.session or "",
            pair.prefix,
        ),
    )
    val_count = max(1, int(round(len(ordered) * val_fraction)))
    val_pairs = ordered[-val_count:]
    train_pairs = ordered[:-val_count]
    if not train_pairs:
        raise ValueError("Validation fraction leaves no training pairs")
    return train_pairs, val_pairs


def _clean_book(book: np.ndarray) -> np.ndarray:
    """Drop rows containing LOBSTER empty-level sentinels in any price column.

    Price columns are the even indices (ask/bid prices); these carry the
    ``+/-9999999999`` placeholders for unpopulated levels (volumes are 0 there).
    Such rows corrupt the normalization statistics, so we remove them.
    """
    price_columns = book[:, 0::2]
    valid = np.all(np.abs(price_columns) < PLACEHOLDER_THRESHOLD, axis=1)
    return book[valid]


def compute_norm_stats(pairs: Iterable[LobsterFilePair]) -> NormStats:
    """Pool price/volume statistics across the given (training) pairs."""
    price_sum = price_sq = price_count = 0.0
    volume_sum = volume_sq = volume_count = 0.0
    for pair in pairs:
        features = _clean_book(_load_matrix(pair.book_path))
        if features.size == 0:
            continue
        prices = features[:, 0::2].astype(np.float64)
        volumes = features[:, 1::2].astype(np.float64)
        price_sum += prices.sum()
        price_sq += np.square(prices).sum()
        price_count += prices.size
        volume_sum += volumes.sum()
        volume_sq += np.square(volumes).sum()
        volume_count += volumes.size

    if price_count == 0 or volume_count == 0:
        raise ValueError("Cannot compute normalization stats from empty data")

    price_mean = price_sum / price_count
    price_std = math.sqrt(max(price_sq / price_count - price_mean**2, 0.0))
    volume_mean = volume_sum / volume_count
    volume_std = math.sqrt(max(volume_sq / volume_count - volume_mean**2, 0.0))
    return NormStats(
        price_mean=price_mean,
        price_std=price_std or 1.0,
        volume_mean=volume_mean,
        volume_std=volume_std or 1.0,
    )


def _apply_norm(features: np.ndarray, stats: NormStats) -> np.ndarray:
    out = features.astype(np.float32, copy=True)
    out[:, 0::2] = (out[:, 0::2] - stats.price_mean) / stats.price_std
    out[:, 1::2] = (out[:, 1::2] - stats.volume_mean) / stats.volume_std
    return out


def _maybe_per_window(window: np.ndarray, normalization: str) -> np.ndarray:
    """Per-window standardization: each feature is zero-mean, unit-std *within the
    window*, which removes the absolute price level and scale that make different
    instruments and eras look distributionally distinct. A no-op unless the
    normalization is 'per_window'."""
    if str(normalization).lower() != "per_window":
        return window
    w = window.astype(np.float32, copy=True)
    mu = w.mean(axis=0, keepdims=True)
    sd = w.std(axis=0, keepdims=True) + 1e-6
    return (w - mu) / sd


class LobsterStreamDataset(torch.utils.data.IterableDataset):
    """Streaming LOBSTER book windows over the raw orderbook columns.

    Rows carrying an empty-level sentinel are dropped, then windows of
    ``window_size`` consecutive rows are emitted at the given ``stride`` with global
    price/volume normalization applied. Work is sharded across DataLoader workers so
    each window is emitted exactly once; ``shuffle_buffer`` decorrelates the heavily
    overlapping consecutive windows. Each item is one ``(1, window_size, n_columns)``
    float tensor; the flow objective needs no target.

    This is a generic raw-column loader. The paper's model is trained on the 41-channel
    ``midspread_dt`` chart instead (``scripts/lib/lob_representation.py``).
    """

    def __init__(
        self,
        root: str | Path,
        pairs: Iterable[LobsterFilePair] | None = None,
        window_size: int = 100,
        normalization: str = "zscore",
        max_samples: int | None = None,
        stride: int = 1,
        norm_stats: NormStats | None = None,
        shuffle_buffer: int = 0,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.window_size = window_size
        self.normalization = normalization
        self.max_samples = max_samples
        self.stride = max(1, int(stride))
        self.shuffle_buffer = max(0, int(shuffle_buffer))
        self.seed = seed
        self.pairs = list(pairs) if pairs is not None else discover_lobster_pairs(self.root)

        normalize = self.normalization.lower() in {"zscore", "standard", "standardize", "price_volume_zscore", "lobster_zscore"}
        if normalize and norm_stats is None:
            norm_stats = compute_norm_stats(self.pairs)
        self.norm_stats = norm_stats if normalize else None

    def _iter_windows(self, pair: LobsterFilePair):
        features = _clean_book(_load_matrix(pair.book_path))
        if len(features) < self.window_size:
            return
        if self.norm_stats is not None:
            features = _apply_norm(features, self.norm_stats)
        n = len(features)
        for start in range(0, n - self.window_size + 1, self.stride):
            window = _maybe_per_window(features[start : start + self.window_size], self.normalization)
            yield torch.from_numpy(np.ascontiguousarray(window)).unsqueeze(0)

    def _raw_iter(self):
        info = torch.utils.data.get_worker_info()
        pairs = self.pairs
        if info is not None and info.num_workers > 1:
            pairs = pairs[info.id :: info.num_workers]
        for pair in pairs:
            yield from self._iter_windows(pair)

    def __iter__(self):
        emitted = 0
        if self.shuffle_buffer > 1:
            rng = np.random.default_rng(self.seed)
            buffer: list = []
            for sample in self._raw_iter():
                buffer.append(sample)
                if len(buffer) >= self.shuffle_buffer:
                    pick = int(rng.integers(len(buffer)))
                    buffer[pick], buffer[-1] = buffer[-1], buffer[pick]
                    yield buffer.pop()
                    emitted += 1
                    if self.max_samples is not None and emitted >= self.max_samples:
                        return
            rng.shuffle(buffer)
            for sample in buffer:
                yield sample
                emitted += 1
                if self.max_samples is not None and emitted >= self.max_samples:
                    return
            return

        for sample in self._raw_iter():
            yield sample
            emitted += 1
            if self.max_samples is not None and emitted >= self.max_samples:
                return
