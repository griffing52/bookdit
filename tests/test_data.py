from __future__ import annotations

from pathlib import Path

import numpy as np

from bookdit.data import LobsterStreamDataset, discover_lobster_pairs


def test_discover_lobster_pairs_and_dataset(tmp_path: Path):
    book = np.column_stack(
        [
            np.linspace(100.0, 120.0, 140),
            np.linspace(10.0, 20.0, 140),
            np.linspace(100.5, 120.5, 140),
            np.linspace(12.0, 22.0, 140),
            np.linspace(99.0, 119.0, 140),
            np.linspace(9.0, 19.0, 140),
        ]
    )
    book_path = tmp_path / "AAPL_20200101_book.csv"
    np.savetxt(book_path, book, delimiter=",")

    pairs = discover_lobster_pairs(tmp_path)
    assert len(pairs) == 1

    dataset = LobsterStreamDataset(root=tmp_path, window_size=20, max_samples=4)
    samples = list(dataset)
    assert len(samples) == 4
    # Accept either a bare window or an (x, ...) tuple, so the test checks the window only.
    x = samples[0][0] if isinstance(samples[0], (tuple, list)) else samples[0]
    assert x.shape == (1, 20, 6)
