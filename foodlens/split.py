"""Build the train/val/test manifest from an extracted Food-101 directory."""
import random
from pathlib import Path

import pandas as pd

from .classes import CLASS_TO_ID, CLASSES

N_TRAIN, N_VAL = 600, 150


def _read_meta(path: Path) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {c: [] for c in CLASSES}
    for line in path.read_text().splitlines():
        cls, name = line.strip().split("/")
        if cls in out:
            out[cls].append(name)
    return out


def make_manifest(food101_dir: Path, seed: int = 42, n_train: int = N_TRAIN, n_val: int = N_VAL) -> pd.DataFrame:
    """Official train -> (n_train train + n_val val) per class with a fixed seed; official test kept as test."""
    food101_dir = Path(food101_dir)
    train = _read_meta(food101_dir / "meta" / "train.txt")
    test = _read_meta(food101_dir / "meta" / "test.txt")
    rows = []
    for cls in CLASSES:
        names = sorted(train[cls])  # sort first so the shuffle is independent of file order
        random.Random(f"{seed}:{cls}").shuffle(names)
        assert len(names) >= n_train + n_val, f"{cls}: only {len(names)} train images"
        parts = {"train": names[:n_train], "val": names[n_train:n_train + n_val], "test": sorted(test[cls])}
        for split, items in parts.items():
            rows += [(f"images/{cls}/{n}.jpg", CLASS_TO_ID[cls], split) for n in items]
    return pd.DataFrame(rows, columns=["path", "class_id", "split"])
