"""Preprocessing transforms for CIFAR datasets."""

from typing import Any


def build_transforms(cfg: dict, split: str) -> Any:
    """Build preprocessing transforms for a given split.

    Args:
        cfg: Configuration dictionary.
        split: Split name (train/val/test).

    Returns:
        Transform callable that outputs tensors of shape (3, 32, 32) normalized
        with CIFAR-10 mean/std.
    """
    raise NotImplementedError("TODO: implement transforms")
