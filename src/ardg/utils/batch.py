"""Shared batch helpers for training/evaluation pipelines."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch


def as_xy_dict(batch: Any) -> Dict[str, Any]:
    """Normalize a batch into a mapping with at least ``x`` and ``y``."""
    if isinstance(batch, dict):
        if "x" not in batch or "y" not in batch:
            raise ValueError("Batch dict must contain 'x' and 'y'.")
        return dict(batch)
    if isinstance(batch, (list, tuple)) and len(batch) >= 2:
        return {"x": batch[0], "y": batch[1]}
    raise ValueError("Unsupported batch format; expected dict with x/y or tuple/list.")


def unpack_xy(batch: Any) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return ``(images, labels)`` from supported batch formats."""
    data = as_xy_dict(batch)
    return data["x"], data["y"]


def unpack_xyg(batch: Any, group_key: str = "g") -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(images, labels, groups)`` with labels as fallback groups."""
    if isinstance(batch, dict):
        if "x" not in batch or "y" not in batch:
            raise ValueError("Batch dict must contain 'x' and 'y'.")
        groups = batch.get(group_key, batch["y"])
        return batch["x"], batch["y"], groups
    if isinstance(batch, (list, tuple)):
        if len(batch) == 2:
            images, labels = batch
            return images, labels, labels
        if len(batch) >= 3:
            return batch[0], batch[1], batch[2]
    raise ValueError("Unsupported batch format for grouped objective.")


def move_to_device(batch: Any, device: torch.device) -> Any:
    """Recursively move tensors in a batch to the target device."""
    if isinstance(batch, dict):
        return {k: move_to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, (list, tuple)):
        moved = [move_to_device(item, device) for item in batch]
        return tuple(moved)
    return batch.to(device) if hasattr(batch, "to") else batch
