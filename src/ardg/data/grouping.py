"""Dataset wrappers for group-aware training (draft)."""

from __future__ import annotations

from typing import Any, Tuple

from torch.utils.data import Dataset


class GroupedDataset(Dataset):
    """Wrap a dataset to return (x, y, g). If no group_ids provided, uses labels."""

    def __init__(self, base: Dataset, group_ids: Any = None) -> None:
        self.base = base
        self.group_ids = group_ids

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.base)

    def __getitem__(self, idx: int) -> Tuple[Any, Any, Any]:  # type: ignore[override]
        sample = self.base[idx]
        if isinstance(sample, dict):
            x, y = sample["x"], sample["y"]
        else:
            x, y = sample
        g = self.group_ids[idx] if self.group_ids is not None else y
        return x, y, g
