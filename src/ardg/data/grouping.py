"""Minimal dataset helpers for native GroupDRO training."""

from __future__ import annotations

import math
from typing import Any, Iterator

import torch
from torch.utils.data import Dataset


class RepeatedGroupDataset(Dataset):
    """Repeat each base example once per configured group."""

    def __init__(self, base: Dataset, num_groups: int) -> None:
        if num_groups <= 0:
            raise ValueError(f"Expected num_groups >= 1, got {num_groups!r}.")
        if len(base) <= 0:
            raise ValueError("RepeatedGroupDataset requires a non-empty base dataset.")
        self.base = base
        self.base_size = len(base)
        self.num_groups = int(num_groups)

    def __len__(self) -> int:  # type: ignore[override]
        return self.base_size * self.num_groups

    def __getitem__(self, idx: int) -> dict[str, Any]:  # type: ignore[override]
        group_id = int(idx // self.base_size)
        sample = self.base[int(idx % self.base_size)]
        if isinstance(sample, dict):
            x, y = sample["x"], sample["y"]
        else:
            x, y = sample
        return {"x": x, "y": y, "group_id": group_id}


class GroupHomogeneousBatchSampler:
    """Yield batches where every sample belongs to the same group."""

    def __init__(
        self,
        *,
        base_size: int,
        num_groups: int,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
    ) -> None:
        if base_size <= 0:
            raise ValueError(f"Expected base_size >= 1, got {base_size!r}.")
        if num_groups <= 0:
            raise ValueError(f"Expected num_groups >= 1, got {num_groups!r}.")
        if batch_size <= 0:
            raise ValueError(f"Expected batch_size >= 1, got {batch_size!r}.")
        self.base_size = int(base_size)
        self.num_groups = int(num_groups)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)

    def __iter__(self) -> Iterator[list[int]]:
        batches: list[list[int]] = []
        for group_id in range(self.num_groups):
            start = group_id * self.base_size
            indices = list(range(start, start + self.base_size))
            if self.shuffle:
                order = torch.randperm(self.base_size).tolist()
                indices = [indices[idx] for idx in order]

            for batch_start in range(0, len(indices), self.batch_size):
                batch = indices[batch_start : batch_start + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                batches.append(batch)

        if self.shuffle and len(batches) > 1:
            order = torch.randperm(len(batches)).tolist()
            batches = [batches[idx] for idx in order]

        yield from batches

    def __len__(self) -> int:
        per_group = self.base_size // self.batch_size
        if not self.drop_last:
            per_group = math.ceil(self.base_size / self.batch_size)
        return per_group * self.num_groups
