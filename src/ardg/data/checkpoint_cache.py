"""Dataset helpers for checkpoint-based adversarial cache training."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Sequence

import torch
from torch.utils.data import Dataset


class CheckpointCacheDataset(Dataset):
    """Wrap a base dataset and swap its active samples with a cached attacked view."""

    def __init__(self, base: Dataset) -> None:
        if len(base) <= 0:
            raise ValueError("CheckpointCacheDataset requires a non-empty base dataset.")
        self.base = base
        self.cached_images: torch.Tensor | None = None
        self.cached_labels: torch.Tensor | None = None
        self.cached_attack_ids: torch.Tensor | None = None
        self.cache_metadata: Dict[str, Any] = {}

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.base)

    def __getitem__(self, idx: int) -> Dict[str, Any]:  # type: ignore[override]
        if not self.has_active_cache():
            raise RuntimeError("Checkpoint cache is not initialized for this dataset.")
        assert self.cached_images is not None
        assert self.cached_labels is not None
        assert self.cached_attack_ids is not None
        return {
            "x": self.cached_images[idx],
            "y": self.cached_labels[idx],
            "attack_id": self.cached_attack_ids[idx],
            "sample_idx": torch.tensor(int(idx), dtype=torch.long),
        }

    def has_active_cache(self) -> bool:
        return (
            self.cached_images is not None
            and self.cached_labels is not None
            and self.cached_attack_ids is not None
        )

    def get_clean_item(self, idx: int) -> tuple[Any, Any]:
        sample = self.base[int(idx)]
        if isinstance(sample, dict):
            return sample["x"], sample["y"]
        if isinstance(sample, (list, tuple)) and len(sample) >= 2:
            return sample[0], sample[1]
        raise ValueError("Unsupported clean sample format for checkpoint cache dataset.")

    def set_cache(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        attack_ids: torch.Tensor,
        *,
        metadata: Dict[str, Any] | None = None,
    ) -> None:
        num_samples = len(self)
        if int(images.size(0)) != num_samples:
            raise ValueError(
                f"Cached image count {int(images.size(0))} does not match dataset size {num_samples}."
            )
        if int(labels.numel()) != num_samples:
            raise ValueError(
                f"Cached label count {int(labels.numel())} does not match dataset size {num_samples}."
            )
        if int(attack_ids.numel()) != num_samples:
            raise ValueError(
                f"Cached attack-id count {int(attack_ids.numel())} does not match dataset size {num_samples}."
            )
        self.cached_images = images.detach().cpu().clone()
        self.cached_labels = labels.detach().cpu().clone().to(dtype=torch.long).view(-1)
        self.cached_attack_ids = attack_ids.detach().cpu().clone().to(dtype=torch.long).view(-1)
        self.cache_metadata = dict(metadata or {})

    def save_cache(self, path: str, *, metadata: Dict[str, Any] | None = None) -> str:
        if not self.has_active_cache():
            raise RuntimeError("Cannot save checkpoint cache before it has been initialized.")
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "images": self.cached_images,
            "labels": self.cached_labels,
            "attack_ids": self.cached_attack_ids,
            "metadata": dict(metadata or self.cache_metadata),
        }
        torch.save(payload, target)
        self.cache_metadata = dict(payload["metadata"])
        return str(target)

    def load_cache(self, path: str) -> Dict[str, Any]:
        source = Path(path)
        if not source.exists():
            raise FileNotFoundError(f"Checkpoint cache file not found: {source}")
        payload = torch.load(source, map_location="cpu")
        if not isinstance(payload, dict):
            raise ValueError(f"Checkpoint cache at {source} must be a mapping.")
        images = payload.get("images")
        labels = payload.get("labels")
        attack_ids = payload.get("attack_ids")
        if not isinstance(images, torch.Tensor) or not isinstance(labels, torch.Tensor) or not isinstance(attack_ids, torch.Tensor):
            raise ValueError(f"Checkpoint cache at {source} is missing tensors.")
        metadata = payload.get("metadata")
        self.set_cache(images, labels, attack_ids, metadata=metadata if isinstance(metadata, dict) else {})
        return dict(self.cache_metadata)


class CleanIndexDataset(Dataset):
    """View clean samples from a wrapped dataset for a fixed list of indices."""

    def __init__(self, dataset: CheckpointCacheDataset, indices: Sequence[int]) -> None:
        self.dataset = dataset
        self.indices = [int(idx) for idx in indices]

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, Any]:  # type: ignore[override]
        sample_idx = self.indices[int(idx)]
        image, label = self.dataset.get_clean_item(sample_idx)
        if not torch.is_tensor(label):
            label = torch.tensor(int(label), dtype=torch.long)
        else:
            label = label.detach().clone().to(dtype=torch.long)
        return {
            "x": image,
            "y": label,
            "sample_idx": torch.tensor(sample_idx, dtype=torch.long),
        }
