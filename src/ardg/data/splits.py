"""Dataset split helpers."""

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from torchvision import datasets, transforms


def _load_targets(dataset: str, data_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Download dataset targets for stratified splitting."""
    name = dataset.lower()
    if name == "cifar10":
        ds = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transforms.ToTensor())
    elif name == "cifar100":
        ds = datasets.CIFAR100(root=data_dir, train=True, download=True, transform=transforms.ToTensor())
    else:
        raise ValueError(f"Unsupported dataset for splitting: {dataset}")
    targets = np.array(ds.targets)
    indices = np.arange(len(targets))
    return indices, targets


def _needs_regen(split_path: Path, dataset: str, seed: int, val_ratio: float) -> bool:
    if not split_path.exists():
        return True
    try:
        with split_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        splits = payload.get("splits", {})
        if payload.get("dataset") != dataset or int(payload.get("seed", -1)) != seed:
            return True
        if float(payload.get("val_ratio", -1.0)) != val_ratio:
            return True
        return not (isinstance(splits, dict) and "train" in splits and "val" in splits and splits["train"] and splits["val"])
    except Exception:
        return True


def ensure_split(cfg: Dict[str, Any]) -> str:
    """Ensure a deterministic train/val split artifact exists."""
    dataset = cfg["dataset"]["name"]
    seed = int(cfg["experiment"]["seed"])
    val_ratio = float(cfg["dataset"]["val_ratio"])
    data_dir = Path(cfg["dataset"].get("data_dir", "data"))

    token = f"{val_ratio:.2f}".replace(".", "p")
    filename = f"{dataset}_seed{seed}_val{token}.json"
    split_dir = data_dir / "processed" / "splits"
    split_path = split_dir / filename

    if _needs_regen(split_path, dataset, seed, val_ratio):
        split_dir.mkdir(parents=True, exist_ok=True)
        all_idx, targets = _load_targets(dataset, data_dir)
        sss = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
        train_idx, val_idx = next(sss.split(all_idx, targets))
        payload = {
            "dataset": dataset,
            "seed": seed,
            "val_ratio": val_ratio,
            "splits": {
                "train": train_idx.tolist(),
                "val": val_idx.tolist(),
            },
        }
        with split_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    return str(split_path)


def load_splits(path: str) -> Dict[str, Any]:
    """Load a split artifact from disk.

    Args:
        path: Path to a JSON split file.

    Returns:
        Dictionary containing split metadata and indices.
    """
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)
