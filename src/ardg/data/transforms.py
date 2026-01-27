"""Preprocessing transforms for supported datasets."""

from typing import Any, Tuple

from torchvision import transforms

_DATASET_STATS = {
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    "cifar100": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    "mnist": ((0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081)),
    "fashion-mnist": ((0.2860, 0.2860, 0.2860), (0.3530, 0.3530, 0.3530)),
    "color-mnist": ((0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081)),
}


def _normalize_name(name: str) -> str:
    return name.lower().replace("_", "-")


def _get_stats(name: str) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    key = _normalize_name(name)
    if key not in _DATASET_STATS:
        raise ValueError(f"Unsupported dataset for transforms: {name}")
    return _DATASET_STATS[key]


def get_dataset_stats(name: str) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """Return normalization stats for a dataset.

    Args:
        name: Dataset name from config or CLI.

    Returns:
        Tuple of (mean, std) for 3-channel inputs.
    """
    return _get_stats(name)


def build_transforms(cfg: dict, split: str) -> Any:
    """Build preprocessing transforms for a given split.

    Args:
        cfg: Configuration dictionary.
        split: Split name (train/val/test).

    Returns:
        Transform callable that outputs tensors of shape (3, 32, 32) normalized
        with dataset-specific mean/std.
    """
    name = _normalize_name(cfg["dataset"]["name"])
    mean, std = _get_stats(name)

    if name in ("cifar10", "cifar100"):
        if split == "train":
            return transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            )
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

    if name in ("mnist", "fashion-mnist"):
        base = [
            transforms.Resize(32),
            transforms.Grayscale(num_output_channels=3),
        ]
        if split == "train":
            base.append(transforms.RandomCrop(32, padding=4))
        base.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        return transforms.Compose(base)

    if name == "color-mnist":
        base = [transforms.Resize(32)]
        if split == "train":
            base.append(transforms.RandomCrop(32, padding=4))
        base.append(transforms.Normalize(mean, std))
        return transforms.Compose(base)

    raise ValueError(f"Unsupported dataset for transforms: {name}")
