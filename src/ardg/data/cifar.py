"""CIFAR dataset helpers."""

from typing import Any, Tuple

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from ardg.data.splits import ensure_split, load_splits

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def load_cifar(name: str, root: str, train: bool, transform: Any) -> Any:
    """Load a CIFAR dataset."""
    name = name.lower()
    if name == "cifar10":
        return datasets.CIFAR10(root=root, train=train, download=True, transform=transform)
    if name == "cifar100":
        return datasets.CIFAR100(root=root, train=train, download=True, transform=transform)
    raise ValueError(f"Unsupported dataset: {name}")


def _build_transforms(split: str) -> Any:
    if split == "train":
        return transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )


def get_dataloaders(cfg: dict) -> Tuple[Any, Any, Any]:
    """Build train/val/test dataloaders for CIFAR."""
    name = cfg["dataset"]["name"]
    data_dir = cfg["dataset"].get("data_dir", "data")
    batch_size = cfg["train"]["batch_size"]
    num_workers = cfg["dataset"].get("num_workers", 0)

    split_path = ensure_split(cfg)
    splits = load_splits(split_path)["splits"]

    train_transform = _build_transforms("train")
    eval_transform = _build_transforms("val")

    train_ds_aug = load_cifar(name, data_dir, train=True, transform=train_transform)
    train_ds_eval = load_cifar(name, data_dir, train=True, transform=eval_transform)
    test_ds = load_cifar(name, data_dir, train=False, transform=eval_transform)

    train_idx = splits["train"]
    val_idx = splits["val"]

    train_subset = Subset(train_ds_aug, train_idx)
    val_subset = Subset(train_ds_eval, val_idx)

    pin = torch.cuda.is_available()

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
    )
    return train_loader, val_loader, test_loader
