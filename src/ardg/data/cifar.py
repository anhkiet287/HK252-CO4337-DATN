"""CIFAR dataset helpers."""

from typing import Any, Tuple


def load_cifar(name: str, root: str) -> Any:
    """Load a CIFAR dataset.

    Args:
        name: Dataset name (e.g., "cifar10", "cifar100").
        root: Data root directory.

    Returns:
        Dataset object.
    """
    raise NotImplementedError("TODO: implement CIFAR loader")


def get_dataloaders(cfg: dict) -> Tuple[Any, Any]:
    """Build train/val dataloaders for CIFAR.

    Args:
        cfg: Configuration dictionary.

    Returns:
        Tuple of (train_loader, val_loader). Each yields (images, labels) where
        images have shape (B, 3, 32, 32) and are normalized with CIFAR-10 mean/std,
        and labels have shape (B,) with dtype torch.long.
    """
    raise NotImplementedError("TODO: implement CIFAR dataloaders")
