"""CIFAR-compatible ResNet definitions."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from torch import nn


class ResNet:
    """CIFAR-compatible ResNet placeholder.

    Notes:
        Expects inputs of shape (B, 3, 32, 32) float32 normalized with CIFAR-10 mean/std.
        Outputs logits of shape (B, C).
    """

    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError("TODO: implement ResNet")

    def forward(self, inputs: "torch.Tensor") -> "torch.Tensor":
        """Forward pass.

        Args:
            inputs: Image tensor of shape (B, 3, 32, 32), float32 normalized.

        Returns:
            Logits tensor of shape (B, C).
        """
        raise NotImplementedError("TODO: implement forward")


def resnet18_cifar(num_classes: int) -> "nn.Module":
    """Create a CIFAR-style ResNet-18 model.

    Args:
        num_classes: Number of output classes.

    Returns:
        A torch.nn.Module mapping (B, 3, 32, 32) to logits of shape (B, C).
    """
    raise NotImplementedError("TODO: implement ResNet-18 factory")
