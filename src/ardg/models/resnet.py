"""CIFAR-compatible ResNet definitions."""

from typing import TYPE_CHECKING

from torch import nn
from torchvision.models.resnet import BasicBlock, Bottleneck, ResNet

if TYPE_CHECKING:
    from torch import Tensor


class ResNetCIFAR(ResNet):
    """ResNet-18 with CIFAR stem (3x3 stride1 conv, no maxpool)."""

    def __init__(self, num_classes: int) -> None:
        super().__init__(block=BasicBlock, layers=[2, 2, 2, 2], num_classes=num_classes)
        # CIFAR stem: smaller receptive field, no initial downsample
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.maxpool = nn.Identity()


class ResNet50CIFAR(ResNet):
    """ResNet-50 with CIFAR stem (3x3 stride1 conv, no maxpool)."""

    def __init__(self, num_classes: int) -> None:
        super().__init__(block=Bottleneck, layers=[3, 4, 6, 3], num_classes=num_classes)
        # CIFAR stem: smaller receptive field, no initial downsample
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.maxpool = nn.Identity()


def resnet18_cifar(num_classes: int) -> nn.Module:
    """Create a CIFAR-style ResNet-18 model.

    Args:
        num_classes: Number of output classes.

    Returns:
        A torch.nn.Module mapping (B, 3, 32, 32) to logits of shape (B, C).
    """
    return ResNetCIFAR(num_classes=num_classes)


def resnet50_cifar(num_classes: int) -> nn.Module:
    """Create a CIFAR-style ResNet-50 model.

    Args:
        num_classes: Number of output classes.

    Returns:
        A torch.nn.Module mapping (B, 3, 32, 32) to logits of shape (B, C).
    """
    return ResNet50CIFAR(num_classes=num_classes)
