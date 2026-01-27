"""Vision Transformer definitions tailored for 32x32 inputs (e.g. CIFAR)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from torchvision.models.vision_transformer import (
    ViT_B_16_Weights,
    VisionTransformer,
    vit_b_16,
)

if TYPE_CHECKING:
    from torch import Tensor


def vit_b16_cifar(
    num_classes: int,
    *,
    weights: Optional[ViT_B_16_Weights] = None,
    image_size: int = 32,
) -> VisionTransformer:
    """Create a ViT-B/16 that accepts 32x32 images.

    Args:
        num_classes: Number of output classes.
        weights: Optional torchvision ``ViT_B_16_Weights`` enum for transfer learning.
        image_size: Spatial size of the input images (defaults to 32 for CIFAR).

    Returns:
        VisionTransformer mapping (B, 3, image_size, image_size) -> (B, num_classes).
    """
    # torchvision's factory forwards all kwargs to VisionTransformer; overriding the
    # image_size keeps patchification valid for CIFAR-sized inputs.
    return vit_b_16(weights=weights, num_classes=num_classes, image_size=image_size)
