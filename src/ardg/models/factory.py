"""Model factory for configurable architectures."""

from typing import Any, Dict

from torchvision.models.vision_transformer import ViT_B_16_Weights

from ardg.models.resnet import resnet18_cifar
from ardg.models.vit import vit_b16_cifar


def build_model(cfg: Dict[str, Any]) -> Any:
    """Build a model based on configuration.

    Args:
        cfg: Configuration dictionary with model settings.

    Returns:
        Instantiated model.

    Raises:
        NotImplementedError: When the requested model is not implemented yet.
        ValueError: When the model name is unsupported.
    """
    name = cfg["model"]["name"].lower()
    num_classes = int(cfg["model"]["num_classes"])

    if name == "resnet18_cifar":
        return resnet18_cifar(num_classes)

    if name in {"vit_b16_cifar", "vit_b_16", "vit_b16", "vit"}:
        # Optional transfer-learning flag; defaults to randomly initialized weights.
        pretrained = cfg["model"].get("pretrained", False)
        weights = ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        return vit_b16_cifar(num_classes=num_classes, weights=weights)

    raise ValueError(f"Unsupported model: {cfg['model']['name']}")
