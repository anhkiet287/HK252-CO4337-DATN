"""Model factory for configurable architectures."""

from typing import Any, Dict

from ardg.models.resnet import resnet18_cifar


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

    if name in {"vit", "vit_tiny", "vit_small", "vit_base"}:
        raise NotImplementedError("Vision Transformer support is not implemented yet.")

    raise ValueError(f"Unsupported model: {cfg['model']['name']}")
