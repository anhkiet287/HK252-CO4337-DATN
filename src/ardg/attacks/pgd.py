"""PGD attack wrapper."""

from typing import Any, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from torch import nn


def build_pgd_attack(cfg: Dict[str, Any], model: "nn.Module") -> Any:
    """Build a PGD attack.

    Args:
        cfg: Configuration dictionary.
        model: Model expecting normalized inputs of shape (B, 3, 32, 32).

    Returns:
        Attack object that takes (images, labels) and returns adversarial images
        of shape (B, 3, 32, 32).

    Notes:
        Threat model: Linf.
        Normalization: Attack expects inputs already normalized with CIFAR-10 mean/std.
    """
    raise NotImplementedError("TODO: implement PGD attack")
