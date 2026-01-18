"""Attack factory for TorchAttacks suites."""

from typing import Any, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from torch import nn


def build_train_attack(cfg: Dict[str, Any], model: "nn.Module") -> Any:
    """Build the training-time adversarial attack.

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
    raise NotImplementedError("TODO: implement build_train_attack")


def build_val_attack(cfg: Dict[str, Any], model: "nn.Module") -> Any:
    """Build the validation-time adversarial attack.

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
    raise NotImplementedError("TODO: implement build_val_attack")


def build_eval_attacks(cfg: Dict[str, Any], model: "nn.Module") -> Dict[str, Any]:
    """Build evaluation-time adversarial attacks.

    Args:
        cfg: Configuration dictionary.
        model: Model expecting normalized inputs of shape (B, 3, 32, 32).

    Returns:
        Mapping from attack name to attack object.

    Notes:
        Threat model: Linf.
        Normalization: Attack expects inputs already normalized with CIFAR-10 mean/std.
    """
    raise NotImplementedError("TODO: implement build_eval_attacks")
