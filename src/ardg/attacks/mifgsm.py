"""MIFGSM attack wrapper."""

from typing import Any, Dict, TYPE_CHECKING

from ardg.attacks.torchattacks_utils import configure_normalization, require_torchattacks

if TYPE_CHECKING:
    from torch import nn


def build_mifgsm_attack(cfg: Dict[str, Any], model: "nn.Module") -> Any:
    """Build an MIFGSM attack via TorchAttacks."""
    eps = float(cfg.get("eps", 8.0 / 255.0))
    alpha = float(cfg.get("step_size", cfg.get("alpha", 2.0 / 255.0)))
    steps = int(cfg.get("num_steps", cfg.get("steps", 10)))
    decay = float(cfg.get("decay", 1.0))
    dataset_name = cfg.get("dataset_name", "cifar10")
    norm_name = str(cfg.get("norm", "Linf")).lower()
    if norm_name not in {"linf", "inf"}:
        raise ValueError(f"Unsupported MIFGSM norm: {norm_name!r}. Use 'Linf'.")
    torchattacks = require_torchattacks()
    attack = torchattacks.MIFGSM(model, eps=eps, alpha=alpha, steps=steps, decay=decay)
    return configure_normalization(attack, dataset_name)
