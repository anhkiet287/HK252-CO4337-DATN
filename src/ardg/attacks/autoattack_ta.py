"""TorchAttacks AutoAttack wrapper."""

from typing import Any, Dict, TYPE_CHECKING

from ardg.attacks.torchattacks_utils import configure_normalization, require_torchattacks

if TYPE_CHECKING:
    from torch import nn


def build_autoattack_attack(cfg: Dict[str, Any], model: "nn.Module") -> Any:
    """Build an AutoAttack attack via TorchAttacks."""
    eps = float(cfg.get("eps", 8.0 / 255.0))
    norm = str(cfg.get("norm", "Linf"))
    version = str(cfg.get("version", "standard"))
    n_classes = int(cfg.get("n_classes", cfg.get("num_classes", 10)))
    seed = cfg.get("seed")
    verbose = bool(cfg.get("verbose", False))
    dataset_name = cfg.get("dataset_name", "cifar10")

    torchattacks = require_torchattacks()
    attack = torchattacks.AutoAttack(
        model,
        norm=norm,
        eps=eps,
        version=version,
        n_classes=n_classes,
        seed=seed,
        verbose=verbose,
    )
    return configure_normalization(attack, dataset_name)
