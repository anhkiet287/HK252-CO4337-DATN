"""FAB attack wrapper."""

from typing import Any, Dict, TYPE_CHECKING

from ardg.attacks.torchattacks_utils import configure_normalization, require_torchattacks

if TYPE_CHECKING:
    from torch import nn


def build_fab_attack(cfg: Dict[str, Any], model: "nn.Module") -> Any:
    """Build a FAB attack via TorchAttacks."""
    norm = str(cfg.get("norm", "Linf"))
    eps = float(cfg.get("eps", 8.0 / 255.0))
    steps = int(cfg.get("num_steps", cfg.get("steps", 10)))
    n_restarts = int(cfg.get("n_restarts", cfg.get("restarts", 1)))
    alpha_max = float(cfg.get("alpha_max", 0.1))
    eta = float(cfg.get("eta", 1.05))
    beta = float(cfg.get("beta", 0.9))
    verbose = bool(cfg.get("verbose", False))
    seed = int(cfg.get("seed", 0))
    multi_targeted = bool(cfg.get("multi_targeted", False))
    n_classes = int(cfg.get("n_classes", 10))
    dataset_name = cfg.get("dataset_name", "cifar10")
    torchattacks = require_torchattacks()
    attack = torchattacks.FAB(
        model,
        norm=norm,
        eps=eps,
        steps=steps,
        n_restarts=n_restarts,
        alpha_max=alpha_max,
        eta=eta,
        beta=beta,
        verbose=verbose,
        seed=seed,
        multi_targeted=multi_targeted,
        n_classes=n_classes,
    )
    return configure_normalization(attack, dataset_name)
