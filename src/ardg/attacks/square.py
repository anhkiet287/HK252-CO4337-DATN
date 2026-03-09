"""Square attack wrapper."""

from typing import Any, Dict, TYPE_CHECKING

from ardg.attacks.torchattacks_utils import configure_normalization, require_torchattacks

if TYPE_CHECKING:
    from torch import nn


def build_square_attack(cfg: Dict[str, Any], model: "nn.Module") -> Any:
    """Build a Square attack via TorchAttacks."""
    norm = str(cfg.get("norm", "Linf"))
    eps = float(cfg.get("eps", 8.0 / 255.0))
    n_queries = int(cfg.get("n_queries", 5000))
    n_restarts = int(cfg.get("n_restarts", cfg.get("restarts", 1)))
    p_init = float(cfg.get("p_init", 0.8))
    loss = str(cfg.get("loss", "margin"))
    resc_schedule = bool(cfg.get("resc_schedule", True))
    seed = int(cfg.get("seed", 0))
    verbose = bool(cfg.get("verbose", False))
    dataset_name = cfg.get("dataset_name", "cifar10")
    torchattacks = require_torchattacks()
    attack = torchattacks.Square(
        model,
        norm=norm,
        eps=eps,
        n_queries=n_queries,
        n_restarts=n_restarts,
        p_init=p_init,
        loss=loss,
        resc_schedule=resc_schedule,
        seed=seed,
        verbose=verbose,
    )
    return configure_normalization(attack, dataset_name)
