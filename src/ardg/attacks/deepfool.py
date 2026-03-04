"""DeepFool attack wrapper."""

from typing import Any, Dict, TYPE_CHECKING

from ardg.attacks.torchattacks_utils import configure_normalization, require_torchattacks

if TYPE_CHECKING:
    from torch import nn


def build_deepfool_attack(cfg: Dict[str, Any], model: "nn.Module") -> Any:
    """Build a DeepFool attack via TorchAttacks."""
    steps = int(cfg.get("num_steps", cfg.get("steps", 50)))
    overshoot = float(cfg.get("overshoot", 0.02))
    dataset_name = cfg.get("dataset_name", "cifar10")
    torchattacks = require_torchattacks()
    attack = torchattacks.DeepFool(model, steps=steps, overshoot=overshoot)
    return configure_normalization(attack, dataset_name)
