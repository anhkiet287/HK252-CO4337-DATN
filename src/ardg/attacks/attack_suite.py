"""Attack factory for TorchAttacks suites."""

from typing import Any, Dict, TYPE_CHECKING

from ardg.attacks.pgd import build_pgd_attack

if TYPE_CHECKING:
    from torch import nn


def build_train_attack(cfg: Dict[str, Any], model: "nn.Module") -> Any:
    """Build the training-time adversarial attack (PGD Linf)."""
    model.eval()
    return build_pgd_attack(cfg["attack"]["train"], model)


def build_val_attack(cfg: Dict[str, Any], model: "nn.Module") -> Any:
    """Build the validation-time adversarial attack (PGD Linf)."""
    model.eval()
    return build_pgd_attack(cfg["attack"]["val"], model)


def build_eval_attacks(cfg: Dict[str, Any], model: "nn.Module") -> Dict[str, Any]:
    """Build evaluation-time adversarial attacks.

    Returns:
        Mapping from attack name to attack object.

    Notes:
        Threat model: Linf.
        Normalization: Attack expects inputs already normalized with CIFAR-10 mean/std.
    """
    model.eval()
    eval_cfg = cfg["attack"]["eval"]
    attacks: Dict[str, Any] = {}
    attacks["pgd"] = build_pgd_attack(
        {
            "eps": eval_cfg.get("eps", cfg["attack"]["val"]["eps"]),
            "step_size": eval_cfg.get("step_size", cfg["attack"]["val"].get("step_size", 0.007843)),
            "num_steps": eval_cfg.get("num_steps", cfg["attack"]["val"].get("num_steps", 20)),
        },
        model,
    )
    return attacks
