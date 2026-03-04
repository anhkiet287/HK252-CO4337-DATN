"""Attack factory for TorchAttacks suites."""

from typing import Any, Dict, TYPE_CHECKING

from ardg.attacks.cw import build_cw_attack
from ardg.attacks.deepfool import build_deepfool_attack
from ardg.attacks.fab import build_fab_attack
from ardg.attacks.fgsm import build_fgsm_attack
from ardg.attacks.pgd import build_pgd_attack
from ardg.attacks.square import build_square_attack

if TYPE_CHECKING:
    from torch import nn


def _build_attack_from_cfg(atk_cfg: Dict[str, Any], model: "nn.Module") -> Any:
    name = str(atk_cfg.get("name", "pgd")).lower()
    if name in {"pgd", "pgd_linf"}:
        return build_pgd_attack(atk_cfg, model)
    if name == "fgsm":
        return build_fgsm_attack(atk_cfg, model)
    if name in {"cw", "carlini-wagner", "carlini_wagner"}:
        return build_cw_attack(atk_cfg, model)
    if name == "deepfool":
        return build_deepfool_attack(atk_cfg, model)
    if name == "square":
        return build_square_attack(atk_cfg, model)
    if name == "fab":
        return build_fab_attack(atk_cfg, model)
    raise ValueError(f"Unsupported attack name: {name}")


def build_train_attack(cfg: Dict[str, Any], model: "nn.Module") -> Any:
    """Build the training-time adversarial attack."""
    model.eval()
    atk_cfg = dict(cfg["attack"]["train"])
    atk_cfg["dataset_name"] = cfg["dataset"]["name"]
    return _build_attack_from_cfg(atk_cfg, model)


def build_val_attack(cfg: Dict[str, Any], model: "nn.Module") -> Any:
    """Build the validation-time adversarial attack."""
    model.eval()
    atk_cfg = dict(cfg["attack"]["val"])
    atk_cfg["dataset_name"] = cfg["dataset"]["name"]
    return _build_attack_from_cfg(atk_cfg, model)


def build_eval_attacks(cfg: Dict[str, Any], model: "nn.Module") -> Dict[str, Any]:
    """Build evaluation-time adversarial attacks.

    Returns:
        Mapping from attack name to attack object.

    Notes:
        Threat model: Linf.
        Normalization: Attack expects inputs already normalized with dataset-specific mean/std.
    """
    model.eval()
    eval_cfg = cfg["attack"]["eval"]
    pgd20_cfg = eval_cfg.get("pgd20", {})
    attacks: Dict[str, Any] = {}
    base = {"dataset_name": cfg["dataset"]["name"]}
    attacks["pgd20"] = build_pgd_attack(
        {
            **base,
            "eps": pgd20_cfg.get("eps", eval_cfg.get("eps", cfg["attack"]["val"]["eps"])),
            "step_size": pgd20_cfg.get(
                "step_size",
                pgd20_cfg.get("alpha", eval_cfg.get("step_size", cfg["attack"]["val"].get("step_size", 0.007843))),
            ),
            "num_steps": int(pgd20_cfg.get("num_steps", pgd20_cfg.get("steps", 20))),
            "restarts": int(pgd20_cfg.get("restarts", 5)),
        },
        model,
    )
    return attacks
