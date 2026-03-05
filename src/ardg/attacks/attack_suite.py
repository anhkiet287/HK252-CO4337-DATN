"""Attack factory for TorchAttacks suites."""

from typing import Any, Dict, Optional, TYPE_CHECKING

from ardg.attacks.autoattack_ta import build_autoattack_attack
from ardg.attacks.cw import build_cw_attack
from ardg.attacks.deepfool import build_deepfool_attack
from ardg.attacks.fab import build_fab_attack
from ardg.attacks.fgsm import build_fgsm_attack
from ardg.attacks.pgd import build_pgd_attack
from ardg.attacks.square import build_square_attack

if TYPE_CHECKING:
    from torch import nn


def _float_close(a: float, b: float, tol: float = 1e-12) -> bool:
    return abs(float(a) - float(b)) <= tol


class _IdentityAttack:
    """Return clean inputs unchanged."""

    def __call__(self, images: Any, labels: Any) -> Any:  # noqa: ARG002
        return images


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
    if name == "autoattack":
        return build_autoattack_attack(atk_cfg, model)
    raise ValueError(f"Unsupported attack name: {name}")


def build_attack(
    spec: Dict[str, Any],
    model: "nn.Module",
    dataset_name: str,
    shared_norm: Optional[str] = None,
    shared_eps: Optional[float] = None,
) -> Any:
    """Build an attack from one spec, optionally enforcing shared threat model."""
    atk_cfg = dict(spec)
    atk_type = str(atk_cfg.get("type", atk_cfg.get("name", "pgd"))).lower()
    atk_cfg["dataset_name"] = dataset_name

    if atk_type == "clean":
        return _IdentityAttack()

    if shared_eps is not None:
        eps_in = atk_cfg.get("eps")
        if eps_in is not None and not _float_close(float(eps_in), float(shared_eps)):
            raise ValueError(
                f"Attack spec eps={eps_in} mismatches shared eps={shared_eps}."
            )
        atk_cfg["eps"] = float(shared_eps)

    if shared_norm is not None and atk_type in {
        "pgd",
        "pgd_linf",
        "pgd_ce",
        "pgd_dlr",
        "fgsm",
        "fgsm_rs",
        "fab",
        "square",
        "autoattack",
    }:
        norm_in = atk_cfg.get("norm")
        if norm_in is not None and str(norm_in).lower() != str(shared_norm).lower():
            raise ValueError(
                f"Attack spec norm={norm_in} mismatches shared norm={shared_norm}."
            )
        atk_cfg["norm"] = shared_norm

    if atk_type == "fgsm_rs":
        atk_cfg["name"] = "fgsm"
        atk_cfg["random_start"] = True
    elif atk_type in {"pgd_ce", "pgd_dlr"}:
        atk_cfg["name"] = "pgd"
        atk_cfg["loss"] = "ce" if atk_type == "pgd_ce" else "dlr"
    else:
        atk_cfg["name"] = atk_type

    return _build_attack_from_cfg(atk_cfg, model)


def build_train_attack(cfg: Dict[str, Any], model: "nn.Module") -> Any:
    """Build the training-time adversarial attack."""
    model.eval()
    atk_cfg = dict(cfg["attack"]["train"])
    return build_attack(atk_cfg, model, dataset_name=cfg["dataset"]["name"])


def build_val_attack(cfg: Dict[str, Any], model: "nn.Module") -> Any:
    """Build the validation-time adversarial attack."""
    model.eval()
    atk_cfg = dict(cfg["attack"]["val"])
    return build_attack(atk_cfg, model, dataset_name=cfg["dataset"]["name"])


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
