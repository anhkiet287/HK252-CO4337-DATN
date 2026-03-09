"""Attack factory for TorchAttacks suites."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, TYPE_CHECKING

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


def _coerce_suite_specs(raw_suite: Any) -> List[Dict[str, Any]]:
    if raw_suite is None:
        return []
    if isinstance(raw_suite, list):
        return [dict(x) for x in raw_suite]
    if not isinstance(raw_suite, dict):
        raise ValueError("Attack suite must be a list or a mapping with 'attacks'.")

    attacks = raw_suite.get("attacks")
    if attacks is None:
        # Support single attack as mapping shorthand.
        if "type" in raw_suite or "name" in raw_suite:
            return [dict(raw_suite)]
        return []
    if not isinstance(attacks, list):
        raise ValueError("Attack suite 'attacks' must be a list.")
    return [dict(x) for x in attacks]


def _build_suite_from_specs(
    specs: Iterable[Dict[str, Any]],
    model: "nn.Module",
    dataset_name: str,
    *,
    shared_norm: Optional[str] = None,
    shared_eps: Optional[float] = None,
    default_label_prefix: str = "attack",
) -> Dict[str, Any]:
    suite: Dict[str, Any] = {}
    for idx, raw in enumerate(specs, start=1):
        spec = dict(raw)
        label = str(spec.pop("label", spec.get("name", f"{default_label_prefix}_{idx}"))).strip()
        if not label:
            label = f"{default_label_prefix}_{idx}"
        if label in suite:
            raise ValueError(f"Duplicate attack label in suite: {label!r}")
        suite[label] = build_attack(
            spec,
            model,
            dataset_name=dataset_name,
            shared_norm=shared_norm,
            shared_eps=shared_eps,
        )
    return suite


def _legacy_eval_specs(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    attack_cfg = cfg.get("attack", {})
    eval_cfg = attack_cfg.get("eval", {})
    val_cfg = attack_cfg.get("val", {})

    specs: List[Dict[str, Any]] = []
    pgd20_cfg = eval_cfg.get("pgd20", {}) if isinstance(eval_cfg, dict) else {}
    if isinstance(eval_cfg, dict):
        if isinstance(eval_cfg.get("attacks"), list):
            specs.extend([dict(x) for x in eval_cfg["attacks"]])
        elif eval_cfg.get("enabled", True):
            eps = pgd20_cfg.get("eps", eval_cfg.get("eps", val_cfg.get("eps", 8.0 / 255.0)))
            step_size = pgd20_cfg.get(
                "step_size",
                pgd20_cfg.get("alpha", eval_cfg.get("step_size", val_cfg.get("step_size", 0.007843))),
            )
            steps = int(pgd20_cfg.get("num_steps", pgd20_cfg.get("steps", 20)))
            restarts = int(pgd20_cfg.get("restarts", 5))
            specs.append(
                {
                    "label": "pgd20",
                    "type": "pgd",
                    "eps": eps,
                    "step_size": step_size,
                    "num_steps": steps,
                    "restarts": restarts,
                }
            )

    autoattack_cfg = attack_cfg.get("autoattack", {})
    if isinstance(autoattack_cfg, dict) and bool(autoattack_cfg.get("enabled", False)):
        has_auto = any(str(s.get("type", s.get("name", "")).lower()) == "autoattack" for s in specs)
        if not has_auto:
            specs.append(
                {
                    "label": "autoattack",
                    "type": "autoattack",
                    "norm": str(autoattack_cfg.get("norm", "Linf")),
                    "eps": float(autoattack_cfg.get("eps", val_cfg.get("eps", 8.0 / 255.0))),
                    "version": str(autoattack_cfg.get("version", "standard")),
                    "n_classes": int(cfg.get("model", {}).get("num_classes", 10)),
                    "verbose": bool(autoattack_cfg.get("verbose", False)),
                }
            )
    return specs


def build_train_suite(cfg: Dict[str, Any], model: "nn.Module") -> Dict[str, Any]:
    """Build train-time attack suite from attack.train_domains/train_suite/train."""
    model.eval()
    attack_cfg = cfg.get("attack", {})
    dataset_name = str(cfg["dataset"]["name"])

    # New preferred config: attack.train_domains
    train_domains = attack_cfg.get("train_domains")
    if isinstance(train_domains, list) and train_domains:
        specs = _coerce_suite_specs(train_domains)
        return _build_suite_from_specs(specs, model, dataset_name, default_label_prefix="train")

    if "train_suite" in attack_cfg:
        train_suite_cfg = attack_cfg.get("train_suite")
        if isinstance(train_suite_cfg, dict) and not bool(train_suite_cfg.get("enabled", True)):
            return {}
        specs = _coerce_suite_specs(train_suite_cfg)
        return _build_suite_from_specs(specs, model, dataset_name, default_label_prefix="train")

    train_spec = attack_cfg.get("train")
    if not isinstance(train_spec, dict):
        return {}
    label = str(train_spec.get("label", train_spec.get("name", "train_attack")))
    return {
        label: build_attack(dict(train_spec), model, dataset_name=dataset_name),
    }


def build_train_attack(cfg: Dict[str, Any], model: "nn.Module") -> Any:
    """Build one training-time adversarial attack.

    This keeps backward compatibility for objectives that expect a single attack.
    """
    suite = build_train_suite(cfg, model)
    if not suite:
        raise ValueError("No train attack configured. Set attack.train_domains, attack.train, or attack.train_suite.")
    if len(suite) != 1:
        keys = list(suite.keys())
        raise ValueError(
            "build_train_attack expects exactly one train attack, "
            f"but got {len(keys)} in attack.train_suite: {keys}."
        )
    return next(iter(suite.values()))


def build_val_attack(cfg: Dict[str, Any], model: "nn.Module") -> Any:
    """Build one validation-time adversarial attack."""
    model.eval()
    attack_cfg = cfg.get("attack", {})
    dataset_name = str(cfg["dataset"]["name"])

    if "val_suite" in attack_cfg:
        val_suite_cfg = attack_cfg.get("val_suite")
        if isinstance(val_suite_cfg, dict) and not bool(val_suite_cfg.get("enabled", True)):
            raise ValueError("attack.val_suite is disabled and no attack.val fallback is provided.")
        specs = _coerce_suite_specs(val_suite_cfg)
        suite = _build_suite_from_specs(specs, model, dataset_name, default_label_prefix="val")
        if len(suite) != 1:
            keys = list(suite.keys())
            raise ValueError(
                "build_val_attack expects exactly one val attack, "
                f"but got {len(keys)} in attack.val_suite: {keys}."
            )
        return next(iter(suite.values()))

    val_spec = attack_cfg.get("val")
    if not isinstance(val_spec, dict):
        raise ValueError("No val attack configured. Set attack.val or attack.val_suite.")
    return build_attack(dict(val_spec), model, dataset_name=dataset_name)


def build_eval_suite(cfg: Dict[str, Any], model: "nn.Module") -> Dict[str, Any]:
    """Build evaluation-time attack suite from attack.eval_suite (preferred).

    Backward compatibility:
    - if attack.eval_suite is missing, fall back to legacy attack.eval + attack.autoattack.
    """
    model.eval()
    attack_cfg = cfg.get("attack", {})
    dataset_name = str(cfg["dataset"]["name"])

    eval_suite_cfg = attack_cfg.get("eval_suite")
    if eval_suite_cfg is not None:
        if isinstance(eval_suite_cfg, dict) and not bool(eval_suite_cfg.get("enabled", True)):
            return {}
        specs = _coerce_suite_specs(eval_suite_cfg)
    else:
        specs = _legacy_eval_specs(cfg)

    return _build_suite_from_specs(specs, model, dataset_name, default_label_prefix="eval")


def build_eval_attacks(cfg: Dict[str, Any], model: "nn.Module") -> Dict[str, Any]:
    """Backward-compatible alias for evaluation attack suite builder."""
    return build_eval_suite(cfg, model)
