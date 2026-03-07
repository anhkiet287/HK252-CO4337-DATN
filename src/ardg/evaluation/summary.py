"""Suite summarization helpers (flatten per-domain metrics)."""

from __future__ import annotations

from typing import Dict, Mapping


def summarize_suite(per_domain: Mapping[str, Mapping[str, float]], prefix: str) -> Dict[str, float]:
    """Flatten per-domain metrics and compute aggregate robustness stats.

    Args:
        per_domain: mapping ``{domain_name: metrics_dict}`` where metrics include
            clean domains with ``acc_clean/loss_clean`` and attack domains with
            ``acc_adv/loss_adv`` (legacy ``acc/loss`` also accepted).
        prefix: string prefix for output keys (e.g., ``"val"`` or ``"test"``).

    Returns:
        Flat dict with keys like ``"val/acc_clean"``, ``"val/acc_pgd20"``,
        ``"val/acc_avg"``, ``"val/acc_worst"``, ``"val/worst_domain"``.
    """

    flat: Dict[str, float] = {}
    adv_accs: Dict[str, float] = {}

    for name, metrics in per_domain.items():
        if not isinstance(metrics, Mapping):
            continue

        if name == "clean":
            acc = float(metrics.get("acc_clean", metrics.get("acc", 0.0)))
            loss = float(metrics.get("loss_clean", metrics.get("loss", 0.0)))
            flat[f"{prefix}/acc_clean"] = acc
            flat[f"{prefix}/loss_clean"] = loss
            continue

        # adversarial / attack domain
        acc = float(metrics.get("acc_adv", metrics.get("acc", 0.0)))
        loss = float(metrics.get("loss_adv", metrics.get("loss", 0.0)))
        flat[f"{prefix}/acc_{name}"] = acc
        flat[f"{prefix}/loss_{name}"] = loss
        adv_accs[name] = acc

    if adv_accs:
        values = list(adv_accs.values())
        flat[f"{prefix}/acc_avg"] = sum(values) / len(values)
        worst_name = min(adv_accs, key=adv_accs.get)
        flat[f"{prefix}/acc_worst"] = adv_accs[worst_name]
        flat[f"{prefix}/worst_domain"] = worst_name

    return flat
