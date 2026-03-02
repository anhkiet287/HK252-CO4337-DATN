"""AutoAttack evaluation runner."""

from __future__ import annotations

import time
from typing import Any, Dict

import torch


def _require_autoattack() -> Any:
    try:
        from autoattack import AutoAttack  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "attack.autoattack.enabled=true but AutoAttack is not installed. "
            "Install with `pip install autoattack` (or "
            "`pip install git+https://github.com/fra31/auto-attack.git`) "
            "or disable via `attack.autoattack.enabled: false`."
        ) from exc
    return AutoAttack


def run_autoattack(
    model: Any,
    loader: Any,
    eps: float,
    device: str,
    *,
    norm: str = "Linf",
    version: str = "standard",
    max_batches: int | None = None,
) -> Dict[str, float]:
    """Run AutoAttack for evaluation.

    Args:
        model: Classification model.
        loader: Dataloader yielding (images, labels).
        eps: Attack epsilon value.
        device: Device string (e.g., "cuda", "cpu").
        norm: Threat norm. Default is Linf.
        version: AutoAttack version. Default is standard.
        max_batches: Optional cap for quick runs.

    Returns:
        Dictionary with robust accuracy and runtime stats.
    """
    AutoAttack = _require_autoattack()
    model.eval()
    device_t = torch.device(device)
    eps_f = float(eps)
    bs_default = int(getattr(loader, "batch_size", 128) or 128)

    adversary = AutoAttack(
        model,
        norm=str(norm),
        eps=eps_f,
        version=str(version),
        device=str(device_t),
    )

    total_correct = 0
    total_seen = 0
    start = time.perf_counter()
    for batch_idx, (images, labels) in enumerate(loader, start=1):
        images = images.to(device_t)
        labels = labels.to(device_t)
        bs = min(bs_default, int(images.size(0)))
        x_adv = adversary.run_standard_evaluation(images, labels, bs=bs)
        with torch.no_grad():
            logits = model(x_adv)
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_seen += int(images.size(0))
        if max_batches and batch_idx >= int(max_batches):
            break

    runtime = time.perf_counter() - start
    return {
        "acc": total_correct / max(total_seen, 1),
        "n_samples": total_seen,
        "eps": eps_f,
        "aa_runtime_sec": runtime,
        "aa_n_samples": total_seen,
        "aa_norm": str(norm),
        "aa_version": str(version),
    }
