"""Shared utilities for configuring TorchAttacks with dataset normalization."""

from __future__ import annotations

from typing import Any

from ardg.data.transforms import get_dataset_stats


def require_torchattacks() -> Any:
    """Import torchattacks with a clear error if missing."""
    try:
        import torchattacks  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "torchattacks is required for this attack. Install with `pip install torchattacks`."
        ) from exc
    return torchattacks


def configure_normalization(attack: Any, dataset_name: str) -> Any:
    """Configure TorchAttacks to treat eps/alpha in pixel space.

    The training pipeline feeds normalized inputs to attacks. TorchAttacks can
    internally inverse-normalize to [0,1], run perturbation with pixel-space
    budgets (e.g., eps=8/255), then normalize back before returning.
    """
    mean, std = get_dataset_stats(dataset_name)
    attack.set_normalization_used(mean, std)
    return attack
