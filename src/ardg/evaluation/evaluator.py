"""Evaluation helpers for clean and adversarial metrics."""

from typing import Any, Dict


def evaluate_clean(model: Any, loader: Any, device: str) -> Dict[str, float]:
    """Evaluate a classifier on clean inputs.

    Args:
        model: Classification model. Expects inputs of shape (B, 3, 32, 32).
        loader: Dataloader yielding (images, labels) where images have shape
            (B, 3, 32, 32) and labels have shape (B,).
        device: Device string (e.g., "cuda", "cpu").

    Returns:
        Dictionary with metrics. Must include acc and n_samples. May include loss.
    """
    raise NotImplementedError("TODO: implement evaluate_clean")


def evaluate_under_attack(model: Any, loader: Any, attack: Any, device: str) -> Dict[str, float]:
    """Evaluate a classifier under a given adversarial attack.

    Args:
        model: Classification model. Expects inputs of shape (B, 3, 32, 32).
        loader: Dataloader yielding (images, labels) where images have shape
            (B, 3, 32, 32) and labels have shape (B,).
        attack: Attack object that generates adversarial images given (images, labels).
        device: Device string (e.g., "cuda", "cpu").

    Returns:
        Dictionary with evaluation metrics. Must include acc and n_samples.
    """
    raise NotImplementedError("TODO: implement evaluate_under_attack")


def evaluate_suite(model: Any, loader: Any, attacks: Dict[str, Any], device: str) -> Dict[str, Dict[str, float]]:
    """Evaluate a classifier across an attack suite.

    Args:
        model: Classification model. Expects inputs of shape (B, 3, 32, 32).
        loader: Dataloader yielding (images, labels) where images have shape
            (B, 3, 32, 32) and labels have shape (B,).
        attacks: Mapping of attack name to attack object.
        device: Device string (e.g., "cuda", "cpu").

    Returns:
        Mapping from attack name to metrics dictionary.
    """
    raise NotImplementedError("TODO: implement evaluate_suite")
