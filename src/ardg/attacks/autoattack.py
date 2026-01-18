"""AutoAttack evaluation runner."""

from typing import Any, Dict


def run_autoattack(model: Any, loader: Any, eps: float, device: str) -> Dict[str, float]:
    """Run AutoAttack for evaluation only.

    Args:
        model: Classification model. Expects inputs of shape (B, 3, 32, 32).
        loader: Dataloader yielding (images, labels) where images have shape
            (B, 3, 32, 32) and labels have shape (B,).
        eps: Attack epsilon value.
        device: Device string (e.g., "cuda", "cpu").

    Returns:
        Dictionary with robust accuracy and optional per-subattack metrics.

    Side effects:
        May take a long time, disables gradients, and sets model.eval().
    """
    raise NotImplementedError("TODO: implement run_autoattack")
