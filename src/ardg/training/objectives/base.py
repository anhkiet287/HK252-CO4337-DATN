"""Base interface for training objectives (ERM, PGD, GroupDRO, etc.)."""

from __future__ import annotations

from typing import Any, Dict, Tuple


class Objective:
    """Abstract objective used by trainer1."""

    def preprocess_batch(self, batch: Any, model: Any) -> Any:
        """Optionally modify batch before forward (e.g., PGD adversarial)."""
        return batch

    def loss(self, model: Any, batch: Any) -> Tuple[Any, Dict[str, float]]:
        """Compute loss and metrics for a batch."""
        raise NotImplementedError

    def eval_metrics(self, logits: Any, batch: Any) -> Dict[str, float]:
        """Optional evaluation-time metrics (acc by group, etc.)."""
        return {}

    def on_epoch_end(self, epoch: int, loaders: Dict[str, Any]) -> None:
        """Optional hook run at epoch end (e.g., recluster for GroupDRO++)."""
        return None
