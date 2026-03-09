"""Base interface for training objectives (ERM, PGD, GroupDRO, etc.)."""

from __future__ import annotations

from typing import Any, Dict, Tuple


class Objective:
    """Abstract objective used by Trainer."""

    def preprocess_batch(self, batch: Any, model: Any) -> Any:
        """Optionally modify batch before forward (e.g., PGD adversarial)."""
        return batch

    def compute_loss(self, model: Any, batch: Any) -> Tuple[Any, Dict[str, Any]]:
        """Compute loss and metrics for a batch."""
        raise NotImplementedError

    # Backward-compatible alias
    def loss(self, model: Any, batch: Any) -> Tuple[Any, Dict[str, Any]]:
        return self.compute_loss(model, batch)

    def eval_metrics(self, logits: Any, batch: Any) -> Dict[str, float]:
        """Optional evaluation-time metrics (acc by group, etc.)."""
        return {}

    def validate(self, model: Any, loader: Any) -> Dict[str, float]:
        """Optional validation hook that can add extra metrics."""
        return {}

    def on_epoch_end(self, epoch: int, loaders: Dict[str, Any]) -> None:
        """Optional hook run at epoch end (e.g., recluster for GroupDRO++)."""
        return None

    def state_dict(self) -> Dict[str, Any]:
        """Return objective-specific state for checkpointing."""
        return {}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Load objective-specific state."""
        return None
