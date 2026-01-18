"""Training loop for ERM and PGD-AT."""

from typing import Any, Dict, Optional


class Trainer:
    """Trainer for ERM and PGD-AT training."""

    def __init__(
        self,
        cfg: Dict[str, Any],
        model: Any,
        train_loader: Any,
        val_loader: Any,
        device: Optional[str] = None,
    ) -> None:
        """Initialize the trainer.

        Args:
            cfg: Configuration dictionary.
            model: Model to train.
            train_loader: Training dataloader yielding (images, labels).
            val_loader: Validation dataloader yielding (images, labels).
            device: Device string (e.g., "cuda", "cpu").
        """
        self.cfg = cfg
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device or "cpu"

    def train(self) -> None:
        """Run the full training loop.

        Side effects:
            Writes checkpoints and logs metrics to outputs.
        """
        raise NotImplementedError("TODO: implement training loop")

    def train_one_epoch(self, epoch: int) -> Dict[str, float]:
        """Run one training epoch.

        Args:
            epoch: Epoch index.

        Returns:
            Dictionary of training metrics (e.g., loss, acc, lr).
        """
        raise NotImplementedError("TODO: implement train_one_epoch")

    def validate(self, epoch: int) -> Dict[str, float]:
        """Run validation for one epoch.

        Args:
            epoch: Epoch index.

        Returns:
            Dictionary of validation metrics (e.g., loss, acc).
        """
        raise NotImplementedError("TODO: implement validate")

    def _train_step(self, batch: Any) -> Dict[str, float]:
        """Run a single training step.

        Args:
            batch: Tuple (images, labels) where images have shape (B, 3, 32, 32)
                and labels have shape (B,).

        Returns:
            Dictionary with keys such as loss, acc, and lr.
        """
        raise NotImplementedError("TODO: implement _train_step")

    def _make_adv_batch(self, images: Any, labels: Any) -> Any:
        """Generate adversarial examples for PGD-AT.

        Args:
            images: Clean images of shape (B, 3, 32, 32).
            labels: Labels of shape (B,).

        Returns:
            Adversarial images of shape (B, 3, 32, 32), clipped to valid range.
        """
        raise NotImplementedError("TODO: implement _make_adv_batch")

    def _save_checkpoint(self, name: str, metrics: Dict[str, float]) -> str:
        """Save a model checkpoint.

        Args:
            name: Checkpoint name.
            metrics: Metrics to serialize alongside checkpoint.

        Returns:
            Path to the written checkpoint file.

        Side effects:
            Writes a checkpoint file to disk.
        """
        raise NotImplementedError("TODO: implement _save_checkpoint")
