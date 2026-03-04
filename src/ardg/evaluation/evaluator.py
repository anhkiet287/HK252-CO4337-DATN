"""Evaluation helpers and class-based evaluator."""

from __future__ import annotations

import itertools
from typing import Any, Dict, Iterable, Tuple

import torch

from ardg.training.losses import compute_loss


class Evaluator:
    """Reusable evaluator for clean and adversarial metrics.

    Args:
        model: Classification model.
        loader: Evaluation dataloader.
        device: Device string, e.g. ``"cuda"`` or ``"cpu"``.
        attack_suite: Optional mapping of ``{attack_name: attack_callable}``.
        max_batches: Optional cap on number of evaluation batches (0 = full loader).
    """

    def __init__(
        self,
        model: Any,
        loader: Any,
        device: str,
        attack_suite: Dict[str, Any] | None = None,
        max_batches: int = 0,
    ) -> None:
        self.model = model
        self.loader = loader
        self.device = torch.device(device)
        self.attack_suite: Dict[str, Any] = dict(attack_suite or {})
        self.max_batches = max(0, int(max_batches))

    def _iter_batches(self) -> Iterable[Any]:
        if self.max_batches <= 0:
            yield from self.loader
            return
        yield from itertools.islice(self.loader, self.max_batches)

    @staticmethod
    def _as_tuple_batch(batch: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(batch, dict):
            if "x" not in batch or "y" not in batch:
                raise ValueError("Batch dict must contain 'x' and 'y'.")
            return batch["x"], batch["y"]
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            return batch[0], batch[1]
        raise ValueError("Unsupported batch format; expected dict with x/y or (images, labels).")

    @torch.no_grad()
    def evaluate_clean(self) -> Dict[str, float]:
        """Evaluate accuracy/loss on clean inputs."""
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_seen = 0
        for batch in self._iter_batches():
            images, labels = self._as_tuple_batch(batch)
            images = images.to(self.device)
            labels = labels.to(self.device)
            logits = self.model(images)
            loss = compute_loss(logits, labels)
            total_loss += float(loss.item()) * int(images.size(0))
            total_correct += int((logits.argmax(dim=1) == labels).sum().item())
            total_seen += int(images.size(0))
        return {
            "loss": total_loss / max(total_seen, 1),
            "acc": total_correct / max(total_seen, 1),
            "n_samples": float(total_seen),
        }

    def evaluate_under_attack(self, attack: Any) -> Dict[str, float]:
        """Evaluate accuracy/loss under one attack callable."""
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_seen = 0
        for batch in self._iter_batches():
            images, labels = self._as_tuple_batch(batch)
            images = images.to(self.device)
            labels = labels.to(self.device)
            adv = attack(images, labels).detach()
            with torch.no_grad():
                logits = self.model(adv)
                loss = compute_loss(logits, labels)
            total_loss += float(loss.item()) * int(images.size(0))
            total_correct += int((logits.argmax(dim=1) == labels).sum().item())
            total_seen += int(images.size(0))
        return {
            "acc": total_correct / max(total_seen, 1),
            "loss": total_loss / max(total_seen, 1),
            "n_samples": float(total_seen),
        }

    def evaluate_suite(self, attack_suite: Dict[str, Any] | None = None) -> Dict[str, Dict[str, float]]:
        """Evaluate all attacks in a suite.

        Args:
            attack_suite: Optional attack mapping. If omitted, uses ``self.attack_suite``.
        """
        suite = attack_suite if attack_suite is not None else self.attack_suite
        return {name: self.evaluate_under_attack(atk) for name, atk in suite.items()}

    def evaluate_all(self, attack_suite: Dict[str, Any] | None = None) -> Dict[str, Any]:
        """Convenience method: clean + suite in one call."""
        suite_metrics = self.evaluate_suite(attack_suite)
        return {
            "clean": self.evaluate_clean(),
            "attacks": suite_metrics,
        }


def evaluate_clean(model: Any, loader: Any, device: str, max_batches: int = 0) -> Dict[str, float]:
    """Function wrapper for backward compatibility."""
    return Evaluator(model, loader, device=device, max_batches=max_batches).evaluate_clean()


def evaluate_under_attack(
    model: Any,
    loader: Any,
    attack: Any,
    device: str,
    max_batches: int = 0,
) -> Dict[str, float]:
    """Function wrapper for backward compatibility."""
    return Evaluator(model, loader, device=device, max_batches=max_batches).evaluate_under_attack(attack)


def evaluate_suite(
    model: Any,
    loader: Any,
    attacks: Dict[str, Any],
    device: str,
    max_batches: int = 0,
) -> Dict[str, Dict[str, float]]:
    """Function wrapper for backward compatibility."""
    return Evaluator(
        model,
        loader,
        device=device,
        attack_suite=attacks,
        max_batches=max_batches,
    ).evaluate_suite()
