"""Scaffold objective for a user-defined training protocol."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch

from ardg.attacks.attack_suite import build_train_attack
from ardg.training.losses import compute_loss
from ardg.training.objectives.base import Objective
from ardg.utils.batch import unpack_xy


class CustomProtocol(Objective):
    """Runnable scaffold for implementing a new training protocol."""

    def __init__(self, cfg: Dict[str, Any], model: Any) -> None:
        self.cfg = cfg
        self.model = model
        proto_cfg = cfg.get("train", {}).get("custom_protocol", {})
        self.weight_main = float(proto_cfg.get("weight_main", 1.0))
        self.weight_aux = float(proto_cfg.get("weight_aux", 0.0))
        self.use_adv_inputs = bool(
            proto_cfg.get(
                "use_adv_inputs",
                cfg.get("train", {}).get("adv_training", False),
            )
        )

        # TODO: add protocol-specific buffers, queues, moving averages, or group state here.
        self.protocol_step = 0
        self.attack = None
        if self.use_adv_inputs:
            # TODO: replace or extend this builder if your protocol uses a custom train-time attack.
            self.attack = build_train_attack(cfg, model)

    def _prepare_inputs(
        self,
        model: Any,
        batch: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, bool]:
        images, labels = unpack_xy(batch)
        used_adv = False
        if self.attack is not None:
            used_adv = True
            was_training = bool(model.training)
            model.eval()
            with self.full_precision_context():
                images = self.attack(images, labels).detach()
            model.train(was_training)
        return images, labels, used_adv

    def compute_loss(self, model: Any, batch: Any) -> Tuple[torch.Tensor, Dict[str, float]]:
        images, labels, used_adv = self._prepare_inputs(model, batch)

        with self.autocast_context():
            logits = model(images)
            base_loss = compute_loss(logits, labels)

            # TODO: implement your protocol-specific objective terms here.
            # The current scaffold keeps training runnable by falling back to plain ERM.
            aux_loss = torch.zeros((), device=logits.device, dtype=base_loss.dtype)
            total_loss = (self.weight_main * base_loss) + (self.weight_aux * aux_loss)

        self.protocol_step += 1
        correct = (logits.argmax(dim=1) == labels).sum().item()
        metrics: Dict[str, float] = {
            "loss": float(total_loss.item()),
            "acc": correct / max(labels.size(0), 1),
            ("loss_adv" if used_adv else "loss_clean"): float(total_loss.item()),
            ("acc_adv" if used_adv else "acc_clean"): correct / max(labels.size(0), 1),
            "loss_base": float(base_loss.item()),
            "loss_aux": float(aux_loss.item()),
            "correct": correct,
            "batch_size": labels.size(0),
            "protocol_step": float(self.protocol_step),
            "weight_main": float(self.weight_main),
            "weight_aux": float(self.weight_aux),
        }
        return total_loss, metrics

    def validate(self, model: Any, loader: Any) -> Dict[str, float]:
        del model, loader
        # TODO: return extra validation metrics if your checkpoint selection needs them.
        return {}

    def on_train_start(self, loaders: Dict[str, Any]) -> None:
        del loaders
        # TODO: initialize protocol state before the first epoch if needed.
        return None

    def on_epoch_end(self, epoch: int, loaders: Dict[str, Any]) -> None:
        del epoch, loaders
        # TODO: update protocol state after each epoch if needed.
        return None

    def state_dict(self) -> Dict[str, Any]:
        # TODO: persist any protocol-specific tensors/state needed for resume.
        return {
            "protocol_step": int(self.protocol_step),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        # TODO: restore any protocol-specific tensors/state needed for resume.
        if state.get("protocol_step") is not None:
            self.protocol_step = int(state["protocol_step"])
