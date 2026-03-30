"""Paper-faithful GroupDRO over fixed adversarial groups."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch

from ardg.attacks.attack_suite import build_attack
from ardg.training.objectives.base import Objective


class GroupDRO(Objective):
    """Native GroupDRO with one homogeneous group per optimizer step."""

    def __init__(self, cfg: Dict[str, Any], model: Any | None = None) -> None:
        if model is None:
            raise ValueError("GroupDRO requires a model.")

        train_domains = cfg.get("attack", {}).get("train_domains")
        if not isinstance(train_domains, list) or not train_domains:
            raise ValueError(
                "GroupDRO requires attack.train_domains to define the fixed groups."
            )

        self.cfg = cfg
        self.model = model
        self.eta_q = float(cfg.get("train", {}).get("groupdro", {}).get("eta_q", 0.02))

        dataset_name = str(cfg["dataset"]["name"])
        self.group_names: list[str] = []
        self.attacks: list[Any] = []
        # Paper-native GroupDRO assumes a fixed set of groups known ahead of time.
        # Here each configured train domain becomes one group and owns one attack.
        for group_idx, spec in enumerate(train_domains):
            if not isinstance(spec, dict):
                raise ValueError(f"attack.train_domains[{group_idx}] must be a mapping.")
            name = str(spec.get("label", spec.get("name", spec.get("type", f"group_{group_idx}")))).strip()
            self.group_names.append(name or f"group_{group_idx}")
            self.attacks.append(build_attack(dict(spec), model, dataset_name=dataset_name))

        self.num_groups = len(self.attacks)
        # q starts uniform over the fixed groups, exactly as in the standard update rule.
        self.q = torch.full(
            (self.num_groups,),
            1.0 / float(self.num_groups),
            dtype=torch.float32,
        )

    def compute_loss(self, model: Any, batch: Any) -> Tuple[torch.Tensor, Dict[str, Any]]:
        images, labels, group_ids = self._extract_group_batch(batch)
        unique_groups = torch.unique(group_ids)
        # Native GroupDRO updates one q[g] per optimizer step, so a mixed batch would
        # no longer match the intended algorithm.
        if int(unique_groups.numel()) != 1:
            raise ValueError(
                "GroupDRO native mode expects each optimizer step to receive exactly one group, "
                f"but got multiple group_ids={unique_groups.detach().cpu().tolist()}."
            )

        group_id = int(unique_groups.item())
        if group_id < 0 or group_id >= self.num_groups:
            raise ValueError(
                f"GroupDRO native mode received group_id={group_id}, "
                f"but configured groups are in [0, {self.num_groups - 1}]."
            )

        attacked = self._apply_group_attack(model, images, labels, group_id)
        with self.autocast_context():
            logits = model(attacked)
            # loss_group is the scalar mean loss for the current group batch.
            loss_group = torch.nn.functional.cross_entropy(logits, labels)

        q = self.q.to(device=labels.device, dtype=torch.float32)
        # Update q with the detached group loss before weighting theta's loss.
        # The detach matters: q is an online state variable, not a learnable tensor.
        q[group_id] = q[group_id] * torch.exp(self.eta_q * loss_group.detach().to(dtype=torch.float32))
        q_sum = q.sum()
        if not torch.isfinite(q_sum) or float(q_sum.item()) <= 0.0:
            raise RuntimeError("GroupDRO q update produced a non-finite or non-positive normalizer.")
        # Keep q on the simplex after every multiplicative update.
        q = q / q_sum
        self.q = q.detach()

        # Native GroupDRO uses the updated q[g] for the same step, but the weight is
        # detached so gradients only flow through loss_group into model parameters.
        weighted_loss = q[group_id].detach() * loss_group
        batch_size = int(labels.size(0))
        correct = int((logits.argmax(dim=1) == labels).sum().item())
        group_name = self.group_names[group_id]

        metrics: Dict[str, Any] = {
            "loss": float(weighted_loss.item()),
            "loss_adv": float(weighted_loss.item()),
            "loss_group": float(loss_group.item()),
            "acc": correct / max(batch_size, 1),
            "acc_adv": correct / max(batch_size, 1),
            "group_id": group_id,
            "q_g": float(q[group_id].item()),
            "q_max": float(q.max().item()),
            "q_min": float(q.min().item()),
            "correct": correct,
            "batch_size": batch_size,
            "domain_name": group_name,
            "group_batch_counts": {group_name: batch_size},
        }
        return weighted_loss, metrics

    def state_dict(self) -> Dict[str, Any]:
        # Checkpoints only need q to resume the paper update exactly.
        return {"q": self.q.detach().cpu()}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        q = state.get("q")
        if q is None:
            return
        q_tensor = q.detach().clone().to(dtype=torch.float32)
        if int(q_tensor.numel()) != self.num_groups:
            raise ValueError(
                f"Checkpoint q has {int(q_tensor.numel())} entries, but GroupDRO expects {self.num_groups} groups."
            )
        q_sum = q_tensor.sum()
        if not torch.isfinite(q_sum) or float(q_sum.item()) <= 0.0:
            raise ValueError("Checkpoint q must sum to a positive finite value.")
        self.q = q_tensor / q_sum

    def _extract_group_batch(self, batch: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Group ids must be provided explicitly by the native GroupDRO loader.
        # We do not silently fall back to labels because labels are not groups.
        if isinstance(batch, dict):
            if "x" not in batch or "y" not in batch:
                raise ValueError("GroupDRO native mode expects batch dicts with 'x' and 'y'.")
            if "group_id" not in batch:
                raise ValueError("GroupDRO native mode requires batch group ids in batch['group_id'].")
            images = batch["x"]
            labels = batch["y"]
            group_ids = batch["group_id"]
        elif isinstance(batch, (list, tuple)) and len(batch) >= 3:
            images, labels, group_ids = batch[0], batch[1], batch[2]
        else:
            raise ValueError(
                "GroupDRO native mode requires each batch to include group ids as a third tuple item "
                "or in batch['group_id']."
            )

        if not torch.is_tensor(group_ids):
            group_ids = torch.as_tensor(group_ids, device=labels.device)
        group_ids = group_ids.to(device=labels.device, dtype=torch.long).view(-1)
        return images, labels, group_ids

    def _apply_group_attack(
        self,
        model: Any,
        images: torch.Tensor,
        labels: torch.Tensor,
        group_id: int,
    ) -> torch.Tensor:
        attack = self.attacks[group_id]
        was_training = bool(model.training)
        # Attack generation runs with the model temporarily in eval mode for stable
        # adversarial examples, then restores the original training flag.
        model.eval()
        try:
            with self.full_precision_context():
                return attack(images, labels).detach()
        finally:
            model.train(was_training)
