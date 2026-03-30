"""GroupDRO over fixed adversarial groups with online and batch updates."""

from __future__ import annotations

import math
import re
from typing import Any, Dict, Tuple

import torch

from ardg.attacks.attack_suite import build_attack
from ardg.training.objectives.base import Objective


class GroupDRO(Objective):
    """GroupDRO with either online or batch-wise q updates."""

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
        groupdro_cfg = cfg.get("train", {}).get("groupdro", {})
        self.eta_q = float(groupdro_cfg.get("eta_q", 0.02))
        self.update_mode = str(groupdro_cfg.get("update_mode", "online")).strip().lower()
        if self.update_mode not in {"online", "batch"}:
            raise ValueError(
                f"Unsupported train.groupdro.update_mode={self.update_mode!r}. Use 'online' or 'batch'."
            )

        dataset_name = str(cfg["dataset"]["name"])
        self.group_names: list[str] = []
        self.group_metric_names: list[str] = []
        self.attacks: list[Any] = []
        metric_name_counts: dict[str, int] = {}
        # Paper-native GroupDRO assumes a fixed set of groups known ahead of time.
        # Here each configured train domain becomes one group and owns one attack.
        for group_idx, spec in enumerate(train_domains):
            if not isinstance(spec, dict):
                raise ValueError(f"attack.train_domains[{group_idx}] must be a mapping.")
            name = str(spec.get("label", spec.get("name", spec.get("type", f"group_{group_idx}")))).strip()
            resolved_name = name or f"group_{group_idx}"
            self.group_names.append(resolved_name)
            metric_name = self._sanitize_group_metric_name(resolved_name)
            suffix = metric_name_counts.get(metric_name, 0)
            metric_name_counts[metric_name] = suffix + 1
            if suffix:
                metric_name = f"{metric_name}_{suffix}"
            self.group_metric_names.append(metric_name)
            self.attacks.append(build_attack(dict(spec), model, dataset_name=dataset_name))

        self.num_groups = len(self.attacks)
        # q starts uniform over the fixed groups, exactly as in the standard update rule.
        self.q = torch.full(
            (self.num_groups,),
            1.0 / float(self.num_groups),
            dtype=torch.float32,
        )
        logging_cfg = groupdro_cfg.get("logging", {})
        self.plot_q_trajectory = bool(logging_cfg.get("plot_q_trajectory", True))
        self.plot_every_epochs = max(1, int(logging_cfg.get("plot_every_epochs", 1)))
        self.max_plot_points = max(1, int(logging_cfg.get("max_plot_points", 2000)))
        self.q_history: list[list[float]] = []
        self.q_history_steps: list[int] = []
        self._q_eps = 1e-12

    def compute_loss(self, model: Any, batch: Any) -> Tuple[torch.Tensor, Dict[str, Any]]:
        images, labels, group_ids = self._extract_group_batch(batch)
        unique_groups = torch.unique(group_ids)
        batch_size = int(labels.size(0))
        group_list = [int(group_id) for group_id in unique_groups.detach().cpu().tolist()]
        for group_id in group_list:
            if group_id < 0 or group_id >= self.num_groups:
                raise ValueError(
                    f"GroupDRO received group_id={group_id}, "
                    f"but configured groups are in [0, {self.num_groups - 1}]."
                )

        if self.update_mode == "online":
            # Native GroupDRO updates one q[g] per optimizer step, so a mixed batch would
            # no longer match the intended algorithm.
            if int(unique_groups.numel()) != 1:
                raise ValueError(
                    "GroupDRO online mode expects each optimizer step to receive exactly one group, "
                    f"but got multiple group_ids={unique_groups.detach().cpu().tolist()}."
                )

            group_id = int(unique_groups.item())
            attacked = self._apply_group_attack(model, images, labels, group_id)
            with self.autocast_context():
                logits = model(attacked)
                # loss_group is the scalar mean loss for the current group batch.
                loss_group = torch.nn.functional.cross_entropy(logits, labels)

            q = self.q.to(device=labels.device, dtype=torch.float32)
            loss_update = loss_group.detach().to(dtype=torch.float32)
            if not torch.isfinite(loss_update):
                raise RuntimeError(
                    f"GroupDRO received non-finite loss_group for group '{self.group_names[group_id]}'."
                )
            # Update q in log-space so the multiplicative rule stays stable even when
            # a hard attack produces a very large but finite CE loss.
            log_q = torch.log(q.clamp_min(self._q_eps))
            log_q[group_id] = log_q[group_id] + (self.eta_q * loss_update)
            q = torch.softmax(log_q, dim=0)
            if not torch.isfinite(q).all():
                raise RuntimeError("GroupDRO q update produced non-finite weights.")
            self.q = q.detach()

            # Native GroupDRO uses the updated q[g] for the same step, but the weight is
            # detached so gradients only flow through loss_group into model parameters.
            weighted_loss = q[group_id].detach() * loss_group
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
        else:
            attacked = images.clone()
            for group_id in group_list:
                mask = group_ids == group_id
                attacked[mask] = self._apply_group_attack(model, images[mask], labels[mask], group_id)

            with self.autocast_context():
                logits = model(attacked)
                per_sample_losses = torch.nn.functional.cross_entropy(logits, labels, reduction="none")

            q = self.q.to(device=labels.device, dtype=torch.float32)
            group_loss_values: list[torch.Tensor] = []
            weighted_loss_terms: list[torch.Tensor] = []
            group_batch_counts: Dict[str, int] = {}
            log_q = torch.log(q.clamp_min(self._q_eps))
            for group_id in group_list:
                mask = group_ids == group_id
                loss_group = per_sample_losses[mask].mean()
                loss_update = loss_group.detach().to(dtype=torch.float32)
                if not torch.isfinite(loss_update):
                    raise RuntimeError(
                        f"GroupDRO received non-finite loss_group for group '{self.group_names[group_id]}'."
                    )
                log_q[group_id] = log_q[group_id] + (self.eta_q * loss_update)
                group_loss_values.append(loss_group)
                group_batch_counts[self.group_names[group_id]] = int(mask.sum().item())
            q = torch.softmax(log_q, dim=0)
            if not torch.isfinite(q).all():
                raise RuntimeError("GroupDRO q update produced non-finite weights.")
            self.q = q.detach()

            for group_id, loss_group in zip(group_list, group_loss_values):
                weighted_loss_terms.append(q[group_id].detach() * loss_group)
            weighted_loss = torch.stack(weighted_loss_terms).sum()
            correct = int((logits.argmax(dim=1) == labels).sum().item())
            mean_group_loss = torch.stack(group_loss_values).mean()

            metrics = {
                "loss": float(weighted_loss.item()),
                "loss_adv": float(weighted_loss.item()),
                "loss_group": float(mean_group_loss.item()),
                "acc": correct / max(batch_size, 1),
                "acc_adv": correct / max(batch_size, 1),
                "q_max": float(q.max().item()),
                "q_min": float(q.min().item()),
                "correct": correct,
                "batch_size": batch_size,
                "domain_name": "mixed",
                "group_batch_counts": group_batch_counts,
            }
        for idx, metric_name in enumerate(self.group_metric_names):
            metrics[f"q/{metric_name}"] = float(q[idx].item())
        return weighted_loss, metrics

    def record_q_snapshot(self, step: int) -> None:
        if not self.plot_q_trajectory:
            return
        snapshot = [float(value) for value in self.q.detach().cpu().tolist()]
        step_value = int(step)
        if self.q_history_steps and self.q_history_steps[-1] == step_value:
            self.q_history[-1] = snapshot
            return
        self.q_history_steps.append(step_value)
        self.q_history.append(snapshot)

    def should_log_q_trajectory(self, epoch: int, *, force: bool = False) -> bool:
        if not self.plot_q_trajectory or not self.q_history_steps:
            return False
        if force:
            return True
        return int(epoch) % self.plot_every_epochs == 0

    def build_q_trajectory_figure(self, max_points: int | None = None) -> Any | None:
        if not self.q_history_steps:
            return None
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return None

        limit = self.max_plot_points if max_points is None else max(1, int(max_points))
        indices = list(range(len(self.q_history_steps)))
        if limit and len(indices) > limit:
            stride = max(1, math.ceil(len(indices) / limit))
            indices = list(range(0, len(self.q_history_steps), stride))
            if indices[-1] != len(self.q_history_steps) - 1:
                indices.append(len(self.q_history_steps) - 1)

        plot_steps = [self.q_history_steps[idx] for idx in indices]
        fig, ax = plt.subplots(figsize=(9.0, 5.0))
        for group_idx, group_name in enumerate(self.group_names):
            plot_values = [self.q_history[idx][group_idx] for idx in indices]
            ax.plot(plot_steps, plot_values, label=f"g={group_idx}: {group_name}")
        title_prefix = "Online" if self.update_mode == "online" else "Batch"
        ax.set_title(f"{title_prefix} GroupDRO - q trajectory")
        ax.set_xlabel("Update step")
        ax.set_ylabel("q_g")
        ax.legend(loc="upper left")
        fig.tight_layout()
        return fig

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
        # Group ids must be provided explicitly by the GroupDRO loader.
        # We do not silently fall back to labels because labels are not groups.
        if isinstance(batch, dict):
            if "x" not in batch or "y" not in batch:
                raise ValueError("GroupDRO expects batch dicts with 'x' and 'y'.")
            if "group_id" not in batch:
                raise ValueError("GroupDRO requires batch group ids in batch['group_id'].")
            images = batch["x"]
            labels = batch["y"]
            group_ids = batch["group_id"]
        elif isinstance(batch, (list, tuple)) and len(batch) >= 3:
            images, labels, group_ids = batch[0], batch[1], batch[2]
        else:
            raise ValueError(
                "GroupDRO requires each batch to include group ids as a third tuple item "
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

    @staticmethod
    def _sanitize_group_metric_name(name: str) -> str:
        sanitized = re.sub(r"[^0-9A-Za-z_]+", "_", str(name).strip()).strip("_").lower()
        return sanitized or "group"
