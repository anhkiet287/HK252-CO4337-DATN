"""GroupDRO++ draft with batch-wise or epoch-wise clustering modes."""

from __future__ import annotations

import logging
from typing import Any, Dict, Tuple

import torch

from ardg.training.cluster_utils import (
    assign_to_centers,
    compute_cluster_centers,
    project_embeddings_pca,
    run_kmeans_with_centers,
    save_cluster_snapshot,
)
from ardg.training.losses import compute_loss
from ardg.training.objectives.base import Objective
from ardg.attacks.attack_suite import build_train_attack
from ardg.utils.batch import move_to_device, unpack_xy
from ardg.utils.paths import get_run_dir


class GroupDROPlus(Objective):
    """GroupDRO with on-the-fly clustering (batch-wise or epoch-wise)."""

    _CLUSTER_MODE_ALIASES = {
        "batch": "batch",
        "per_batch": "batch",
        "epoch": "epoch",
        "per_epoch": "epoch",
    }

    def __init__(self, cfg: Dict[str, Any], model: Any) -> None:
        self.cfg = cfg
        self.model = model
        self.logger = logging.getLogger(__name__)
        gd_cfg = cfg.get("train", {}).get("groupdro_plus", {})
        self.num_clusters = int(gd_cfg.get("num_clusters", 4))
        self.eta = float(gd_cfg.get("eta", 0.02))
        self.lambda_reg = float(gd_cfg.get("lambda_reg", 0.5))
        self.gamma = float(gd_cfg.get("gamma", 1.0))
        cluster_mode_raw = str(gd_cfg.get("cluster_mode", gd_cfg.get("group_mode", "batch"))).lower()
        self.cluster_mode = self._CLUSTER_MODE_ALIASES.get(cluster_mode_raw)
        if self.cluster_mode is None:
            raise ValueError(
                f"Unsupported train.groupdro_plus.cluster_mode={cluster_mode_raw!r}. "
                "Use one of ['batch', 'per_batch', 'epoch', 'per_epoch']."
            )

        self.kmeans_iters = max(1, int(gd_cfg.get("kmeans_iters", 20)))
        epoch_cluster_cfg = gd_cfg.get("epoch_cluster", {})
        self.epoch_cluster_every_epochs = max(
            1,
            int(epoch_cluster_cfg.get("every_epochs", epoch_cluster_cfg.get("recluster_every_epochs", 1))),
        )
        self.epoch_cluster_source = str(epoch_cluster_cfg.get("source", "clean")).lower()
        if self.epoch_cluster_source not in {"clean", "adv"}:
            raise ValueError(
                f"Unsupported train.groupdro_plus.epoch_cluster.source={self.epoch_cluster_source!r}. "
                "Use 'clean' or 'adv'."
            )
        self.epoch_cluster_max_batches = max(0, int(epoch_cluster_cfg.get("max_batches", 0)))
        self.refresh_on_train_start = bool(epoch_cluster_cfg.get("refresh_on_train_start", True))

        snapshot_cfg = gd_cfg.get("snapshot", {})
        self.snapshot_enabled = bool(snapshot_cfg.get("enabled", False))
        self.snapshot_every_epochs = max(
            1,
            int(snapshot_cfg.get("every_epochs", snapshot_cfg.get("every_k_epochs", 1))),
        )
        self.snapshot_max_batches = max(0, int(snapshot_cfg.get("max_batches", 1)))
        splits_raw = snapshot_cfg.get("splits", snapshot_cfg.get("split", ["train"]))
        if isinstance(splits_raw, str):
            self.snapshot_splits = [splits_raw]
        elif isinstance(splits_raw, (list, tuple, set)):
            self.snapshot_splits = [str(split) for split in splits_raw]
        else:
            self.snapshot_splits = ["train"]
        self.snapshot_output_subdir = str(snapshot_cfg.get("output_subdir", "cluster_snapshots"))
        self.snapshot_save_pca = bool(snapshot_cfg.get("save_pca", True))

        self.q: torch.Tensor | None = None
        self.cluster_centers: torch.Tensor | None = None
        self.cluster_centers_epoch = 0
        self.cluster_counts: torch.Tensor | None = None
        self.attack = None
        if cfg["train"].get("adv_training", False):
            self.attack = build_train_attack(cfg, model)

    def _maybe_init_q(self, num_groups: int, device: torch.device) -> None:
        if self.q is None or self.q.numel() != num_groups:
            self.q = torch.ones(num_groups, device=device) / float(num_groups)

    def _prepare_inputs(
        self,
        model: Any,
        batch: Any,
        *,
        attack_source: str = "train",
    ) -> tuple[torch.Tensor, torch.Tensor, bool]:
        images, labels = unpack_xy(batch)
        use_adv = self.attack is not None and (
            attack_source == "train" or (attack_source == "epoch_cluster" and self.epoch_cluster_source == "adv")
        )
        if use_adv:
            was_training = bool(model.training)
            model.eval()
            with self.full_precision_context():
                images = self.attack(images, labels).detach()
            model.train(was_training)
        return images, labels, use_adv

    def _compute_group_losses(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        cluster_ids: torch.Tensor,
        num_groups: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = labels.device
        loss_g = torch.zeros(num_groups, device=device)
        counts = torch.zeros(num_groups, device=device)
        for group_idx in range(num_groups):
            mask = cluster_ids == group_idx
            if mask.any():
                loss_g[group_idx] = compute_loss(logits[mask], labels[mask])
                counts[group_idx] = mask.sum()
        return loss_g, counts

    def _cluster_batch(
        self,
        embeddings: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        labels, centers = run_kmeans_with_centers(
            embeddings.detach(),
            self.num_clusters,
            num_iters=self.kmeans_iters,
        )
        _, counts = compute_cluster_centers(embeddings.detach(), labels, self.num_clusters)
        return labels, centers, counts

    def _assign_epoch_clusters(
        self,
        embeddings: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.cluster_centers is None:
            labels, centers, counts = self._cluster_batch(embeddings)
            self.cluster_centers = centers.detach().to(device=embeddings.device, dtype=embeddings.dtype)
            self.cluster_counts = counts.detach().to(device=embeddings.device)
            return labels, self.cluster_centers, self.cluster_counts

        centers = self.cluster_centers.to(device=embeddings.device, dtype=embeddings.dtype)
        labels = assign_to_centers(embeddings.detach(), centers)
        _, counts = compute_cluster_centers(embeddings.detach(), labels, centers.size(0))
        return labels, centers, counts

    def compute_loss(self, model: Any, batch: Any) -> Tuple[torch.Tensor, Dict[str, float]]:
        images, labels, is_adv = self._prepare_inputs(model, batch, attack_source="train")
        device = labels.device

        with self.autocast_context():
            logits = model(images)
            base_loss = compute_loss(logits, labels)
        if self.cluster_mode == "epoch":
            cluster_ids, centers, counts = self._assign_epoch_clusters(logits.detach().float())
            num_groups = int(centers.size(0))
        else:
            cluster_ids, centers, counts = self._cluster_batch(logits.detach().float())
            num_groups = int(cluster_ids.max().item()) + 1 if int(cluster_ids.numel()) > 0 else 0
            counts = counts[:num_groups]

        self._maybe_init_q(num_groups, device)
        loss_g, observed_counts = self._compute_group_losses(logits, labels, cluster_ids, num_groups)
        observed = observed_counts > 0
        if observed.any():
            self.q[observed] = self.q[observed] * torch.exp(self.eta * loss_g[observed].detach())
        self.q = self.q / self.q.sum()

        group_weighted = (loss_g * self.q).sum()
        reg = (base_loss * (self.q[cluster_ids] ** self.gamma)).mean()
        total_loss = group_weighted + self.lambda_reg * reg

        correct = (logits.argmax(dim=1) == labels).sum().item()
        metrics: Dict[str, float] = {
            "loss": float(total_loss.item()),
            "acc": correct / max(labels.size(0), 1),
            ("loss_adv" if is_adv else "loss_clean"): float(total_loss.item()),
            ("acc_adv" if is_adv else "acc_clean"): correct / max(labels.size(0), 1),
            "correct": correct,
            "batch_size": labels.size(0),
            "q_max": float(self.q.max().item()),
            "q_min": float(self.q.min().item()),
            "reg": float(reg.item()),
            "cluster_count_active": float((observed_counts > 0).sum().item()),
            "cluster_mode_epoch": 1.0 if self.cluster_mode == "epoch" else 0.0,
        }
        for group_idx in range(min(3, num_groups)):
            metrics[f"loss_g{group_idx}"] = float(loss_g[group_idx].item())
            metrics[f"count_g{group_idx}"] = float(observed_counts[group_idx].item())
        return total_loss, metrics

    def state_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {}
        if self.q is not None:
            payload["q"] = self.q
        if self.cluster_centers is not None:
            payload["cluster_centers"] = self.cluster_centers.detach().cpu()
        if self.cluster_counts is not None:
            payload["cluster_counts"] = self.cluster_counts.detach().cpu()
        if self.cluster_centers_epoch:
            payload["cluster_centers_epoch"] = int(self.cluster_centers_epoch)
        payload["cluster_mode"] = self.cluster_mode
        return payload

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        q = state.get("q")
        if q is not None:
            self.q = q
        centers = state.get("cluster_centers")
        if centers is not None:
            self.cluster_centers = centers
        counts = state.get("cluster_counts")
        if counts is not None:
            self.cluster_counts = counts
        if state.get("cluster_centers_epoch") is not None:
            self.cluster_centers_epoch = int(state["cluster_centers_epoch"])

    def on_train_start(self, loaders: Dict[str, Any]) -> None:
        if self.cluster_mode != "epoch":
            return
        if not self.refresh_on_train_start:
            return
        if self.cluster_centers is not None:
            return
        self._refresh_epoch_clusters(loaders, epoch=0)

    def on_epoch_end(self, epoch: int, loaders: Dict[str, Any]) -> None:
        if self.cluster_mode == "epoch" and epoch > 0 and epoch % self.epoch_cluster_every_epochs == 0:
            self._refresh_epoch_clusters(loaders, epoch=epoch)

        if not self.snapshot_enabled:
            return
        if epoch <= 0 or epoch % self.snapshot_every_epochs != 0:
            return

        saved_paths = []
        for split in self.snapshot_splits:
            loader = loaders.get(split)
            if loader is None:
                continue
            saved_paths.extend(self._save_split_snapshots(epoch, split, loader))

        if saved_paths:
            self.logger.info(
                "Saved %d GroupDRO++ cluster snapshots at epoch %d under %s",
                len(saved_paths),
                epoch,
                f"{get_run_dir(self.cfg)}/{self.snapshot_output_subdir}",
            )

    def _refresh_epoch_clusters(self, loaders: Dict[str, Any], epoch: int) -> None:
        loader = loaders.get("train")
        model = self.model
        if loader is None or model is None:
            return

        device = next(model.parameters()).device
        was_training = bool(model.training)
        embeddings_all = []

        try:
            model.eval()
            for batch_idx, batch in enumerate(loader, start=1):
                moved = move_to_device(batch, device)
                images, _, _ = self._prepare_inputs(model, moved, attack_source="epoch_cluster")
                with torch.no_grad():
                    with self.autocast_context():
                        embeddings_all.append(model(images).detach().float().cpu())

                if self.epoch_cluster_max_batches > 0 and batch_idx >= self.epoch_cluster_max_batches:
                    break
        finally:
            model.train(was_training)

        if not embeddings_all:
            return

        stacked = torch.cat(embeddings_all, dim=0)
        _, centers = run_kmeans_with_centers(stacked, self.num_clusters, num_iters=self.kmeans_iters)
        cluster_ids = assign_to_centers(stacked, centers)
        _, counts = compute_cluster_centers(stacked, cluster_ids, self.num_clusters)

        self.cluster_centers = centers.detach().to(device=device, dtype=torch.float32)
        self.cluster_counts = counts.detach().to(device=device)
        self.cluster_centers_epoch = int(epoch)
        self._maybe_init_q(self.num_clusters, device)

        self.logger.info(
            "Refreshed GroupDRO++ epoch clusters at epoch=%d mode=%s source=%s counts=%s",
            epoch,
            self.cluster_mode,
            self.epoch_cluster_source,
            counts.detach().cpu().tolist(),
        )

    def _save_split_snapshots(self, epoch: int, split: str, loader: Any) -> list[str]:
        model = self.model
        if model is None:
            return []

        device = next(model.parameters()).device
        was_training = bool(model.training)
        saved_paths: list[str] = []

        try:
            model.eval()
            for batch_idx, batch in enumerate(loader, start=1):
                moved = move_to_device(batch, device)
                images, labels, is_adv = self._prepare_inputs(model, moved, attack_source="train")

                with torch.no_grad():
                    with self.autocast_context():
                        embeddings = model(images).float()
                    if self.cluster_mode == "epoch":
                        cluster_ids, centers, counts = self._assign_epoch_clusters(embeddings.detach())
                    else:
                        cluster_ids, centers, counts = self._cluster_batch(embeddings.detach())

                projection = (
                    project_embeddings_pca(embeddings.detach(), n_components=2)
                    if self.snapshot_save_pca
                    else None
                )

                target = (
                    f"{get_run_dir(self.cfg)}/{self.snapshot_output_subdir}/"
                    f"{split}/epoch_{epoch:04d}_batch_{batch_idx:04d}.npz"
                )
                save_cluster_snapshot(
                    target,
                    embeddings=embeddings.detach(),
                    labels=labels.detach(),
                    cluster_ids=cluster_ids.detach(),
                    centers=centers.detach(),
                    counts=counts.detach(),
                    q=self._snapshot_q(),
                    epoch=epoch,
                    batch_idx=batch_idx,
                    split=split,
                    projection_2d=projection,
                    is_adv=is_adv,
                )
                saved_paths.append(target)

                if self.snapshot_max_batches > 0 and batch_idx >= self.snapshot_max_batches:
                    break
        finally:
            model.train(was_training)

        return saved_paths

    def _snapshot_q(self) -> torch.Tensor | None:
        if self.q is None:
            return None
        if int(self.q.numel()) == self.num_clusters:
            return self.q.detach()
        if int(self.q.numel()) > self.num_clusters:
            return self.q[: self.num_clusters].detach()

        padded = torch.zeros(self.num_clusters, device=self.q.device, dtype=self.q.dtype)
        padded[: int(self.q.numel())] = self.q.detach()
        return padded
