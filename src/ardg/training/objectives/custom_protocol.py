"""Custom training protocols, including checkpoint_base."""

from __future__ import annotations

import logging
from pathlib import Path
import time
from typing import Any, Dict, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ardg.attacks.attack_suite import build_train_attack, build_train_suite
from ardg.data.checkpoint_cache import CheckpointCacheDataset, CleanIndexDataset
from ardg.training.losses import compute_loss
from ardg.training.objectives.base import Objective
from ardg.utils.batch import unpack_xy
from ardg.utils.paths import ensure_dir


def _resolve_custom_protocol_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    nested = cfg.get("train", {}).get("custom_protocol", {})
    if isinstance(nested, dict):
        merged.update(nested)
    top_level = cfg.get("custom_protocol", {})
    if isinstance(top_level, dict):
        merged.update(top_level)
    return merged


class CustomProtocol(Objective):
    """Custom protocol objective with a checkpoint-based robust training mode."""

    def __init__(self, cfg: Dict[str, Any], model: Any) -> None:
        self.cfg = cfg
        self.model = model
        self.logger = logging.getLogger(__name__)
        self.proto_cfg = _resolve_custom_protocol_cfg(cfg)
        self.protocol_name = str(self.proto_cfg.get("name", "scaffold")).strip().lower()
        self.protocol_step = 0

        if self.protocol_name == "checkpoint_base":
            self._init_checkpoint_base()
            return

        self.weight_main = float(self.proto_cfg.get("weight_main", 1.0))
        self.weight_aux = float(self.proto_cfg.get("weight_aux", 0.0))
        self.use_adv_inputs = bool(
            self.proto_cfg.get(
                "use_adv_inputs",
                cfg.get("train", {}).get("adv_training", False),
            )
        )
        self.attack = build_train_attack(cfg, model) if self.use_adv_inputs else None

    def _init_checkpoint_base(self) -> None:
        suite = build_train_suite(self.cfg, self.model)
        if not suite:
            raise ValueError(
                "checkpoint_base requires attack.train_domains, attack.train_suite, or attack.train."
            )
        self.attack_names = list(suite.keys())
        self.attacks = [suite[name] for name in self.attack_names]
        self.num_groups = len(self.attacks)
        self.q = torch.full((self.num_groups,), 1.0 / float(self.num_groups), dtype=torch.float32)

        self.period_epochs = max(1, int(self.proto_cfg.get("period_epochs", 2)))
        self.warmup_epochs = max(0, int(self.proto_cfg.get("warmup_epochs", 0)))
        self.score_metric = str(self.proto_cfg.get("score_metric", "ce_loss")).strip().lower()
        if self.score_metric != "ce_loss":
            raise ValueError(
                f"checkpoint_base only supports score_metric='ce_loss', got {self.score_metric!r}."
            )

        update_rule = self.proto_cfg.get("update_rule", {})
        self.temperature = max(1e-8, float(update_rule.get("temperature", 1.0)))
        self.alpha = float(update_rule.get("alpha", 0.5))
        self.beta = float(update_rule.get("beta", 0.1))

        allocation_cfg = self.proto_cfg.get("allocation", {})
        self.allocation_mode = str(allocation_cfg.get("mode", "quota")).strip().lower()
        if self.allocation_mode != "quota":
            raise ValueError(
                f"checkpoint_base only supports allocation.mode='quota', got {self.allocation_mode!r}."
            )
        self.one_attack_per_sample = bool(allocation_cfg.get("one_attack_per_sample", True))
        if not self.one_attack_per_sample:
            raise ValueError("checkpoint_base v1 requires allocation.one_attack_per_sample=true.")

        cache_cfg = self.proto_cfg.get("cache", {})
        self.save_to_disk = bool(cache_cfg.get("save_to_disk", True))
        self.cache_directory = self._resolve_cache_directory(
            str(cache_cfg.get("directory", "artifacts/checkpoint_base_cache") or "artifacts/checkpoint_base_cache")
        )

        self.refresh_subset_spec = self.proto_cfg.get("refresh_subset_size", 0.05)
        self.refresh_subset_indices: list[int] = []
        self.refresh_subset_count = 0

        self.cache_dataset: CheckpointCacheDataset | None = None
        self.generation_batch_size = int(self.cfg.get("train", {}).get("batch_size", 128))
        self.current_period_idx = -1
        self.current_period_start_epoch = 0
        self.current_period_end_epoch = 0
        self.current_epoch = 1
        self.current_cache_manifest: Dict[str, Any] = {}
        self.latest_scores: Dict[str, float] = {name: 0.0 for name in self.attack_names}
        self.latest_cache_counts: Dict[str, int] = {name: 0 for name in self.attack_names}
        self.latest_refresh_score_time_sec = 0.0
        self.latest_cache_generate_time_sec = 0.0
        self._pending_train_time_sec = 0.0
        self.seed = int(self.cfg.get("experiment", {}).get("seed", 0))

    def _resolve_cache_directory(self, directory: str) -> str:
        path = Path(directory)
        if not path.is_absolute():
            run_dir = self.cfg.get("_meta", {}).get("run_dir")
            base = Path(run_dir) if run_dir else Path(".")
            path = base / path
        return str(path)

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
        if self.protocol_name == "checkpoint_base":
            return self._compute_checkpoint_base_loss(model, batch)

        images, labels, used_adv = self._prepare_inputs(model, batch)
        with self.autocast_context():
            logits = model(images)
            base_loss = compute_loss(logits, labels)
            aux_loss = torch.zeros((), device=logits.device, dtype=base_loss.dtype)
            total_loss = (self.weight_main * base_loss) + (self.weight_aux * aux_loss)

        self.protocol_step += 1
        correct = int((logits.argmax(dim=1) == labels).sum().item())
        metrics: Dict[str, float] = {
            "loss": float(total_loss.item()),
            "acc": correct / max(labels.size(0), 1),
            ("loss_adv" if used_adv else "loss_clean"): float(total_loss.item()),
            ("acc_adv" if used_adv else "acc_clean"): correct / max(labels.size(0), 1),
            "loss_base": float(base_loss.item()),
            "loss_aux": float(aux_loss.item()),
            "correct": correct,
            "batch_size": int(labels.size(0)),
            "protocol_step": float(self.protocol_step),
            "weight_main": float(self.weight_main),
            "weight_aux": float(self.weight_aux),
        }
        return total_loss, metrics

    def _compute_checkpoint_base_loss(
        self,
        model: Any,
        batch: Any,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        if self._checkpoint_base_warmup_active():
            return self._compute_checkpoint_base_warmup_loss(model, batch)
        dataset = self._require_cache_dataset()
        if not dataset.has_active_cache():
            raise RuntimeError(
                "checkpoint_base expected a cached attacked dataset after warmup, but no cache is active."
            )
        images, labels = unpack_xy(batch)
        with self.autocast_context():
            logits = model(images)
            loss = compute_loss(logits, labels)

        self.protocol_step += 1
        batch_size = int(labels.size(0))
        correct = int((logits.argmax(dim=1) == labels).sum().item())
        metrics: Dict[str, float] = {
            "loss": float(loss.item()),
            "loss_adv": float(loss.item()),
            "acc": correct / max(batch_size, 1),
            "acc_adv": correct / max(batch_size, 1),
            "correct": correct,
            "batch_size": batch_size,
        }
        return loss, metrics

    def _compute_checkpoint_base_warmup_loss(
        self,
        model: Any,
        batch: Any,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        images, labels = unpack_xy(batch)
        batch_size = int(labels.size(0))
        attack_ids = self._sample_online_attack_ids(batch_size, device=labels.device)
        attacked = images.detach().clone()
        was_training = bool(model.training)
        try:
            model.eval()
            for attack_idx in torch.unique(attack_ids).detach().cpu().tolist():
                mask = attack_ids == int(attack_idx)
                with self.full_precision_context():
                    attacked[mask] = self.attacks[int(attack_idx)](images[mask], labels[mask]).detach()
        finally:
            model.train(was_training)

        with self.autocast_context():
            logits = model(attacked)
            loss = compute_loss(logits, labels)

        self.protocol_step += 1
        correct = int((logits.argmax(dim=1) == labels).sum().item())
        metrics: Dict[str, float] = {
            "loss": float(loss.item()),
            "loss_adv": float(loss.item()),
            "acc": correct / max(batch_size, 1),
            "acc_adv": correct / max(batch_size, 1),
            "correct": correct,
            "batch_size": batch_size,
            "warmup_online": 1.0,
        }
        return loss, metrics

    def validate(self, model: Any, loader: Any) -> Dict[str, float]:
        del model, loader
        return {}

    def on_train_start(self, loaders: Dict[str, Any]) -> None:
        if self.protocol_name != "checkpoint_base":
            return None

        train_loader = loaders.get("train")
        if train_loader is None or not hasattr(train_loader, "dataset"):
            raise ValueError("checkpoint_base requires access to the training DataLoader.")
        if getattr(train_loader, "batch_size", None):
            self.generation_batch_size = int(train_loader.batch_size)

        dataset = train_loader.dataset
        if not isinstance(dataset, CheckpointCacheDataset):
            raise ValueError(
                "checkpoint_base requires the training dataset to be wrapped by CheckpointCacheDataset."
            )
        self.cache_dataset = dataset

        if not self.refresh_subset_indices:
            self.refresh_subset_indices = self._build_refresh_subset_indices(len(dataset))
            self.refresh_subset_count = len(self.refresh_subset_indices)

        if dataset.has_active_cache():
            return None

        if self.current_cache_manifest:
            self._load_current_cache_from_disk()
            return None

        if self._checkpoint_base_warmup_active():
            self.logger.info(
                "checkpoint_base warmup active for epochs 1-%s; using online adversarial generation before cache periods.",
                self.warmup_epochs,
            )
            return None

        total_epochs = int(self.cfg.get("train", {}).get("epochs", self.period_epochs))
        self.current_period_idx = 0
        self.current_period_start_epoch = 1
        self.current_period_end_epoch = min(self.period_epochs, total_epochs)
        attack_ids = self._allocate_attack_ids(len(dataset), period_idx=self.current_period_idx)
        self.current_cache_manifest = self._generate_period_cache(
            attack_ids,
            period_idx=self.current_period_idx,
            period_start_epoch=self.current_period_start_epoch,
            period_end_epoch=self.current_period_end_epoch,
        )
        self.latest_scores = {name: 0.0 for name in self.attack_names}
        self.latest_cache_counts = dict(self.current_cache_manifest.get("cache_counts", {}))
        self._emit_boundary_metrics(event="initial_cache")
        return None

    def on_epoch_end(self, epoch: int, loaders: Dict[str, Any]) -> None:
        del loaders
        if self.protocol_name != "checkpoint_base":
            return None

        total_epochs = int(self.cfg.get("train", {}).get("epochs", epoch))
        if epoch >= total_epochs:
            self.current_epoch = int(epoch) + 1
            return None

        if self._checkpoint_base_warmup_active():
            if epoch < self.warmup_epochs:
                self.current_epoch = int(epoch) + 1
                return None

            scores = self._score_attacks_on_refresh_subset()
            self._update_q_from_scores(scores)
            self.current_period_idx = 0
            self.current_period_start_epoch = epoch + 1
            self.current_period_end_epoch = min(
                self.current_period_start_epoch + self.period_epochs - 1,
                total_epochs,
            )
            attack_ids = self._allocate_attack_ids(
                len(self._require_cache_dataset()),
                period_idx=self.current_period_idx,
            )
            self.current_cache_manifest = self._generate_period_cache(
                attack_ids,
                period_idx=self.current_period_idx,
                period_start_epoch=self.current_period_start_epoch,
                period_end_epoch=self.current_period_end_epoch,
            )
            self.latest_scores = scores
            self.latest_cache_counts = dict(self.current_cache_manifest.get("cache_counts", {}))
            self._emit_boundary_metrics(event="post_warmup_initial_cache")
            self.current_epoch = int(epoch) + 1
            return None

        if epoch != self.current_period_end_epoch:
            self.current_epoch = int(epoch) + 1
            return None

        scores = self._score_attacks_on_refresh_subset()
        self._update_q_from_scores(scores)

        self.current_period_idx += 1
        self.current_period_start_epoch = epoch + 1
        self.current_period_end_epoch = min(
            self.current_period_start_epoch + self.period_epochs - 1,
            total_epochs,
        )
        attack_ids = self._allocate_attack_ids(len(self._require_cache_dataset()), period_idx=self.current_period_idx)
        self.current_cache_manifest = self._generate_period_cache(
            attack_ids,
            period_idx=self.current_period_idx,
            period_start_epoch=self.current_period_start_epoch,
            period_end_epoch=self.current_period_end_epoch,
        )
        self.latest_scores = scores
        self.latest_cache_counts = dict(self.current_cache_manifest.get("cache_counts", {}))
        self._emit_boundary_metrics(event="period_refresh")
        self.current_epoch = int(epoch) + 1
        return None

    def _require_cache_dataset(self) -> CheckpointCacheDataset:
        if self.cache_dataset is None:
            raise RuntimeError("checkpoint_base cache dataset is not attached yet.")
        return self.cache_dataset

    def _build_refresh_subset_indices(self, num_samples: int) -> list[int]:
        if num_samples <= 0:
            raise ValueError("checkpoint_base requires a non-empty training dataset.")
        spec = self.refresh_subset_spec
        if isinstance(spec, float) and 0.0 < spec <= 1.0:
            count = max(1, int(round(spec * num_samples)))
        else:
            count = int(spec)
        count = max(1, min(count, num_samples))
        generator = torch.Generator()
        generator.manual_seed(self.seed + 101)
        order = torch.randperm(num_samples, generator=generator).tolist()
        return [int(idx) for idx in order[:count]]

    def _score_attacks_on_refresh_subset(self) -> Dict[str, float]:
        dataset = self._require_cache_dataset()
        was_training = bool(self.model.training)
        device = self._model_device()
        scores: Dict[str, float] = {}
        start = time.perf_counter()
        try:
            self.model.eval()
            for attack_idx, attack_name in enumerate(self.attack_names):
                view = CleanIndexDataset(dataset, self.refresh_subset_indices)
                loader = DataLoader(
                    view,
                    batch_size=self.generation_batch_size,
                    shuffle=False,
                    num_workers=0,
                    pin_memory=torch.cuda.is_available(),
                )
                total_loss = 0.0
                total_seen = 0
                attack = self.attacks[attack_idx]
                for batch in loader:
                    images = batch["x"].to(device)
                    labels = batch["y"].to(device=device, dtype=torch.long).view(-1)
                    with self.full_precision_context():
                        attacked = attack(images, labels).detach()
                        logits = self.model(attacked)
                        loss_sum = F.cross_entropy(logits, labels, reduction="sum")
                    total_loss += float(loss_sum.detach().cpu().item())
                    total_seen += int(labels.size(0))
                scores[attack_name] = total_loss / max(total_seen, 1)
        finally:
            self.model.train(was_training)
        self.latest_refresh_score_time_sec = time.perf_counter() - start
        return scores

    def _update_q_from_scores(self, scores: Dict[str, float]) -> None:
        ordered_scores = torch.tensor(
            [float(scores[name]) for name in self.attack_names],
            dtype=torch.float32,
        )
        probs = torch.softmax(ordered_scores / self.temperature, dim=0)
        uniform = torch.full_like(probs, 1.0 / float(self.num_groups))
        q_bar = (1.0 - self.alpha) * self.q + self.alpha * probs.cpu()
        q_new = (1.0 - self.beta) * q_bar + self.beta * uniform.cpu()
        q_sum = q_new.sum()
        if not torch.isfinite(q_sum) or float(q_sum.item()) <= 0.0:
            raise RuntimeError("checkpoint_base q update produced a non-finite or non-positive normalizer.")
        self.q = (q_new / q_sum).detach().cpu()

    def _allocate_attack_ids(self, num_samples: int, *, period_idx: int) -> torch.Tensor:
        expected = self.q * float(num_samples)
        counts = torch.floor(expected).to(dtype=torch.long)
        remainder = int(num_samples - int(counts.sum().item()))
        if remainder > 0:
            fractional = expected - counts.to(dtype=torch.float32)
            order = torch.argsort(fractional, descending=True).tolist()
            for idx in order[:remainder]:
                counts[int(idx)] += 1

        attack_ids = torch.empty(num_samples, dtype=torch.long)
        cursor = 0
        for attack_idx, count in enumerate(counts.tolist()):
            if count <= 0:
                continue
            attack_ids[cursor : cursor + count] = int(attack_idx)
            cursor += int(count)
        if cursor != num_samples:
            raise RuntimeError("checkpoint_base attack allocation did not cover the full dataset.")

        generator = torch.Generator()
        generator.manual_seed(self.seed + 1000 + int(period_idx))
        perm = torch.randperm(num_samples, generator=generator)
        return attack_ids[perm]

    def _generate_period_cache(
        self,
        attack_ids: torch.Tensor,
        *,
        period_idx: int,
        period_start_epoch: int,
        period_end_epoch: int,
    ) -> Dict[str, Any]:
        dataset = self._require_cache_dataset()
        num_samples = len(dataset)
        cache_labels = torch.empty(num_samples, dtype=torch.long)
        cache_attack_ids = attack_ids.detach().cpu().clone().to(dtype=torch.long).view(-1)
        cache_images: torch.Tensor | None = None
        cache_counts = {name: 0 for name in self.attack_names}
        device = self._model_device()
        was_training = bool(self.model.training)
        start = time.perf_counter()

        try:
            self.model.eval()
            for attack_idx, attack_name in enumerate(self.attack_names):
                selected = (cache_attack_ids == attack_idx).nonzero(as_tuple=False).view(-1).tolist()
                cache_counts[attack_name] = len(selected)
                if not selected:
                    continue

                loader = DataLoader(
                    CleanIndexDataset(dataset, selected),
                    batch_size=self.generation_batch_size,
                    shuffle=False,
                    num_workers=0,
                    pin_memory=torch.cuda.is_available(),
                )
                attack = self.attacks[attack_idx]
                for batch in loader:
                    images = batch["x"].to(device)
                    labels = batch["y"].to(device=device, dtype=torch.long).view(-1)
                    sample_indices = batch["sample_idx"].to(dtype=torch.long).view(-1).cpu()
                    with self.full_precision_context():
                        attacked = attack(images, labels).detach().cpu()
                    if cache_images is None:
                        cache_images = torch.empty(
                            (num_samples, *attacked.shape[1:]),
                            dtype=attacked.dtype,
                        )
                    cache_images[sample_indices] = attacked
                    cache_labels[sample_indices] = batch["y"].to(dtype=torch.long).view(-1).cpu()
        finally:
            self.model.train(was_training)

        if cache_images is None:
            raise RuntimeError("checkpoint_base failed to generate any attacked samples for the next period.")

        metadata = {
            "period_idx": int(period_idx),
            "period_start_epoch": int(period_start_epoch),
            "period_end_epoch": int(period_end_epoch),
            "cache_counts": dict(cache_counts),
            "attack_names": list(self.attack_names),
            "refresh_subset_indices": list(self.refresh_subset_indices),
        }
        dataset.set_cache(cache_images, cache_labels, cache_attack_ids, metadata=metadata)

        cache_path = None
        if self.save_to_disk:
            ensure_dir(self.cache_directory)
            cache_path = dataset.save_cache(self._cache_file_path(period_idx), metadata=metadata)

        self.latest_cache_generate_time_sec = time.perf_counter() - start
        self._pending_train_time_sec += self.latest_cache_generate_time_sec
        manifest = dict(metadata)
        manifest["path"] = cache_path
        return manifest

    def _cache_file_path(self, period_idx: int) -> str:
        return str(Path(self.cache_directory) / f"period_{int(period_idx):04d}.pt")

    def _load_current_cache_from_disk(self) -> None:
        dataset = self._require_cache_dataset()
        cache_path = str(self.current_cache_manifest.get("path", "") or "").strip()
        if not cache_path:
            raise RuntimeError(
                "checkpoint_base resume requires a saved cache on disk, but current_cache_manifest.path is missing."
            )
        metadata = dataset.load_cache(cache_path)
        if metadata:
            self.current_cache_manifest = dict(self.current_cache_manifest)
            self.current_cache_manifest.setdefault("cache_counts", metadata.get("cache_counts", {}))
            self.current_cache_manifest.setdefault("period_idx", metadata.get("period_idx"))
            self.current_cache_manifest.setdefault("period_start_epoch", metadata.get("period_start_epoch"))
            self.current_cache_manifest.setdefault("period_end_epoch", metadata.get("period_end_epoch"))

    def _emit_boundary_metrics(self, *, event: str) -> None:
        if self.protocol_name != "checkpoint_base":
            return
        metrics: Dict[str, Any] = {
            "checkpoint_base/event": str(event),
            "checkpoint_base/period_idx": int(self.current_period_idx),
            "checkpoint_base/period_start_epoch": int(self.current_period_start_epoch),
            "checkpoint_base/period_end_epoch": int(self.current_period_end_epoch),
            "checkpoint_base/q_entropy": float(self._q_entropy().item()),
            "checkpoint_base/refresh_subset_size": int(self.refresh_subset_count or len(self.refresh_subset_indices)),
            "checkpoint_base/refresh_score_time_sec": float(self.latest_refresh_score_time_sec),
            "checkpoint_base/cache_generate_time_sec": float(self.latest_cache_generate_time_sec),
        }
        for attack_idx, attack_name in enumerate(self.attack_names):
            metrics[f"checkpoint_base/q/{attack_name}"] = float(self.q[attack_idx].item())
            metrics[f"checkpoint_base/score/{attack_name}"] = float(self.latest_scores.get(attack_name, 0.0))
            metrics[f"checkpoint_base/cache_count/{attack_name}"] = int(self.latest_cache_counts.get(attack_name, 0))

        self.logger.info(
            "checkpoint_base event=%s period=%s epochs=%s-%s refresh_subset=%s "
            "refresh_score_time_sec=%.3f cache_generate_time_sec=%.3f cache_path=%s",
            event,
            self.current_period_idx,
            self.current_period_start_epoch,
            self.current_period_end_epoch,
            int(self.refresh_subset_count or len(self.refresh_subset_indices)),
            self.latest_refresh_score_time_sec,
            self.latest_cache_generate_time_sec,
            self.current_cache_manifest.get("path"),
        )
        self.logger.info("checkpoint_base q=%s", {name: float(self.q[idx].item()) for idx, name in enumerate(self.attack_names)})
        self.logger.info("checkpoint_base scores=%s", dict(self.latest_scores))
        self.logger.info("checkpoint_base cache_counts=%s", dict(self.latest_cache_counts))
        wandb_cfg = self.cfg.get("logging", {}).get("wandb", {})
        if not bool(wandb_cfg.get("enabled", False)):
            return
        try:
            import wandb  # type: ignore
        except ImportError:
            return
        if getattr(wandb, "run", None) is not None:
            wandb.log(metrics)

    def _q_entropy(self) -> torch.Tensor:
        q = self.q.clamp_min(1e-12)
        return -(q * q.log()).sum()

    def _checkpoint_base_warmup_active(self) -> bool:
        return int(self.current_epoch) <= int(self.warmup_epochs)

    def _sample_online_attack_ids(self, batch_size: int, *, device: torch.device) -> torch.Tensor:
        probs = self.q.detach().to(dtype=torch.float32).cpu()
        probs = probs / probs.sum().clamp_min(1e-12)
        sampled = torch.multinomial(probs, num_samples=int(batch_size), replacement=True)
        return sampled.to(device=device, dtype=torch.long)

    def consume_train_time_seconds(self) -> float:
        extra = float(self._pending_train_time_sec)
        self._pending_train_time_sec = 0.0
        return extra

    def _model_device(self) -> torch.device:
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def state_dict(self) -> Dict[str, Any]:
        if self.protocol_name != "checkpoint_base":
            return {"protocol_step": int(self.protocol_step)}
        return {
            "protocol_name": self.protocol_name,
            "protocol_step": int(self.protocol_step),
            "q": self.q.detach().cpu(),
            "period_idx": int(self.current_period_idx),
            "period_start_epoch": int(self.current_period_start_epoch),
            "period_end_epoch": int(self.current_period_end_epoch),
            "current_epoch": int(self.current_epoch),
            "refresh_subset_indices": [int(idx) for idx in self.refresh_subset_indices],
            "refresh_subset_count": int(self.refresh_subset_count or len(self.refresh_subset_indices)),
            "current_cache_manifest": dict(self.current_cache_manifest),
            "latest_scores": dict(self.latest_scores),
            "latest_cache_counts": dict(self.latest_cache_counts),
            "latest_refresh_score_time_sec": float(self.latest_refresh_score_time_sec),
            "latest_cache_generate_time_sec": float(self.latest_cache_generate_time_sec),
            "pending_train_time_sec": float(self._pending_train_time_sec),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if self.protocol_name != "checkpoint_base":
            if state.get("protocol_step") is not None:
                self.protocol_step = int(state["protocol_step"])
            return None

        if state.get("protocol_step") is not None:
            self.protocol_step = int(state["protocol_step"])
        q = state.get("q")
        if q is not None:
            q_tensor = q.detach().clone().to(dtype=torch.float32)
            if int(q_tensor.numel()) != self.num_groups:
                raise ValueError(
                    f"checkpoint_base checkpoint q has {int(q_tensor.numel())} entries, expected {self.num_groups}."
                )
            q_sum = q_tensor.sum()
            if not torch.isfinite(q_sum) or float(q_sum.item()) <= 0.0:
                raise ValueError("checkpoint_base checkpoint q must sum to a positive finite value.")
            self.q = (q_tensor / q_sum).detach().cpu()

        if state.get("period_idx") is not None:
            self.current_period_idx = int(state["period_idx"])
        if state.get("period_start_epoch") is not None:
            self.current_period_start_epoch = int(state["period_start_epoch"])
        if state.get("period_end_epoch") is not None:
            self.current_period_end_epoch = int(state["period_end_epoch"])
        if state.get("current_epoch") is not None:
            self.current_epoch = int(state["current_epoch"])
        if isinstance(state.get("refresh_subset_indices"), list):
            self.refresh_subset_indices = [int(idx) for idx in state["refresh_subset_indices"]]
        if state.get("refresh_subset_count") is not None:
            self.refresh_subset_count = int(state["refresh_subset_count"])
        elif self.refresh_subset_indices:
            self.refresh_subset_count = len(self.refresh_subset_indices)
        if isinstance(state.get("current_cache_manifest"), dict):
            self.current_cache_manifest = dict(state["current_cache_manifest"])
        if isinstance(state.get("latest_scores"), dict):
            self.latest_scores = {str(k): float(v) for k, v in state["latest_scores"].items()}
        if isinstance(state.get("latest_cache_counts"), dict):
            self.latest_cache_counts = {str(k): int(v) for k, v in state["latest_cache_counts"].items()}
        if state.get("latest_refresh_score_time_sec") is not None:
            self.latest_refresh_score_time_sec = float(state["latest_refresh_score_time_sec"])
        if state.get("latest_cache_generate_time_sec") is not None:
            self.latest_cache_generate_time_sec = float(state["latest_cache_generate_time_sec"])
        if state.get("pending_train_time_sec") is not None:
            self._pending_train_time_sec = float(state["pending_train_time_sec"])
        return None
