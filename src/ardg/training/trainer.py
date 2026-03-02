"""Trainer with pluggable objectives (ERM/PGD/REx/GroupDRO/GroupDRO++)."""

import logging
import time
from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader

from ardg.training.losses import compute_loss
from ardg.training.objectives import build_objective
from ardg.utils.logging import log_metrics
from ardg.utils.paths import ensure_dir, get_run_dir


class Trainer:
    """Trainer that delegates loss logic to Objective registry."""

    def __init__(
        self,
        cfg: Dict[str, Any],
        model: Any,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        """Initialize the trainer."""
        self.cfg = cfg
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = torch.device(device or "cpu")
        self.model.to(self.device)
        self.logger = logger or logging.getLogger(__name__)
        self.max_train_batches = cfg["train"].get("max_batches")
        self.max_val_batches = cfg["train"].get("max_val_batches")
        self.log_interval = cfg["train"].get("log_interval", 50)
        self.global_step = 0

        opt_cfg = cfg["train"]["optimizer"]
        self.optimizer = torch.optim.SGD(
            model.parameters(),
            lr=opt_cfg["lr"],
            momentum=opt_cfg.get("momentum", 0.9),
            weight_decay=opt_cfg.get("weight_decay", 0.0),
        )
        self.scheduler = None
        sched_cfg = cfg["train"].get("scheduler", {})
        if sched_cfg.get("name") == "multistep":
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer,
                milestones=sched_cfg.get("milestones", []),
                gamma=sched_cfg.get("gamma", 0.1),
            )
        if sched_cfg.get("name") == "cosine":
            t_max = sched_cfg.get("t_max", cfg["train"].get("epochs", 100))
            eta_min = sched_cfg.get("eta_min", 0.0)
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=t_max,
                eta_min=eta_min,
            )

        self.objective = build_objective(cfg, model)
        self.best_metric = float("-inf")
        self.best_epoch = 0
        self.best_ckpt_path: Optional[str] = None

        es_cfg = cfg["train"].get("early_stopping", {})
        self.es_enabled = bool(es_cfg.get("enabled", False))
        self.es_patience = int(es_cfg.get("patience", 10))
        self.es_min_delta = float(es_cfg.get("min_delta", 0.0))
        self.es_warmup_epochs = int(es_cfg.get("warmup_epochs", 0))
        self.es_metric_key = _normalize_es_metric_key(es_cfg.get("metric", "acc"))
        mode_default = "min" if self.es_metric_key == "loss" else "max"
        self.es_mode = str(es_cfg.get("mode", mode_default)).lower()
        if self.es_mode not in {"min", "max"}:
            raise ValueError(f"Unsupported early_stopping.mode: {self.es_mode!r}. Use 'min' or 'max'.")
        self.es_best_metric = float("inf") if self.es_mode == "min" else float("-inf")
        self.es_bad_epochs = 0

    def train(self) -> Dict[str, str]:
        """Run the full training loop."""
        epochs = self.cfg["train"]["epochs"]
        last_ckpt_path = ""
        for epoch in range(1, epochs + 1):
            self.logger.info("Starting epoch %s", epoch)
            start = time.perf_counter()

            train_metrics = self.train_one_epoch(epoch)
            train_metrics["time_sec"] = time.perf_counter() - start
            train_metrics["device"] = str(self.device)
            log_metrics(self.logger, train_metrics, self.global_step, "train")

            val_metrics = self.validate(epoch)
            val_metrics["device"] = str(self.device)
            log_metrics(self.logger, val_metrics, self.global_step, "val")
            val_acc = float(val_metrics.get("acc", 0.0))

            try:
                self.objective.on_epoch_end(epoch, {"train": self.train_loader, "val": self.val_loader})
            except AttributeError:
                pass

            last_ckpt_path = self._save_checkpoint(
                "last",
                {
                    "epoch": epoch,
                    "val_acc": val_acc,
                    "val_loss": float(val_metrics.get("loss", 0.0)),
                },
                None,
            )

            if val_acc >= self.best_metric:
                self.best_metric = val_acc
                self.best_epoch = epoch
                self.best_ckpt_path = self._save_checkpoint(
                    "best",
                    {
                        "epoch": epoch,
                        "val_acc": val_acc,
                        "val_loss": float(val_metrics.get("loss", 0.0)),
                    },
                    None,
                )

            should_stop = self._update_early_stopping(epoch, val_metrics)
            if self.scheduler is not None:
                self.scheduler.step()
            self.logger.info("Finished epoch %s in %.2fs", epoch, train_metrics["time_sec"])
            if should_stop:
                self.logger.info("Stopping training early at epoch %s.", epoch)
                break

        if not last_ckpt_path:
            raise RuntimeError("No checkpoint saved during training.")
        best_path = self.best_ckpt_path or last_ckpt_path
        return {"last": last_ckpt_path, "best": best_path}

    def _update_early_stopping(self, epoch: int, val_metrics: Dict[str, float]) -> bool:
        if not self.es_enabled:
            return False

        metric_value = val_metrics.get(self.es_metric_key)
        if metric_value is None:
            self.logger.warning(
                "early_stopping.metric=%s not found in val metrics; disabling early stopping.",
                self.es_metric_key,
            )
            self.es_enabled = False
            return False

        current = float(metric_value)
        if self._is_es_improved(current):
            self.es_best_metric = current
            self.es_bad_epochs = 0
        elif epoch > self.es_warmup_epochs:
            self.es_bad_epochs += 1

        self.logger.info(
            "early_stopping metric=%s current=%.6f best=%.6f bad_epochs=%d/%d",
            self.es_metric_key,
            current,
            self.es_best_metric,
            self.es_bad_epochs,
            self.es_patience,
        )
        return epoch > self.es_warmup_epochs and self.es_bad_epochs >= self.es_patience

    def _is_es_improved(self, current: float) -> bool:
        if self.es_mode == "min":
            return current < (self.es_best_metric - self.es_min_delta)
        return current > (self.es_best_metric + self.es_min_delta)

    def train_one_epoch(self, epoch: int) -> Dict[str, float]:
        """Run one training epoch."""
        del epoch
        self.model.train()

        total_loss = 0.0
        total_correct = 0
        total_seen = 0

        for step_idx, batch in enumerate(self.train_loader, start=1):
            self.global_step += 1
            metrics = self._train_step(batch)
            total_loss += metrics["loss"] * metrics["batch_size"]
            total_correct += metrics["correct"]
            total_seen += metrics["batch_size"]

            if self.log_interval and step_idx % self.log_interval == 0:
                batch_metrics = {
                    "loss": metrics["loss"],
                    "acc": metrics["acc"],
                    "lr": metrics["lr"],
                }
                log_metrics(self.logger, batch_metrics, self.global_step, "train_batch")
            if self.max_train_batches and step_idx >= self.max_train_batches:
                break

        avg_loss = total_loss / max(total_seen, 1)
        acc = total_correct / max(total_seen, 1)
        current_lr = self.optimizer.param_groups[0]["lr"]
        return {"loss": avg_loss, "acc": acc, "lr": current_lr}

    @torch.no_grad()
    def validate(self, epoch: int) -> Dict[str, float]:
        """Run validation for one epoch."""
        del epoch
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_seen = 0
        for step_idx, batch in enumerate(self.val_loader, start=1):
            if isinstance(batch, dict):
                images, labels = batch["x"], batch["y"]
            else:
                images, labels = batch
            images = images.to(self.device)
            labels = labels.to(self.device)
            logits = self.model(images)
            loss = compute_loss(logits, labels)
            total_loss += loss.item() * images.size(0)
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_seen += images.size(0)
            if self.max_val_batches and step_idx >= self.max_val_batches:
                break
        return {"loss": total_loss / max(total_seen, 1), "acc": total_correct / max(total_seen, 1)}

    def _train_step(self, batch: Any) -> Dict[str, float]:
        """Run a single training step."""
        batch = _to_device(batch, self.device)
        batch = self.objective.preprocess_batch(batch, self.model)

        self.optimizer.zero_grad(set_to_none=True)
        loss, metrics = self.objective.loss(self.model, batch)
        loss.backward()
        self.optimizer.step()

        metrics["lr"] = self.optimizer.param_groups[0]["lr"]
        return metrics

    def _save_checkpoint(self, name: str, metrics: Dict[str, float], epoch: int | None) -> str:
        """Save a model checkpoint."""
        del epoch
        run_dir = get_run_dir(self.cfg)
        ensure_dir(run_dir)
        ckpt_path = f"{run_dir}/{name}.pt"
        torch.save({"model": self.model.state_dict(), "metrics": metrics}, ckpt_path)
        return ckpt_path


def _to_device(batch: Any, device: torch.device) -> Any:
    """Recursively move tensors in batch to device."""
    if isinstance(batch, dict):
        return {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}
    if isinstance(batch, (list, tuple)):
        moved = []
        for item in batch:
            moved.append(item.to(device) if hasattr(item, "to") else item)
        return tuple(moved)
    return batch.to(device) if hasattr(batch, "to") else batch


def _normalize_es_metric_key(metric: Any) -> str:
    key = str(metric or "acc").strip().lower().replace("val/", "").replace("val_", "")
    return key
