"""Trainer with pluggable objectives (ERM/PGD/REx/GroupDRO/GroupDRO++)."""

import logging
import time
from pathlib import Path
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
        self.model = model # build model 

        # get train and val loader
        self.train_loader = train_loader 
        self.val_loader = val_loader

        # load model to device
        self.device = torch.device(device or "cpu")
        self.model.to(self.device)
        
        self.logger = logger or logging.getLogger(__name__) # init logging

        # smoke test if need 
        self.max_train_batches = cfg["train"].get("max_batches")
        self.max_val_batches = cfg["train"].get("max_val_batches")
        self.log_interval = cfg["train"].get("log_interval", 50)
        self.global_step = 0
        self.start_epoch = 1
        wandb_cfg = cfg.get("logging", {}).get("wandb", {})
        self.wandb_run_id = str(wandb_cfg.get("run_id", "")).strip() or None

        # optimizer
        opt_cfg = cfg["train"]["optimizer"]
        self.optimizer = torch.optim.SGD(
            model.parameters(),
            lr=opt_cfg["lr"],
            momentum=opt_cfg.get("momentum", 0.9),
            weight_decay=opt_cfg.get("weight_decay", 0.0),
        )

        # scheduler
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

        # build loss function 
        self.objective = build_objective(cfg, model)
        self.best_metric = float("-inf")
        self.best_epoch = 0
        self.best_ckpt_path: Optional[str] = None

        # early stopping
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

        # resume previous training if available
        if self.start_epoch > epochs:
            self.logger.info(
                "Resume epoch (%s) is beyond configured epochs (%s). Skipping training.",
                self.start_epoch,
                epochs,
            )
            run_dir = Path(get_run_dir(self.cfg))
            last_path = str(run_dir / "last.pt")
            best_path = self.best_ckpt_path or str(run_dir / "best.pt")
            if not Path(best_path).exists():
                best_path = last_path
            return {"last": last_path, "best": best_path}

        # training loop
        for epoch in range(self.start_epoch, epochs + 1):
            self.logger.info("Starting epoch %s", epoch)
            start = time.perf_counter()

            train_metrics = self.train_one_epoch(epoch) # loss, acc, lr
            train_metrics["time_sec"] = time.perf_counter() - start # compute time
            train_metrics["device"] = str(self.device)
            log_metrics(self.logger, train_metrics, self.global_step, "train")

            val_metrics = self.validate(epoch) # loss, acc
            val_metrics["device"] = str(self.device)
            log_metrics(self.logger, val_metrics, self.global_step, "val")
            val_acc = float(val_metrics.get("acc", 0.0)) # cache val acc for select best model 
            train_mode = str(self.cfg.get("train", {}).get("mode", "")).lower()
            ckpt_metric_name = "acc"
            ckpt_metric = val_acc
            if train_mode in {"multi_attack_erm", "multi-attack-erm", "multi_attack"}:
                if "pgd20_probe_acc" in val_metrics:
                    ckpt_metric_name = "pgd20_probe_acc"
                    ckpt_metric = float(val_metrics["pgd20_probe_acc"])

            try:
                self.objective.on_epoch_end(epoch, {"train": self.train_loader, "val": self.val_loader})
            except AttributeError:
                pass
            
            # select best model 
            if ckpt_metric >= self.best_metric:
                self.best_metric = ckpt_metric
                self.best_epoch = epoch
                self.best_ckpt_path = self._save_checkpoint(
                    "best",
                    {
                        "epoch": epoch,
                        "val_acc": val_acc,
                        "selection_metric": ckpt_metric,
                        "selection_metric_name": ckpt_metric_name,
                        "val_loss": float(val_metrics.get("loss", 0.0)),
                    },
                    epoch,
                )

            last_ckpt_path = self._save_checkpoint(
                "last",
                {
                    "epoch": epoch,
                    "val_acc": val_acc,
                    "selection_metric": ckpt_metric,
                    "selection_metric_name": ckpt_metric_name,
                    "val_loss": float(val_metrics.get("loss", 0.0)),
                },
                epoch,
            )

            # early stopping
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

    def load_checkpoint(self, ckpt_path: str) -> Dict[str, Any]:
        """Load training state from checkpoint and prepare resume."""
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        if not isinstance(checkpoint, dict):
            raise ValueError(f"Unsupported checkpoint format at {ckpt_path}.")
        metrics = checkpoint.get("metrics")
        if not isinstance(metrics, dict):
            metrics = {}

        state_dict = checkpoint.get("model") or checkpoint.get("state_dict")
        if not isinstance(state_dict, dict):
            raise ValueError(f"Checkpoint at {ckpt_path} has no model state_dict.")
        self.model.load_state_dict(state_dict)

        if isinstance(checkpoint.get("optimizer"), dict):
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        else:
            self.logger.warning("No optimizer state in checkpoint; optimizer will restart fresh.")

        resumed_epoch = int(checkpoint.get("epoch", metrics.get("epoch", 0)))
        if self.scheduler is not None:
            if isinstance(checkpoint.get("scheduler"), dict):
                self.scheduler.load_state_dict(checkpoint["scheduler"])
            elif resumed_epoch > 0:
                # Best-effort fallback for legacy checkpoints without scheduler state.
                for _ in range(resumed_epoch):
                    self.scheduler.step()

        if checkpoint.get("global_step") is not None:
            self.global_step = int(checkpoint["global_step"])
        else:
            self.global_step = resumed_epoch * self._steps_per_epoch()

        if checkpoint.get("best_metric") is not None:
            self.best_metric = float(checkpoint["best_metric"])
        else:
            self.best_metric = float(metrics.get("val_acc", self.best_metric))
        self.best_epoch = int(checkpoint.get("best_epoch", resumed_epoch))

        if checkpoint.get("es_state") and isinstance(checkpoint["es_state"], dict):
            es_state = checkpoint["es_state"]
            if es_state.get("best_metric") is not None:
                self.es_best_metric = float(es_state["best_metric"])
            if es_state.get("bad_epochs") is not None:
                self.es_bad_epochs = int(es_state["bad_epochs"])
            if es_state.get("enabled") is not None:
                self.es_enabled = bool(es_state["enabled"])

        ckpt_run_id = checkpoint.get("wandb_run_id")
        if ckpt_run_id:
            self.wandb_run_id = str(ckpt_run_id)

        run_dir = Path(get_run_dir(self.cfg))
        best_path = run_dir / "best.pt"
        if best_path.exists():
            self.best_ckpt_path = str(best_path)

        self.start_epoch = max(1, resumed_epoch + 1)
        self.logger.info(
            "Resumed from %s (epoch=%s, next_epoch=%s, global_step=%s).",
            ckpt_path,
            resumed_epoch,
            self.start_epoch,
            self.global_step,
        )
        return {
            "epoch": resumed_epoch,
            "next_epoch": self.start_epoch,
            "global_step": self.global_step,
            "wandb_run_id": self.wandb_run_id,
        }

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

    def train_one_epoch(self, epoch: int) -> Dict[str, Any]:
        """Run one training epoch."""
        del epoch
        self.model.train()

        total_loss = 0.0
        total_correct = 0
        total_seen = 0
        domain_counts: Dict[str, float] = {}
        last_domain_name: str | None = None

        for step_idx, batch in enumerate(self.train_loader, start=1):
            self.global_step += 1
            metrics = self._train_step(batch)
            total_loss += metrics["loss"] * metrics["batch_size"]
            total_correct += metrics["correct"]
            total_seen += metrics["batch_size"]
            if "domain_name" in metrics:
                last_domain_name = str(metrics["domain_name"])
            batch_domain_counts = metrics.get("domain_batch_counts")
            if isinstance(batch_domain_counts, dict):
                for name, count in batch_domain_counts.items():
                    domain_counts[str(name)] = domain_counts.get(str(name), 0.0) + float(count)

            if self.log_interval and step_idx % self.log_interval == 0:
                batch_metrics = {
                    "loss": metrics["loss"],
                    "acc": metrics["acc"],
                    "lr": metrics["lr"],
                }
                if "domain_name" in metrics:
                    batch_metrics["domain_name"] = metrics["domain_name"]
                log_metrics(self.logger, batch_metrics, self.global_step, "train_batch")
            if self.max_train_batches and step_idx >= self.max_train_batches:
                break

        avg_loss = total_loss / max(total_seen, 1)
        acc = total_correct / max(total_seen, 1)
        current_lr = self.optimizer.param_groups[0]["lr"]
        epoch_metrics: Dict[str, Any] = {"loss": avg_loss, "acc": acc, "lr": current_lr}
        if last_domain_name is not None:
            epoch_metrics["domain_name"] = last_domain_name
        for name, count in domain_counts.items():
            epoch_metrics[f"domain_count/{name}"] = count
        return epoch_metrics

    def validate(self, epoch: int) -> Dict[str, float]:
        """Run validation for one epoch."""
        del epoch
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_seen = 0
        with torch.no_grad():
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
        metrics: Dict[str, float] = {
            "loss": total_loss / max(total_seen, 1),
            "acc": total_correct / max(total_seen, 1),
        }
        try:
            extra = self.objective.validate(self.model, self.val_loader)
        except AttributeError:
            extra = {}
        if isinstance(extra, dict):
            metrics.update(extra)
        return metrics

    def _train_step(self, batch: Any) -> Dict[str, Any]:
        """Run a single training step."""
        batch = _to_device(batch, self.device)
        batch = self.objective.preprocess_batch(batch, self.model)

        self.optimizer.zero_grad(set_to_none=True)
        loss, metrics = self.objective.loss(self.model, batch)
        loss.backward()
        self.optimizer.step()

        metrics["lr"] = self.optimizer.param_groups[0]["lr"]
        return metrics

    def _save_checkpoint(self, name: str, metrics: Dict[str, Any], epoch: int | None) -> str:
        """Save a model checkpoint."""
        epoch_value = int(epoch if epoch is not None else metrics.get("epoch", 0))
        run_dir = get_run_dir(self.cfg)
        ensure_dir(run_dir)
        ckpt_path = f"{run_dir}/{name}.pt"
        state = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
            "metrics": metrics,
            "epoch": epoch_value,
            "global_step": int(self.global_step),
            "best_metric": float(self.best_metric),
            "best_epoch": int(self.best_epoch),
            "es_state": {
                "enabled": bool(self.es_enabled),
                "best_metric": float(self.es_best_metric),
                "bad_epochs": int(self.es_bad_epochs),
                "patience": int(self.es_patience),
            },
            "wandb_run_id": self.wandb_run_id,
        }
        torch.save(state, ckpt_path)
        return ckpt_path

    def _steps_per_epoch(self) -> int:
        full_steps = len(self.train_loader)
        if self.max_train_batches:
            return min(full_steps, int(self.max_train_batches))
        return full_steps


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
