"""Training loop for ERM and PGD-AT."""

import logging
import time
from typing import Any, Dict, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader

from ardg.attacks.attack_suite import build_train_attack
from ardg.training.losses import compute_loss
from ardg.utils.logging import log_metrics
from ardg.utils.paths import ensure_dir, get_run_dir


class Trainer:
    """Trainer for ERM and PGD-AT training."""

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

        self.attack = build_train_attack(cfg, model) if cfg["train"]["mode"] == "pgd_at" else None

    def train(self) -> None:
        """Run the full training loop."""
        epochs = self.cfg["train"]["epochs"]
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
            if self.scheduler is not None:
                self.scheduler.step()
            self.logger.info("Finished epoch %s in %.2fs", epoch, train_metrics["time_sec"])

    def train_one_epoch(self, epoch: int) -> Dict[str, float]:
        """Run one training epoch."""
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
        return {"loss": avg_loss, "acc": acc}

    @torch.no_grad()
    def validate(self, epoch: int) -> Dict[str, float]:
        """Run validation for one epoch."""
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_seen = 0
        for step_idx, (images, labels) in enumerate(self.val_loader, start=1):
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
        images, labels = batch
        images = images.to(self.device)
        labels = labels.to(self.device)

        if self.attack is not None:
            images = self._make_adv_batch(images, labels)

        self.optimizer.zero_grad(set_to_none=True)
        logits = self.model(images)
        loss = compute_loss(logits, labels)
        loss.backward()
        self.optimizer.step()

        correct = (logits.argmax(dim=1) == labels).sum().item()
        return {
            "loss": float(loss.item()),
            "acc": correct / max(images.size(0), 1),
            "correct": correct,
            "batch_size": images.size(0),
            "lr": self.optimizer.param_groups[0]["lr"],
        }

    def _make_adv_batch(self, images: Any, labels: Any) -> Any:
        """Generate adversarial examples for PGD-AT."""
        self.model.eval()
        adv = self.attack(images, labels)
        self.model.train()
        return adv.detach()

    def _save_checkpoint(self, name: str, metrics: Dict[str, float]) -> str:
        """Save a model checkpoint."""
        run_dir = get_run_dir(self.cfg)
        ensure_dir(run_dir)
        ckpt_path = f"{run_dir}/{name}.pt"
        torch.save({"model": self.model.state_dict(), "metrics": metrics}, ckpt_path)
        return ckpt_path
