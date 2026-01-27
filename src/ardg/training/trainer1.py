"""Draft trainer with pluggable objectives (ERM/PGD/REx/GroupDRO/GroupDRO++)."""

import logging
import time
from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader

from ardg.training.losses import compute_loss
from ardg.training.objectives import build_objective
from ardg.utils.logging import log_metrics
from ardg.utils.paths import ensure_dir, get_run_dir


class Trainer1:
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
        """Initialize the trainer.

        Args:
            cfg: Configuration dictionary.
            model: Model to train.
            train_loader: Training data loader.
            val_loader: Validation data loader.
            device: Device string such as "cpu" or "cuda".
            logger: Logger instance.

        Raises:
            KeyError: If required config keys are missing.
        """
        self.cfg = cfg  # store config
        self.model = model  # load model
        self.train_loader = train_loader  # load training data
        self.val_loader = val_loader  # load validation data
        self.device = torch.device(device or "cpu")  # set device
        self.model.to(self.device)  # move model to device
        self.logger = logger or logging.getLogger(__name__)  # set up logger
        self.max_train_batches = cfg["train"].get("max_batches")  # max training batches
        self.max_val_batches = cfg["train"].get("max_val_batches")  # max validation batches
        self.log_interval = cfg["train"].get("log_interval", 50)  # logging interval
        self.global_step = 0  # initialize global step

        # Set up optimizer and scheduler
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

        # Objective (handles ERM/PGD/REx/GroupDRO/GroupDRO++)
        self.objective = build_objective(cfg, model)
        self.best_metric = float("-inf")
        self.best_epoch = 0
        self.best_ckpt_path: Optional[str] = None

    def train(self) -> Dict[str, str]:
        """Run the full training loop.

        Returns:
            Mapping with "last" and "best" checkpoint paths.

        Raises:
            RuntimeError: If no checkpoint is saved during training.
        """
        epochs = self.cfg["train"]["epochs"] # total number of epochs
        last_ckpt_path = "" # initialize last checkpoint path
        for epoch in range(1, epochs + 1): # loop over epochs
            self.logger.info("Starting epoch %s", epoch)
            start = time.perf_counter() # start timer

            # Run training for one epoch
            train_metrics = self.train_one_epoch(epoch)
            train_metrics["time_sec"] = time.perf_counter() - start
            train_metrics["device"] = str(self.device)
            log_metrics(self.logger, train_metrics, self.global_step, "train") # log training metrics

            # Run validation
            val_metrics = self.validate(epoch)
            val_metrics["device"] = str(self.device)
            log_metrics(self.logger, val_metrics, self.global_step, "val") # log validation metrics
            val_acc = float(val_metrics.get("acc", 0.0)) 

            # Objective hook (e.g., recluster for GroupDRO++)
            try:
                self.objective.on_epoch_end(epoch, {"train": self.train_loader, "val": self.val_loader})
            except AttributeError:
                pass

            # Save last checkpoint
            last_ckpt_path = self._save_checkpoint(
                "last",
                {
                    "epoch": epoch,
                    "val_acc": val_acc,
                    "val_loss": float(val_metrics.get("loss", 0.0)),
                },
            )

            # Save best checkpoint
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
                )
            if self.scheduler is not None:
                # Step scheduler with explicit epoch index so cosine decays from first epoch.
                self.scheduler.step()
            self.logger.info("Finished epoch %s in %.2fs", epoch, train_metrics["time_sec"])
        if not last_ckpt_path:
            raise RuntimeError("No checkpoint saved during training.")
        best_path = self.best_ckpt_path or last_ckpt_path
        return {"last": last_ckpt_path, "best": best_path}

    def train_one_epoch(self, epoch: int) -> Dict[str, float]:
        """Run one training epoch.

        Args:
            epoch: 1-based epoch index.

        Returns:
            Dictionary with averaged loss and accuracy.

        Raises:
            RuntimeError: If a training step fails.
        """
        self.model.train() # set model to training mode

        # Initialize metrics
        total_loss = 0.0
        total_correct = 0
        total_seen = 0

        for step_idx, batch in enumerate(self.train_loader, start=1): # loop over training batches
            self.global_step += 1 # increment global step
            metrics = self._train_step(batch) # perform training step
            total_loss += metrics["loss"] * metrics["batch_size"] # accumulate loss
            total_correct += metrics["correct"] # accumulate correct predictions
            total_seen += metrics["batch_size"] # accumulate seen samples

            if self.log_interval and step_idx % self.log_interval == 0: # log at intervals
                batch_metrics = {
                    "loss": metrics["loss"], 
                    "acc": metrics["acc"],
                    "lr": metrics["lr"],
                }
                log_metrics(self.logger, batch_metrics, self.global_step, "train_batch")
            if self.max_train_batches and step_idx >= self.max_train_batches:
                break
        avg_loss = total_loss / max(total_seen, 1) # compute average loss
        acc = total_correct / max(total_seen, 1) # compute accuracy
        # Log the last LR for epoch-level metrics
        current_lr = self.optimizer.param_groups[0]["lr"]
        return {"loss": avg_loss, "acc": acc, "lr": current_lr} 

    @torch.no_grad()
    def validate(self, epoch: int) -> Dict[str, float]:
        """Run validation for one epoch.

        Args:
            epoch: 1-based epoch index.

        Returns:
            Dictionary with averaged loss and accuracy.

        Raises:
            RuntimeError: If validation fails.
        """
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
        """Run a single training step.

        Args:
            batch: Batch tuple of images and labels.

        Returns:
            Dictionary of per-batch metrics.

        Raises:
            RuntimeError: If the optimization step fails.
        """
        # Move batch tensors to device and let objective handle preprocessing.
        batch = _to_device(batch, self.device)
        batch = self.objective.preprocess_batch(batch, self.model)

        self.optimizer.zero_grad(set_to_none=True) # zero gradients

        loss, metrics = self.objective.loss(self.model, batch) # forward & loss
        loss.backward() # backward pass
        self.optimizer.step() # optimization step

        metrics["lr"] = self.optimizer.param_groups[0]["lr"]
        return metrics

    def _save_checkpoint(self, name: str, metrics: Dict[str, float]) -> str:
        """Save a model checkpoint.

        Args:
            name: Checkpoint name (e.g. "last", "best").
            metrics: Metrics to store alongside the model state.

        Returns:
            Path to the saved checkpoint.

        Raises:
            RuntimeError: If saving the checkpoint fails.
        """
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
