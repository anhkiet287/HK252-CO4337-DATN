"""Trainer with pluggable objectives (ERM/PGD/REx/GroupDRO/GroupDRO++)."""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader

from ardg.attacks.attack_suite import build_eval_suite
from ardg.evaluation.evaluator import Evaluator
from ardg.evaluation.summary import summarize_suite
from ardg.training.losses import compute_loss
from ardg.training.objectives import build_objective
from ardg.training.objectives.multi_attack_erm import resolve_multi_attack_train_cfg
from ardg.utils.batch import move_to_device, unpack_xy
from ardg.utils.artifacts import update_artifact_manifest
from ardg.utils.logging import log_metrics
from ardg.utils.paths import (
    ensure_dir,
    find_checkpoint,
    get_checkpoint_path,
    get_legacy_train_summary_path,
    get_run_dir,
    get_train_summary_path,
    initialize_run_layout,
)
from ardg.utils.precision import PrecisionController
from ardg.utils.run_metadata import update_run_manifest


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
        self.run_dir = get_run_dir(self.cfg)
        initialize_run_layout(self.run_dir)
        
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

        self.precision = PrecisionController.from_config(cfg, device=str(self.device))

        # build loss function 
        self.objective = build_objective(cfg, model)
        self.objective.set_precision_controller(self.precision)
        self.val_suite = _build_val_suite(cfg, model)
        self.val_evaluator = Evaluator(
            model,
            val_loader,
            device=str(self.device),
            attack_suite=self.val_suite,
            max_batches=int(self.max_val_batches or 0),
            precision=self.precision,
        )
        self.best_metric = float("-inf")
        self.best_metric_name = "acc_clean"
        self.best_epoch = 0
        self.best_vector: Optional[tuple[float, ...]] = None
        sel_cfg = cfg.get("train", {}).get("selection", {})
        self.ckpt_eps = float(sel_cfg.get("eps", 1e-6))
        self.best_ckpt_path: Optional[str] = None
        self.best_worst_metric = float("-inf")
        self.best_worst_path: Optional[str] = None
        self.best_avg_metric = float("-inf")
        self.best_avg_path: Optional[str] = None
        self.train_mode = str(cfg.get("train", {}).get("mode", "")).lower()
        self._groupdro_q_plot_warning_emitted = False

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
        self.logger.info("Precision runtime: %s", self.precision.describe())
        update_run_manifest(
            self.run_dir,
            {
                "runtime": {
                    "precision": self.precision.as_dict(),
                }
            },
        )

    def train(self) -> Dict[str, str]:
        """Run the full training loop."""
        epochs = self.cfg["train"]["epochs"]
        last_ckpt_path = ""
        completed_epoch = max(self.start_epoch - 1, 0)

        # resume previous training if available
        if self.start_epoch > epochs:
            self.logger.info(
                "Resume epoch (%s) is beyond configured epochs (%s). Skipping training.",
                self.start_epoch,
                epochs,
            )
            run_dir = Path(self.run_dir)
            last_path = find_checkpoint(str(run_dir), names=("last", "best")) or str(get_checkpoint_path(str(run_dir), "last"))
            best_path = self.best_ckpt_path or find_checkpoint(str(run_dir), names=("best", "last")) or last_path
            return {"last": last_path, "best": best_path}

        # training loop
        try:
            self.objective.on_train_start({"train": self.train_loader, "val": self.val_loader})
        except AttributeError:
            pass

        for epoch in range(self.start_epoch, epochs + 1):
            completed_epoch = epoch
            self.logger.info("Starting epoch %s", epoch)
            start = time.perf_counter()

            train_metrics = self.train_one_epoch(epoch) # loss, acc, lr
            train_metrics["time_sec"] = time.perf_counter() - start # compute time
            train_metrics["device"] = str(self.device)
            log_metrics(
                self.logger,
                train_metrics,
                self.global_step,
                "train",
                epoch=epoch,
                log_by_epoch=True,
            )

            val_metrics = self.validate(epoch) # prefixed val metrics
            val_metrics["device"] = str(self.device)
            log_metrics(
                self.logger,
                {k.replace("val/", ""): v for k, v in val_metrics.items()},
                self.global_step,
                "val",
                epoch=epoch,
                log_by_epoch=True,
            )
            if self.train_mode in {"groupdro", "group_dro"}:
                groupdro_val_metrics: Dict[str, Any] = {}
                if "val/acc_avg" in val_metrics:
                    groupdro_val_metrics["val_overall_acc"] = float(val_metrics["val/acc_avg"])
                if "val/acc_worst" in val_metrics:
                    groupdro_val_metrics["val_worst_group_acc"] = float(val_metrics["val/acc_worst"])
                if "val/worst_domain" in val_metrics:
                    groupdro_val_metrics["val_worst_domain"] = str(val_metrics["val/worst_domain"])
                if groupdro_val_metrics:
                    log_metrics(
                        self.logger,
                        groupdro_val_metrics,
                        self.global_step,
                        "groupdro",
                        epoch=epoch,
                        log_by_epoch=True,
                    )
                self._maybe_log_groupdro_q_trajectory(epoch)
            val_acc = float(val_metrics.get("val/acc_clean", val_metrics.get("val/acc", 0.0))) # cache val acc for select best model 
            selection_names = _resolve_selection_names(self.cfg, val_metrics)
            ckpt_vector = tuple(float(val_metrics.get(n, float("-inf"))) for n in selection_names)
            ckpt_metric_name = selection_names[0] if selection_names else "acc_clean"
            ckpt_metric = ckpt_vector[0] if ckpt_vector else val_acc

            try:
                self.objective.on_epoch_end(epoch, {"train": self.train_loader, "val": self.val_loader})
            except AttributeError:
                pass
            
            # select best model 
            is_better = self._is_better_checkpoint(ckpt_vector, self.best_vector, self.ckpt_eps)
            if is_better:
                self.best_metric = ckpt_metric
                self.best_metric_name = ckpt_metric_name
                self.best_epoch = epoch
                self.best_vector = tuple(ckpt_vector)
                self.best_ckpt_path = self._save_checkpoint(
                    "best",
                    {
                        "epoch": epoch,
                        "val_acc_clean": val_acc,
                        "selection_metric": ckpt_metric,
                        "selection_metric_name": ckpt_metric_name,
                        "val_loss_clean": float(
                            val_metrics.get(
                                "val/loss_clean",
                                val_metrics.get("loss_clean", val_metrics.get("val/loss_adv", val_metrics.get("loss_adv", 0.0))),
                            )
                        ),
                        "selection_names": selection_names,
                        "selection_vector": ckpt_vector,
                        "val_metrics": dict(val_metrics),
                    },
                    epoch,
                )
                log_metrics(
                    self.logger,
                    {
                        "metric_name": ckpt_metric_name,
                        "epoch": epoch,
                        "score": ckpt_metric,
                    },
                    self.global_step,
                    "best",
                    epoch=epoch,
                    log_by_epoch=True,
                )

            last_ckpt_path = self._save_checkpoint(
                "last",
                {
                    "epoch": epoch,
                    "val_acc_clean": val_acc,
                    "selection_metric": ckpt_metric,
                    "selection_metric_name": ckpt_metric_name,
                    "val_loss_clean": float(
                        val_metrics.get(
                            "val/loss_clean",
                            val_metrics.get("loss_clean", val_metrics.get("val/loss_adv", val_metrics.get("loss_adv", 0.0))),
                        )
                    ),
                    "selection_names": selection_names,
                    "selection_vector": ckpt_vector,
                    "val_metrics": dict(val_metrics),
                },
                epoch,
            )

            if self.train_mode in {"multi_attack_erm", "multi-attack-erm", "multi_attack", "groupdro", "group_dro"}:
                if "val/acc_worst" in val_metrics and val_metrics["val/acc_worst"] > self.best_worst_metric + self.ckpt_eps:
                    self.best_worst_metric = float(val_metrics["val/acc_worst"])
                    self.best_worst_path = self._save_checkpoint(
                        "best_worst",
                        {
                            "epoch": epoch,
                            "selection_metric": self.best_worst_metric,
                            "selection_metric_name": "val/acc_worst",
                            "val_acc_clean": val_acc,
                            "selection_names": selection_names,
                            "selection_vector": ckpt_vector,
                            "val_metrics": dict(val_metrics),
                        },
                        epoch,
                    )
                if "val/acc_avg" in val_metrics and val_metrics["val/acc_avg"] > self.best_avg_metric + self.ckpt_eps:
                    self.best_avg_metric = float(val_metrics["val/acc_avg"])
                    self.best_avg_path = self._save_checkpoint(
                        "best_avg",
                        {
                            "epoch": epoch,
                            "selection_metric": self.best_avg_metric,
                            "selection_metric_name": "val/acc_avg",
                            "val_acc_clean": val_acc,
                            "selection_names": selection_names,
                            "selection_vector": ckpt_vector,
                            "val_metrics": dict(val_metrics),
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
        self._maybe_log_groupdro_q_trajectory(completed_epoch, force=True)
        best_path = self.best_ckpt_path or last_ckpt_path
        summary_path = self._write_train_summary(
            {
                "epochs_configured": int(epochs),
                "last_epoch": int(completed_epoch),
                "best_epoch": int(self.best_epoch),
                "best_metric": float(self.best_metric),
                "best_metric_name": str(self.best_metric_name),
                "global_step": int(self.global_step),
                "best_checkpoint": best_path,
                "last_checkpoint": last_ckpt_path,
                "wandb_run_id": self.wandb_run_id,
            }
        )
        update_run_manifest(
            self.run_dir,
            {
                "training": {
                    "best_checkpoint": best_path,
                    "last_checkpoint": last_ckpt_path,
                    "best_epoch": int(self.best_epoch),
                    "best_metric": float(self.best_metric),
                    "best_metric_name": str(self.best_metric_name),
                    "summary_json": summary_path,
                    "legacy_summary_json": str(get_legacy_train_summary_path(self.run_dir)),
                }
            },
        )
        update_artifact_manifest(
            {
                "latest": {
                    str(self.cfg.get("_meta", {}).get("run_metadata", {}).get("backbone", "model")): {
                        "train": {
                            "run_name": self.cfg.get("_meta", {}).get("run_metadata", {}).get("run_name"),
                            "run_dir": self.run_dir,
                            "best_checkpoint": best_path,
                            "last_checkpoint": last_ckpt_path,
                            "train_summary_json": summary_path,
                            "resolved_config": self.cfg.get("_meta", {}).get("resolved_config_path"),
                            "wandb_run_id": self.wandb_run_id,
                            "wandb_url": self.cfg.get("_meta", {}).get("run_metadata", {}).get("wandb_url"),
                        }
                    }
                }
            }
        )
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

        if isinstance(checkpoint.get("objective"), dict):
            try:
                self.objective.load_state_dict(checkpoint["objective"])
            except Exception:
                self.logger.warning("Failed to load objective state; continuing without it.")

        if checkpoint.get("global_step") is not None:
            self.global_step = int(checkpoint["global_step"])
        else:
            self.global_step = resumed_epoch * self._steps_per_epoch()
        self.precision.load_state_dict(checkpoint.get("grad_scaler"))

        if checkpoint.get("best_metric") is not None:
            self.best_metric = float(checkpoint["best_metric"])
        else:
            self.best_metric = float(metrics.get("val_acc", self.best_metric))
        self.best_metric_name = str(checkpoint.get("best_metric_name", self.best_metric_name))
        self.best_epoch = int(checkpoint.get("best_epoch", resumed_epoch))
        if isinstance(checkpoint.get("best_vector"), (list, tuple)):
            self.best_vector = tuple(float(x) for x in checkpoint["best_vector"])
        elif checkpoint.get("best_metric") is not None:
            self.best_vector = (float(checkpoint["best_metric"]),)
        if checkpoint.get("best_worst_metric") is not None:
            self.best_worst_metric = float(checkpoint["best_worst_metric"])
        if checkpoint.get("best_avg_metric") is not None:
            self.best_avg_metric = float(checkpoint["best_avg_metric"])

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

    @staticmethod
    def _is_better_checkpoint(
        candidate: tuple[float, ...],
        current_best: Optional[tuple[float, ...]],
        eps: float,
    ) -> bool:
        if current_best is None:
            return True
        max_len = max(len(candidate), len(current_best))
        for idx in range(max_len):
            cand = candidate[idx] if idx < len(candidate) else float("-inf")
            best = current_best[idx] if idx < len(current_best) else float("-inf")
            if cand > best + eps:
                return True
            if cand < best - eps:
                return False
        return False

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
        group_counts: Dict[str, float] = {}
        extra_metric_sums: Dict[str, float] = {}
        extra_metric_weights: Dict[str, float] = {}
        extra_metric_last: Dict[str, Any] = {}
        last_domain_name: str | None = None
        loss_key, acc_key = self._train_metric_keys()

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
                _accumulate_named_counts(domain_counts, batch_domain_counts)
            batch_group_counts = metrics.get("group_batch_counts")
            if isinstance(batch_group_counts, dict):
                _accumulate_named_counts(group_counts, batch_group_counts)
            _accumulate_step_metrics(
                extra_metric_sums,
                extra_metric_weights,
                extra_metric_last,
                metrics,
            )
            if self.train_mode in {"groupdro", "group_dro"}:
                record_q_snapshot = getattr(self.objective, "record_q_snapshot", None)
                if callable(record_q_snapshot):
                    record_q_snapshot(self.global_step)
                groupdro_step_metrics = _extract_groupdro_step_metrics(metrics)
                if groupdro_step_metrics:
                    log_metrics(self.logger, groupdro_step_metrics, self.global_step, "train_step")

            if self.log_interval and step_idx % self.log_interval == 0:
                batch_metrics = {
                    loss_key: metrics["loss"],
                    acc_key: metrics["acc"],
                    "optim/lr": metrics["optim/lr"],
                }
                if "domain_name" in metrics:
                    batch_metrics["domain_name"] = metrics["domain_name"]
                batch_metrics.update(_extract_loggable_step_metrics(metrics))
                log_metrics(self.logger, batch_metrics, self.global_step, "train_batch")
            if self.max_train_batches and step_idx >= self.max_train_batches:
                break

        avg_loss = total_loss / max(total_seen, 1)
        acc = total_correct / max(total_seen, 1)
        current_lr = self.optimizer.param_groups[0]["lr"]
        epoch_metrics: Dict[str, Any] = {loss_key: avg_loss, acc_key: acc, "optim/lr": current_lr}
        for name, total in extra_metric_sums.items():
            weight = max(extra_metric_weights.get(name, 0.0), 1.0)
            epoch_metrics[name] = total / weight
        epoch_metrics.update(extra_metric_last)
        if last_domain_name is not None:
            epoch_metrics["domain_name"] = last_domain_name
        for name, count in domain_counts.items():
            epoch_metrics[f"domain_count/{name}"] = count
        for name, count in group_counts.items():
            epoch_metrics[f"group_count/{name}"] = count
        return epoch_metrics

    def validate(self, epoch: int) -> Dict[str, float]:
        """Run validation for one epoch."""
        del epoch
        per_domain = {}
        if self.val_suite:
            per_domain.update(self.val_evaluator.evaluate_suite(self.val_suite))
            # ensure clean present for comparability
            per_domain.setdefault("clean", self.val_evaluator.evaluate_clean())
        else:
            per_domain["clean"] = self.val_evaluator.evaluate_clean()

        summary = summarize_suite(per_domain, prefix="val")
        metrics: Dict[str, float] = {k: v for k, v in summary.items() if k.startswith("val/")}
        try:
            extra = self.objective.validate(self.model, self.val_loader)
        except AttributeError:
            extra = {}
        if isinstance(extra, dict):
            metrics.update({f"val/{k}" if not k.startswith("val/") else k: v for k, v in extra.items()})
        return metrics

    def _maybe_log_groupdro_q_trajectory(self, epoch: int, *, force: bool = False) -> None:
        if self.train_mode not in {"groupdro", "group_dro"}:
            return
        should_log = getattr(self.objective, "should_log_q_trajectory", None)
        build_figure = getattr(self.objective, "build_q_trajectory_figure", None)
        if not callable(should_log) or not callable(build_figure):
            return
        if not should_log(epoch, force=force):
            return

        fig = build_figure()
        if fig is None:
            if not self._groupdro_q_plot_warning_emitted:
                self.logger.warning(
                    "Skipping GroupDRO q trajectory figure because matplotlib is not available."
                )
                self._groupdro_q_plot_warning_emitted = True
            return

        figure_path = Path(self.run_dir) / "artifacts" / "groupdro_q_trajectory.png"
        ensure_dir(str(figure_path.parent))
        try:
            fig.savefig(figure_path, dpi=160)
            try:
                import wandb  # type: ignore
            except ImportError:
                wandb = None
            if wandb is not None and getattr(wandb, "run", None) is not None:
                wandb.log({"groupdro/q_trajectory": wandb.Image(fig)}, step=self.global_step)
        finally:
            try:
                import matplotlib.pyplot as plt
            except ImportError:
                plt = None
            if plt is not None:
                plt.close(fig)

    def _train_step(self, batch: Any) -> Dict[str, Any]:
        """Run a single training step."""
        batch = move_to_device(batch, self.device)
        batch = self.objective.preprocess_batch(batch, self.model)

        self.optimizer.zero_grad(set_to_none=True)
        loss, metrics = self.objective.compute_loss(self.model, batch)
        self.precision.backward(loss)
        self.precision.step_optimizer(self.optimizer)
        self.precision.update()

        metrics["optim/lr"] = self.optimizer.param_groups[0]["lr"]
        return metrics

    def _save_checkpoint(self, name: str, metrics: Dict[str, Any], epoch: int | None) -> str:
        """Save a model checkpoint."""
        epoch_value = int(epoch if epoch is not None else metrics.get("epoch", 0))
        run_dir = self.run_dir
        ensure_dir(run_dir)
        ckpt_path = str(get_checkpoint_path(run_dir, name))
        selection_names = metrics.get("selection_names")
        state = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
            "objective": self.objective.state_dict() if hasattr(self.objective, "state_dict") else {},
            "precision": self.precision.as_dict(),
            "grad_scaler": self.precision.state_dict(),
            "metrics": metrics,
            "selection_names": selection_names,
            "selection_vector": metrics.get("selection_vector"),
            "epoch": epoch_value,
            "global_step": int(self.global_step),
            "best_metric": float(self.best_metric),
            "best_metric_name": str(self.best_metric_name),
            "best_epoch": int(self.best_epoch),
            "best_vector": list(self.best_vector) if self.best_vector is not None else None,
            "best_worst_metric": float(self.best_worst_metric),
            "best_avg_metric": float(self.best_avg_metric),
            "es_state": {
                "enabled": bool(self.es_enabled),
                "best_metric": float(self.es_best_metric),
                "bad_epochs": int(self.es_bad_epochs),
                "patience": int(self.es_patience),
            },
            "wandb_run_id": self.wandb_run_id,
        }
        torch.save(state, ckpt_path)
        update_run_manifest(
            run_dir,
            {
                "checkpoints": {
                    name: ckpt_path,
                }
            },
        )
        return ckpt_path

    def _write_train_summary(self, payload: Dict[str, Any]) -> str:
        run_dir = Path(self.run_dir)
        ensure_dir(str(run_dir))
        summary_path = get_train_summary_path(self.run_dir)
        summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        get_legacy_train_summary_path(self.run_dir).write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return str(summary_path)

    def _steps_per_epoch(self) -> int:
        full_steps = len(self.train_loader)
        if self.max_train_batches:
            return min(full_steps, int(self.max_train_batches))
        return full_steps

    def _train_metric_keys(self) -> tuple[str, str]:
        mode = str(self.cfg.get("train", {}).get("mode", "")).lower()
        if mode in {"erm", "clean"}:
            return "loss_clean", "acc_clean"
        return "loss_adv", "acc_adv"

def _normalize_es_metric_key(metric: Any) -> str:
    key = str(metric or "acc_clean").strip().lower().replace("val/", "").replace("val_", "")
    if key == "acc":
        return "acc_clean"
    if key == "loss":
        return "loss_clean"
    return key


def _resolve_selection_names(cfg: Dict[str, Any], metrics: Dict[str, float]) -> list[str]:
    """Choose a selection vector based on config (train.selection.vector) or available metrics (prefixed)."""
    sel_cfg = cfg.get("train", {}).get("selection", {})
    explicit = sel_cfg.get("vector")
    if isinstance(explicit, (list, tuple)):
        names = [str(x) for x in explicit]
    elif isinstance(explicit, str) and explicit.strip():
        names = [part.strip() for part in explicit.split(",") if part.strip()]
    else:
        names = []

    if names:
        return names

    train_cfg = cfg.get("train", {})
    mode = str(train_cfg.get("mode", "")).lower()
    if mode in {"groupdro", "group_dro"}:
        groupdro_metric = str(train_cfg.get("groupdro", {}).get("selection_metric", "")).strip()
        if groupdro_metric:
            names = [groupdro_metric]
            for fallback in ("val/acc_avg",):
                if fallback in metrics and fallback not in names:
                    names.append(fallback)
            return names

    # default heuristic
    if "val/acc_worst" in metrics and "val/acc_avg" in metrics:
        return ["val/acc_worst", "val/acc_avg"]
    if "val/acc_clean" in metrics:
        return ["val/acc_clean"]
    if "acc_worst" in metrics and "acc_avg" in metrics:
        return ["acc_worst", "acc_avg"]
    if "acc_clean" in metrics:
        return ["acc_clean"]
    if "acc" in metrics:
        return ["acc"]
    # fallback: first numeric key
    for key, val in metrics.items():
        try:
            float(val)
            return [key]
        except Exception:
            continue
    return []


def _normalize_ckpt_token(token: str) -> str:
    normalized = token.strip().lower().replace("val/", "").replace("val_", "")
    alias = {
        "worst": "worst_acc",
        "acc_worst": "worst_acc",
        "worst_acc": "worst_acc",
        "avg": "avg_acc",
        "acc_avg": "avg_acc",
        "avg_acc": "avg_acc",
        "clean": "clean_acc",
        "acc_clean": "clean_acc",
        "clean_acc": "clean_acc",
    }
    return alias.get(normalized, normalized)


def _resolve_ckpt_metric_value(token: str, metrics: Dict[str, float]) -> tuple[str, float]:
    # accept prefixed tokens
    if token.startswith("val/"):
        name, value = _resolve_ckpt_metric_value(token.replace("val/", ""), {k.replace("val/",""):v for k,v in metrics.items()})
        return f"val/{name}", value
    if token == "worst_acc":
        value = metrics.get("acc_worst", metrics.get("acc", float("-inf")))
        return "worst_acc", float(value)
    if token == "avg_acc":
        value = metrics.get("acc_avg", metrics.get("acc", float("-inf")))
        return "avg_acc", float(value)
    if token == "clean_acc":
        value = metrics.get("acc_clean", metrics.get("acc", float("-inf")))
        return "clean_acc", float(value)
    # Fallback: direct metric key in val_metrics
    return token, float(metrics.get(token, float("-inf")))


def _build_val_suite(cfg: Dict[str, Any], model: Any) -> Dict[str, Any]:
    attack_cfg = cfg.get("attack", {})
    val_specs = attack_cfg.get("val_suite") or attack_cfg.get("val") or None
    # Backward compatibility: if no val suite, optionally reuse eval suite for validation
    if val_specs is None:
        val_specs = attack_cfg.get("eval_suite")
    if not val_specs:
        # Provide a minimal default suite: clean + pgd20 using shared eps if present
        shared_eps = (
            attack_cfg.get("eps")
            or attack_cfg.get("step_size")
            or attack_cfg.get("epsilon")
            or 8.0 / 255.0
        )
        step_size = attack_cfg.get("step_size") or (shared_eps / 4.0)
        val_specs = [
            {"label": "clean", "type": "clean"},
            {
                "label": "pgd20",
                "type": "pgd",
                "eps": float(shared_eps),
                "step_size": float(step_size),
                "num_steps": 20,
                "restarts": 1,
                "loss": "ce",
                "random_start": True,
            },
        ]
    # reuse eval suite builder by temporarily injecting specs
    tmp_cfg = dict(cfg)
    tmp_attack = dict(attack_cfg)
    tmp_attack["eval_suite"] = val_specs
    tmp_cfg["attack"] = tmp_attack
    return build_eval_suite(tmp_cfg, model)


def _accumulate_named_counts(target: Dict[str, float], counts: Dict[str, Any]) -> None:
    for name, count in counts.items():
        target[str(name)] = target.get(str(name), 0.0) + float(count)


def _is_numeric_metric(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _should_aggregate_step_metric(name: str, value: Any) -> bool:
    if name in {"loss", "acc", "correct", "batch_size", "optim/lr"}:
        return False
    if name in {
        "domain_name",
        "worst_group",
        "worst_group_mode",
        "worst_group_by_loss",
        "worst_group_by_acc",
        "worst_group_by_normalized_loss",
    }:
        return False
    if name.endswith("_batch_counts"):
        return False
    return _is_numeric_metric(value)


def _accumulate_step_metrics(
    sums: Dict[str, float],
    weights: Dict[str, float],
    latest: Dict[str, Any],
    metrics: Dict[str, Any],
) -> None:
    batch_size = float(metrics.get("batch_size", 1) or 1)
    for key, value in metrics.items():
        if _should_aggregate_step_metric(key, value):
            sums[key] = sums.get(key, 0.0) + float(value) * batch_size
            weights[key] = weights.get(key, 0.0) + batch_size
        elif key in {
            "domain_name",
            "worst_group",
            "worst_group_mode",
            "worst_group_by_loss",
            "worst_group_by_acc",
            "worst_group_by_normalized_loss",
        } and value is not None:
            latest[key] = value


def _extract_loggable_step_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    logged: Dict[str, Any] = {}
    for key, value in metrics.items():
        if key in {"loss", "acc", "correct", "batch_size", "optim/lr", "domain_name"}:
            continue
        if key.endswith("_batch_counts"):
            continue
        if _is_numeric_metric(value) or isinstance(value, str):
            logged[key] = value
    return logged


def _extract_groupdro_step_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    logged: Dict[str, Any] = {}
    allowed = {"q_g", "q_max", "q_min", "group_id", "loss_group", "domain_name"}
    for key, value in metrics.items():
        if key.startswith("q/") or key in allowed:
            if _is_numeric_metric(value) or isinstance(value, str):
                logged[key] = value
    return logged
