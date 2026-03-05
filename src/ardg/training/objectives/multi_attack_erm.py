"""Multi-attack ERM objective (attacks-as-domains)."""

from __future__ import annotations

import random
from typing import Any, Dict, List, Tuple

import torch

from ardg.attacks.attack_suite import build_attack
from ardg.training.losses import compute_loss
from ardg.training.objectives.base import Objective
from ardg.utils.batch import as_xy_dict


def _float_close(a: float, b: float, tol: float = 1e-12) -> bool:
    return abs(float(a) - float(b)) <= tol


def resolve_multi_attack_train_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve multi-attack train config with backward-compatible alias.

    Canonical key is ``train.multi_attack``.
    Alias ``train.multi_domain`` is supported with the same schema.
    """
    train_cfg = cfg.get("train", {})
    ma = train_cfg.get("multi_attack")
    md = train_cfg.get("multi_domain")
    has_ma = isinstance(ma, dict)
    has_md = isinstance(md, dict)
    if has_ma and has_md:
        raise ValueError(
            "Both train.multi_attack and train.multi_domain are set. "
            "Please keep only one to avoid ambiguous configuration."
        )
    if has_md:
        return dict(md)
    if has_ma:
        return dict(ma)
    return {}


class MultiAttackERM(Objective):
    """ERM on adversarial domains with configurable per-batch strategy."""

    _SUPPORTED_STRATEGIES = {"per_batch", "split_batch", "all_domains"}
    _SUPPORTED_TYPES = {"clean", "fgsm", "fgsm_rs", "pgd", "pgd_ce", "pgd_dlr"}

    def __init__(self, cfg: Dict[str, Any], model: Any) -> None:
        self.cfg = cfg
        self.model = model
        train_cfg = resolve_multi_attack_train_cfg(cfg)
        self.strategy = str(train_cfg.get("strategy", "per_batch")).lower()
        if self.strategy not in self._SUPPORTED_STRATEGIES:
            raise ValueError(
                f"Unsupported train.multi_attack.strategy={self.strategy!r}. "
                f"Use one of {sorted(self._SUPPORTED_STRATEGIES)}."
            )

        self.aggregation = str(train_cfg.get("aggregation", "mean")).lower()
        if self.aggregation != "mean":
            raise ValueError(
                f"Unsupported train.multi_attack.aggregation={self.aggregation!r}. "
                "Only 'mean' is supported."
            )

        self.include_clean = bool(train_cfg.get("include_clean", True))
        self.shared_norm, self.shared_eps, self.domains = self._parse_domains(cfg)
        seed = int(cfg.get("experiment", {}).get("seed", 42))
        self.rng = random.Random(seed)

        dataset_name = str(cfg["dataset"]["name"])
        self.attacks = {
            d["name"]: build_attack(
                d,
                model,
                dataset_name=dataset_name,
                shared_norm=self.shared_norm,
                shared_eps=self.shared_eps,
            )
            for d in self.domains
        }

        self.probe_cfg = dict(cfg.get("val", {}).get("probe", {}))
        self.probe_enabled = bool(self.probe_cfg.get("enabled", False))
        self.val_max_batches = int(train_cfg.get("val_max_batches", self.probe_cfg.get("max_batches", 10)))
        if self.val_max_batches < 0:
            self.val_max_batches = 0
        self.probe_attack = None
        if self.probe_enabled:
            self.probe_attack = self._build_probe_attack(model)

    def _parse_domains(self, cfg: Dict[str, Any]) -> Tuple[str, float, List[Dict[str, Any]]]:
        multi_cfg = cfg.get("attack", {}).get("multi_train", {})
        if not isinstance(multi_cfg, dict) or not multi_cfg:
            raise ValueError("attack.multi_train config is required for train.mode=multi_attack_erm.")
        if "eps" not in multi_cfg:
            raise ValueError("attack.multi_train.eps is required.")
        shared_eps = float(multi_cfg["eps"])
        shared_norm = str(multi_cfg.get("norm", "Linf"))
        raw_domains = multi_cfg.get("domains", [])
        if not isinstance(raw_domains, list):
            raise ValueError("attack.multi_train.domains must be a list.")

        normalized: List[Dict[str, Any]] = []
        clean_seen = False
        for raw in raw_domains:
            if not isinstance(raw, dict):
                raise ValueError(f"Each multi-attack domain must be a mapping, got: {type(raw)!r}")
            domain = dict(raw)
            domain_type = str(domain.get("type", domain.get("name", ""))).strip().lower()
            if not domain_type:
                raise ValueError(f"Domain is missing 'type' (or 'name'): {raw}")
            if domain_type not in self._SUPPORTED_TYPES:
                raise ValueError(
                    f"Unsupported domain type={domain_type!r}. "
                    f"Use one of {sorted(self._SUPPORTED_TYPES)}."
                )
            domain_name = str(domain.get("name", domain_type)).strip()
            if not domain_name:
                raise ValueError(f"Invalid empty domain name in spec: {raw}")

            eps_override = domain.get("eps")
            if eps_override is not None and not _float_close(float(eps_override), shared_eps):
                raise ValueError(
                    f"Domain '{domain_name}' has eps={eps_override}, "
                    f"but attack.multi_train.eps={shared_eps}."
                )
            norm_override = domain.get("norm")
            if norm_override is not None and str(norm_override).lower() != shared_norm.lower():
                raise ValueError(
                    f"Domain '{domain_name}' has norm={norm_override}, "
                    f"but attack.multi_train.norm={shared_norm}."
                )

            domain["type"] = domain_type
            domain["name"] = domain_name
            if domain_type == "clean":
                if clean_seen:
                    continue
                clean_seen = True
            normalized.append(domain)

        if self.include_clean and not clean_seen:
            normalized.insert(0, {"name": "clean", "type": "clean"})

        if not normalized:
            raise ValueError("attack.multi_train.domains is empty after normalization.")

        names = [d["name"] for d in normalized]
        if len(set(names)) != len(names):
            raise ValueError(f"Duplicate domain names in attack.multi_train.domains: {names}")
        return shared_norm, shared_eps, normalized

    def _build_probe_attack(self, model: Any) -> Any:
        probe_type = str(self.probe_cfg.get("type", "pgd")).lower()
        if probe_type != "pgd":
            raise ValueError(
                f"Unsupported val.probe.type={probe_type!r}. Only 'pgd' is supported."
            )
        eps = float(self.probe_cfg.get("eps", self.shared_eps))
        alpha = float(self.probe_cfg.get("alpha", self.probe_cfg.get("step_size", 2.0 / 255.0)))
        steps = int(self.probe_cfg.get("steps", self.probe_cfg.get("num_steps", 20)))
        restarts = int(self.probe_cfg.get("restarts", 1))
        norm = str(self.probe_cfg.get("norm", self.shared_norm))
        loss_name = str(self.probe_cfg.get("loss", "ce")).lower()
        spec = {
            "name": "probe_pgd",
            "type": "pgd",
            "eps": eps,
            "alpha": alpha,
            "step_size": alpha,
            "steps": steps,
            "num_steps": steps,
            "restarts": restarts,
            "norm": norm,
            "loss": loss_name,
            "random_start": bool(self.probe_cfg.get("random_start", True)),
        }
        return build_attack(
            spec,
            model,
            dataset_name=str(self.cfg["dataset"]["name"]),
            shared_norm=norm,
            shared_eps=eps,
        )

    def _attack_batch(
        self, attack: Any, images: torch.Tensor, labels: torch.Tensor, model: Any
    ) -> torch.Tensor:
        model.eval()
        adv = attack(images, labels).detach()
        model.train()
        return adv

    def preprocess_batch(self, batch: Any, model: Any) -> Any:
        data = as_xy_dict(batch)
        images = data["x"]
        labels = data["y"]
        batch_size = int(labels.size(0))

        if self.strategy == "per_batch":
            idx = self.rng.randrange(len(self.domains))
            name = self.domains[idx]["name"]
            adv = self._attack_batch(self.attacks[name], images, labels, model)
            data["x"] = adv
            data["domain_name"] = name
            data["domain_batch_counts"] = {name: batch_size}
            return data

        if self.strategy == "split_batch":
            domain_ids = torch.tensor(
                [self.rng.randrange(len(self.domains)) for _ in range(batch_size)],
                device=labels.device,
                dtype=torch.long,
            )
            adv = images.detach().clone()
            batch_counts: Dict[str, int] = {}
            model.eval()
            for idx, domain in enumerate(self.domains):
                mask = domain_ids == idx
                if not mask.any():
                    continue
                name = domain["name"]
                count = int(mask.sum().item())
                adv[mask] = self.attacks[name](images[mask], labels[mask]).detach()
                batch_counts[name] = count
            model.train()
            data["x"] = adv
            data["domain_ids"] = domain_ids
            data["domain_name"] = "mixed"
            data["domain_batch_counts"] = batch_counts
            return data

        if self.strategy == "all_domains":
            x_domains: List[torch.Tensor] = []
            domain_names: List[str] = []
            batch_counts: Dict[str, int] = {}
            model.eval()
            for domain in self.domains:
                name = domain["name"]
                x_domains.append(self.attacks[name](images, labels).detach())
                domain_names.append(name)
                batch_counts[name] = batch_size
            model.train()
            data["x_domains"] = x_domains
            data["domain_names"] = domain_names
            data["domain_name"] = "all_domains"
            data["domain_batch_counts"] = batch_counts
            return data

        raise ValueError(f"Unsupported strategy: {self.strategy}")

    def loss(self, model: Any, batch: Any) -> Tuple[torch.Tensor, Dict[str, Any]]:
        data = as_xy_dict(batch)
        labels = data["y"]
        batch_size = int(labels.size(0))

        domain_counts = data.get("domain_batch_counts", {})
        metrics: Dict[str, Any] = {"batch_size": batch_size, "domain_batch_counts": domain_counts}
        if "domain_name" in data:
            metrics["domain_name"] = data["domain_name"]

        if "x_domains" in data:
            losses: List[torch.Tensor] = []
            acc_values: List[float] = []
            for x_dom, name in zip(data["x_domains"], data.get("domain_names", [])):
                logits_dom = model(x_dom)
                loss_dom = compute_loss(logits_dom, labels)
                acc_dom = float((logits_dom.argmax(dim=1) == labels).float().mean().item())
                losses.append(loss_dom)
                acc_values.append(acc_dom)
                metrics[f"acc_{name}"] = acc_dom
            loss = torch.stack(losses).mean()
            acc = float(sum(acc_values) / max(len(acc_values), 1))
            metrics.update(
                {
                    "loss": float(loss.item()),
                    "acc": acc,
                    "correct": acc * batch_size,
                }
            )
            return loss, metrics

        images = data["x"]
        logits = model(images)
        loss = compute_loss(logits, labels)
        correct = float((logits.argmax(dim=1) == labels).sum().item())
        metrics.update(
            {
                "loss": float(loss.item()),
                "acc": correct / max(batch_size, 1),
                "correct": correct,
            }
        )
        return loss, metrics

    def validate(self, model: Any, loader: Any) -> Dict[str, float]:
        metrics: Dict[str, float] = {}

        device = next(model.parameters()).device
        domain_totals: Dict[str, Dict[str, float]] = {
            d["name"]: {"loss": 0.0, "correct": 0.0, "seen": 0.0} for d in self.domains
        }

        model.eval()
        for step_idx, batch in enumerate(loader, start=1):
            data = as_xy_dict(batch)
            images = data["x"].to(device)
            labels = data["y"].to(device)

            for domain in self.domains:
                name = str(domain["name"])
                if domain.get("type") == "clean":
                    attacked = images
                else:
                    with torch.enable_grad():
                        attacked = self.attacks[name](images, labels).detach()
                with torch.no_grad():
                    logits = model(attacked)
                    loss = compute_loss(logits, labels)

                seen = float(images.size(0))
                correct = float((logits.argmax(dim=1) == labels).sum().item())
                domain_totals[name]["loss"] += float(loss.item()) * seen
                domain_totals[name]["correct"] += correct
                domain_totals[name]["seen"] += seen

            if self.val_max_batches > 0 and step_idx >= self.val_max_batches:
                break

        acc_by_domain: Dict[str, float] = {}
        for name, totals in domain_totals.items():
            seen = max(float(totals["seen"]), 1.0)
            loss_avg = float(totals["loss"]) / seen
            acc_avg = float(totals["correct"]) / seen
            metrics[f"loss_{name}"] = loss_avg
            metrics[f"acc_{name}"] = acc_avg
            acc_by_domain[name] = acc_avg

        if acc_by_domain:
            metrics["acc_avg"] = float(sum(acc_by_domain.values()) / len(acc_by_domain))
            worst_domain = min(acc_by_domain, key=acc_by_domain.get)
            metrics["acc_worst"] = float(acc_by_domain[worst_domain])
            metrics["worst_domain"] = worst_domain
            if "clean" in acc_by_domain:
                metrics["acc_clean"] = float(acc_by_domain["clean"])

        if not self.probe_enabled or self.probe_attack is None:
            return metrics

        probe_max_batches = int(self.probe_cfg.get("max_batches", 10))
        if probe_max_batches <= 0:
            return metrics

        probe_total_loss = 0.0
        probe_total_correct = 0
        probe_total_seen = 0
        for step_idx, batch in enumerate(loader, start=1):
            data = as_xy_dict(batch)
            images = data["x"].to(device)
            labels = data["y"].to(device)
            with torch.enable_grad():
                adv = self.probe_attack(images, labels).detach()
            with torch.no_grad():
                logits = model(adv)
                loss = compute_loss(logits, labels)
            probe_total_loss += float(loss.item()) * images.size(0)
            probe_total_correct += int((logits.argmax(dim=1) == labels).sum().item())
            probe_total_seen += int(images.size(0))
            if step_idx >= probe_max_batches:
                break

        metrics["pgd20_probe_loss"] = probe_total_loss / max(probe_total_seen, 1)
        metrics["pgd20_probe_acc"] = probe_total_correct / max(probe_total_seen, 1)
        return metrics
