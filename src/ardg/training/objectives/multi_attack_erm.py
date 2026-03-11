"""Multi-attack ERM objective (attacks-as-domains)."""

from __future__ import annotations

import random
from typing import Any, Dict, List, Sequence, Tuple

import torch

from ardg.attacks.attack_suite import build_attack
from ardg.training.losses import compute_loss
from ardg.training.objectives.base import Objective
from ardg.utils.batch import as_xy_dict

SUPPORTED_DOMAIN_TYPES = {"clean", "fgsm", "fgsm_rs", "pgd", "pgd_ce", "pgd_dlr", "cw", "deepfool"}
_SHARED_EPS_TYPES = {"fgsm", "fgsm_rs", "pgd", "pgd_ce", "pgd_dlr"}
_SHARED_NORM_TYPES = {"fgsm", "fgsm_rs", "pgd", "pgd_ce", "pgd_dlr"}


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


def _infer_shared_threat_model(raw_domains: Sequence[Dict[str, Any]]) -> Tuple[str | None, float | None]:
    eps_values: List[float] = []
    norm_values: List[str] = []
    for raw in raw_domains:
        if not isinstance(raw, dict):
            raise ValueError(f"Each train domain must be a mapping, got: {type(raw)!r}")
        domain_type = str(raw.get("type", raw.get("name", raw.get("label", "")))).strip().lower()
        if domain_type == "clean":
            continue
        eps = raw.get("eps") if domain_type in _SHARED_EPS_TYPES else None
        if eps is not None:
            eps_values.append(float(eps))
        norm = raw.get("norm") if domain_type in _SHARED_NORM_TYPES else None
        if norm is not None:
            norm_values.append(str(norm))

    shared_eps: float | None = None
    if eps_values and all(_float_close(eps_values[0], value) for value in eps_values[1:]):
        shared_eps = eps_values[0]

    shared_norm: str | None = None
    if norm_values and all(value.lower() == norm_values[0].lower() for value in norm_values[1:]):
        shared_norm = norm_values[0]
    if shared_norm is None and shared_eps is not None:
        shared_norm = "Linf"
    return shared_norm, shared_eps


def _normalize_domain_specs(
    raw_domains: Sequence[Dict[str, Any]],
    *,
    include_clean: bool,
    supported_types: Sequence[str],
    shared_norm: str | None = None,
    shared_eps: float | None = None,
) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    clean_seen = False
    supported = {str(name).lower() for name in supported_types}

    for raw in raw_domains:
        if not isinstance(raw, dict):
            raise ValueError(f"Each train domain must be a mapping, got: {type(raw)!r}")

        domain = dict(raw)
        domain_type = str(domain.get("type", domain.get("name", domain.get("label", "")))).strip().lower()
        if not domain_type:
            raise ValueError(f"Domain is missing 'type' (or 'name'/'label'): {raw}")
        if domain_type not in supported:
            raise ValueError(
                f"Unsupported domain type={domain_type!r}. Use one of {sorted(supported)}."
            )

        domain_name = str(domain.get("label", domain.get("name", domain_type))).strip()
        if not domain_name:
            raise ValueError(f"Invalid empty domain name in spec: {raw}")

        if domain_type in _SHARED_EPS_TYPES and shared_eps is not None:
            eps_override = domain.get("eps")
            if eps_override is not None and not _float_close(float(eps_override), shared_eps):
                raise ValueError(
                    f"Domain '{domain_name}' has eps={eps_override}, but shared eps={shared_eps}."
                )
            domain.setdefault("eps", float(shared_eps))

        if domain_type in _SHARED_NORM_TYPES and shared_norm is not None:
            norm_override = domain.get("norm")
            if norm_override is not None and str(norm_override).lower() != shared_norm.lower():
                raise ValueError(
                    f"Domain '{domain_name}' has norm={norm_override}, but shared norm={shared_norm}."
                )
            domain.setdefault("norm", shared_norm)

        domain["type"] = domain_type
        domain["name"] = domain_name
        if "label" in domain:
            domain["label"] = domain_name

        if domain_type == "clean":
            if clean_seen:
                continue
            clean_seen = True
        normalized.append(domain)

    if include_clean and not clean_seen:
        normalized.insert(0, {"name": "clean", "label": "clean", "type": "clean"})

    if not normalized:
        raise ValueError("Train domains are empty after normalization.")

    names = [d["name"] for d in normalized]
    if len(set(names)) != len(names):
        raise ValueError(f"Duplicate domain names in train domains: {names}")
    return normalized


def resolve_attack_domains_cfg(
    cfg: Dict[str, Any],
    *,
    include_clean: bool,
    supported_types: Sequence[str] = tuple(sorted(SUPPORTED_DOMAIN_TYPES)),
) -> Tuple[str | None, float | None, List[Dict[str, Any]]]:
    """Resolve attack-domain definitions for attack-domain objectives.

    Preferred config is ``attack.train_domains``.
    Legacy ``attack.multi_train`` is also supported for compatibility.
    """
    attack_cfg = cfg.get("attack", {})

    train_domains = attack_cfg.get("train_domains")
    if isinstance(train_domains, list) and train_domains:
        shared_norm, shared_eps = _infer_shared_threat_model(train_domains)
        normalized = _normalize_domain_specs(
            train_domains,
            include_clean=include_clean,
            supported_types=supported_types,
            shared_norm=shared_norm,
            shared_eps=shared_eps,
        )
        return shared_norm, shared_eps, normalized

    multi_cfg = attack_cfg.get("multi_train", {})
    if isinstance(multi_cfg, dict) and multi_cfg:
        raw_domains = multi_cfg.get("domains", [])
        if not isinstance(raw_domains, list):
            raise ValueError("attack.multi_train.domains must be a list.")
        shared_eps = float(multi_cfg["eps"]) if multi_cfg.get("eps") is not None else None
        shared_norm = str(multi_cfg.get("norm")) if multi_cfg.get("norm") is not None else None
        normalized = _normalize_domain_specs(
            raw_domains,
            include_clean=include_clean,
            supported_types=supported_types,
            shared_norm=shared_norm,
            shared_eps=shared_eps,
        )
        return shared_norm, shared_eps, normalized

    raise ValueError(
        "Attack-domain objectives require attack.train_domains or attack.multi_train.domains."
    )


class AttackDomainObjective(Objective):
    """Shared helpers for objectives that treat attacks as fixed domains."""

    _SUPPORTED_TYPES = SUPPORTED_DOMAIN_TYPES

    def __init__(self, cfg: Dict[str, Any], model: Any, *, include_clean: bool = True) -> None:
        self.cfg = cfg
        self.model = model
        self.include_clean = bool(include_clean)
        self.shared_norm, self.shared_eps, self.domains = resolve_attack_domains_cfg(
            cfg,
            include_clean=self.include_clean,
            supported_types=tuple(sorted(self._SUPPORTED_TYPES)),
        )
        self.domain_names = [str(domain["name"]) for domain in self.domains]

        dataset_name = str(cfg["dataset"]["name"])
        self.attacks = {
            domain["name"]: build_attack(
                domain,
                model,
                dataset_name=dataset_name,
                shared_norm=self.shared_norm,
                shared_eps=self.shared_eps,
            )
            for domain in self.domains
        }

    def _attack_batch(
        self,
        attack: Any,
        images: torch.Tensor,
        labels: torch.Tensor,
        model: Any,
    ) -> torch.Tensor:
        model.eval()
        adv = attack(images, labels).detach()
        model.train()
        return adv

    def build_per_batch_batch(self, batch: Any, model: Any, rng: random.Random) -> Dict[str, Any]:
        data = as_xy_dict(batch)
        images = data["x"]
        labels = data["y"]
        batch_size = int(labels.size(0))

        idx = rng.randrange(len(self.domains))
        name = self.domains[idx]["name"]
        adv = self._attack_batch(self.attacks[name], images, labels, model)
        data["x"] = adv
        data["domain_name"] = name
        data["domain_batch_counts"] = {name: batch_size}
        return data

    def build_split_batch(self, batch: Any, model: Any, rng: random.Random) -> Dict[str, Any]:
        data = as_xy_dict(batch)
        images = data["x"]
        labels = data["y"]
        batch_size = int(labels.size(0))

        domain_ids = torch.tensor(
            [rng.randrange(len(self.domains)) for _ in range(batch_size)],
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

    def build_all_domains_batch(self, batch: Any, model: Any) -> Dict[str, Any]:
        data = as_xy_dict(batch)
        images = data["x"]
        labels = data["y"]
        batch_size = int(labels.size(0))

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

    def _compute_all_domain_stats(
        self,
        model: Any,
        x_domains: Sequence[torch.Tensor],
        labels: torch.Tensor,
        domain_names: Sequence[str],
    ) -> Dict[str, Any]:
        losses: List[torch.Tensor] = []
        acc_values: Dict[str, float] = {}
        metric_values: Dict[str, Any] = {}

        for x_dom, name in zip(x_domains, domain_names):
            logits_dom = model(x_dom)
            loss_dom = compute_loss(logits_dom, labels)
            acc_dom = float((logits_dom.argmax(dim=1) == labels).float().mean().item())
            losses.append(loss_dom)
            acc_values[str(name)] = acc_dom
            metric_values[f"loss_{name}"] = float(loss_dom.item())
            metric_values[f"acc_{name}"] = acc_dom

        if losses:
            domain_losses = torch.stack(losses)
            avg_group_loss = float(domain_losses.mean().item())
            worst_idx = int(domain_losses.argmax().item())
            worst_group_by_loss = str(domain_names[worst_idx])
        else:
            domain_losses = torch.empty(0, device=labels.device)
            avg_group_loss = 0.0
            worst_group_by_loss = ""

        mean_acc = float(sum(acc_values.values()) / max(len(acc_values), 1))
        worst_group_by_acc = min(acc_values, key=acc_values.get) if acc_values else ""
        return {
            "domain_losses": domain_losses,
            "domain_accs": acc_values,
            "metrics": metric_values,
            "mean_acc": mean_acc,
            "avg_group_loss": avg_group_loss,
            "worst_group_by_loss": worst_group_by_loss,
            "worst_group_by_acc": str(worst_group_by_acc),
        }


class MultiAttackERM(AttackDomainObjective):
    """ERM on adversarial domains with configurable per-batch strategy."""

    _SUPPORTED_STRATEGIES = {"per_batch", "split_batch", "all_domains"}

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
        super().__init__(cfg, model, include_clean=self.include_clean)

        seed = int(cfg.get("experiment", {}).get("seed", 42))
        self.rng = random.Random(seed)

        self.probe_cfg = dict(cfg.get("val", {}).get("probe", {}))
        self.probe_enabled = bool(self.probe_cfg.get("enabled", False))
        self.val_max_batches = int(train_cfg.get("val_max_batches", self.probe_cfg.get("max_batches", 10)))
        if self.val_max_batches < 0:
            self.val_max_batches = 0
        self.probe_attack = None
        if self.probe_enabled:
            self.probe_attack = self._build_probe_attack(model)

    def _build_probe_attack(self, model: Any) -> Any:
        probe_type = str(self.probe_cfg.get("type", "pgd")).lower()
        if probe_type != "pgd":
            raise ValueError(
                f"Unsupported val.probe.type={probe_type!r}. Only 'pgd' is supported."
            )
        eps = float(self.probe_cfg.get("eps", self.shared_eps if self.shared_eps is not None else 8.0 / 255.0))
        alpha = float(self.probe_cfg.get("alpha", self.probe_cfg.get("step_size", 2.0 / 255.0)))
        steps = int(self.probe_cfg.get("steps", self.probe_cfg.get("num_steps", 20)))
        restarts = int(self.probe_cfg.get("restarts", 1))
        norm = str(self.probe_cfg.get("norm", self.shared_norm if self.shared_norm is not None else "Linf"))
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

    def preprocess_batch(self, batch: Any, model: Any) -> Any:
        if self.strategy == "per_batch":
            return self.build_per_batch_batch(batch, model, self.rng)
        if self.strategy == "split_batch":
            return self.build_split_batch(batch, model, self.rng)
        if self.strategy == "all_domains":
            return self.build_all_domains_batch(batch, model)
        raise ValueError(f"Unsupported strategy: {self.strategy}")

    def compute_loss(self, model: Any, batch: Any) -> Tuple[torch.Tensor, Dict[str, Any]]:
        data = as_xy_dict(batch)
        labels = data["y"]
        batch_size = int(labels.size(0))

        domain_counts = data.get("domain_batch_counts", {})
        metrics: Dict[str, Any] = {"batch_size": batch_size, "domain_batch_counts": domain_counts}
        if "domain_name" in data:
            metrics["domain_name"] = data["domain_name"]

        if "x_domains" in data:
            stats = self._compute_all_domain_stats(
                model,
                data["x_domains"],
                labels,
                data.get("domain_names", self.domain_names),
            )
            domain_losses = stats["domain_losses"]
            loss = domain_losses.mean()
            acc = stats["mean_acc"]
            metrics.update(stats["metrics"])
            metrics.update(
                {
                    "loss": float(loss.item()),
                    "acc": acc,
                    "loss_adv": float(loss.item()),
                    "acc_adv": acc,
                    "loss_total": float(loss.item()),
                    "acc_total": acc,
                    "correct": acc * batch_size,
                    "avg_group_loss": stats["avg_group_loss"],
                    "worst_group_by_loss": stats["worst_group_by_loss"],
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
                "loss_adv": float(loss.item()),
                "acc_adv": correct / max(batch_size, 1),
                "loss_total": float(loss.item()),
                "acc_total": correct / max(batch_size, 1),
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
