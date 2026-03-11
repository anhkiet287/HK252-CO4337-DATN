"""Shared q-state helpers for GroupDRO objectives."""

from __future__ import annotations

from typing import Any, Dict, Sequence

import torch


def normalize_q(q: torch.Tensor) -> torch.Tensor:
    """Project a non-negative vector back onto the probability simplex."""
    q = torch.clamp(q, min=0.0)
    total = float(q.sum().item())
    if not torch.isfinite(q).all() or total <= 0.0:
        return torch.ones_like(q) / max(int(q.numel()), 1)
    return q / total


def update_q_from_losses(q: torch.Tensor, loss_g: torch.Tensor, eta_q: float) -> torch.Tensor:
    """Multiplicative GroupDRO update using detached group losses."""
    updated = q * torch.exp(float(eta_q) * loss_g.detach())
    return normalize_q(updated)


def weighted_group_loss(q: torch.Tensor, loss_g: torch.Tensor) -> torch.Tensor:
    """Compute the GroupDRO weighted loss."""
    return torch.sum(q * loss_g)


def q_entropy(q: torch.Tensor) -> torch.Tensor:
    """Shannon entropy of q with numerical protection."""
    safe_q = torch.clamp(q, min=1e-12)
    return -(safe_q * safe_q.log()).sum()


def normalize_group_losses(loss_g: torch.Tensor, baseline_g: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalize group losses against a positive per-group baseline."""
    safe_baseline = torch.clamp(baseline_g, min=float(eps))
    return loss_g / safe_baseline


class DomainWeightState:
    """Manage GroupDRO q state for fixed domain names."""

    def __init__(self, domain_names: Sequence[str], eta_q: float, init_q: str = "uniform") -> None:
        self.domain_names = [str(name) for name in domain_names]
        self.eta_q = float(eta_q)
        self.init_q = str(init_q).lower()
        if self.init_q != "uniform":
            raise ValueError(f"Unsupported train.groupdro.init_q={init_q!r}. Only 'uniform' is supported.")
        self.q: torch.Tensor | None = None

    def ensure_initialized(self, device: torch.device | str) -> torch.Tensor:
        if not self.domain_names:
            raise ValueError("DomainWeightState requires at least one domain.")
        if self.q is None or int(self.q.numel()) != len(self.domain_names):
            value = 1.0 / float(len(self.domain_names))
            self.q = torch.full((len(self.domain_names),), value, device=device, dtype=torch.float32)
        else:
            self.q = self.q.to(device=device, dtype=torch.float32)
            self.q = normalize_q(self.q)
        return self.q

    def set_q(self, q: torch.Tensor) -> torch.Tensor:
        if int(q.numel()) != len(self.domain_names):
            raise ValueError(
                f"Expected q with {len(self.domain_names)} entries, but got shape={tuple(q.shape)}."
            )
        self.q = normalize_q(q.detach().clone().to(dtype=torch.float32))
        return self.q

    def update(self, loss_g: torch.Tensor) -> torch.Tensor:
        q = self.ensure_initialized(loss_g.device)
        self.q = update_q_from_losses(q, loss_g, eta_q=self.eta_q)
        return self.q

    def entropy(self) -> float:
        q = self.ensure_initialized("cpu")
        return float(q_entropy(q).item())

    def state_dict(self) -> Dict[str, Any]:
        return {
            "q": None if self.q is None else self.q.detach().cpu(),
            "domain_names": list(self.domain_names),
            "eta_q": float(self.eta_q),
            "init_q": str(self.init_q),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        q = state.get("q")
        saved_names = state.get("domain_names")
        if q is None:
            self.q = None
            return

        q_tensor = q.detach().clone().to(dtype=torch.float32)
        if saved_names:
            saved_names = [str(name) for name in saved_names]
            if set(saved_names) != set(self.domain_names):
                raise ValueError(
                    "Checkpoint q domain names do not match current config. "
                    f"saved={saved_names}, current={self.domain_names}"
                )
            reorder = [saved_names.index(name) for name in self.domain_names]
            q_tensor = q_tensor[reorder]

        self.set_q(q_tensor)


class DomainLossEMAState:
    """Track per-domain EMA baselines for normalized-loss comparisons."""

    def __init__(self, domain_names: Sequence[str], momentum: float = 0.9, eps: float = 1e-8) -> None:
        self.domain_names = [str(name) for name in domain_names]
        self.momentum = float(momentum)
        self.eps = float(eps)
        if not 0.0 <= self.momentum < 1.0:
            raise ValueError(f"Expected normalized-loss EMA momentum in [0, 1), got {momentum!r}.")
        if self.eps <= 0.0:
            raise ValueError(f"Expected normalized-loss eps > 0, got {eps!r}.")
        self.ema_losses: torch.Tensor | None = None

    def _validate_shape(self, tensor: torch.Tensor) -> None:
        if int(tensor.numel()) != len(self.domain_names):
            raise ValueError(
                f"Expected tensor with {len(self.domain_names)} entries, but got shape={tuple(tensor.shape)}."
            )

    def ensure_initialized(self, loss_g: torch.Tensor) -> torch.Tensor:
        self._validate_shape(loss_g)
        if self.ema_losses is None or int(self.ema_losses.numel()) != len(self.domain_names):
            self.ema_losses = loss_g.detach().clone().to(dtype=torch.float32)
        else:
            self.ema_losses = self.ema_losses.to(device=loss_g.device, dtype=torch.float32)
        return self.ema_losses

    def set_ema_losses(self, ema_losses: torch.Tensor) -> torch.Tensor:
        self._validate_shape(ema_losses)
        self.ema_losses = ema_losses.detach().clone().to(dtype=torch.float32)
        return self.ema_losses

    def normalized(self, loss_g: torch.Tensor) -> torch.Tensor:
        self._validate_shape(loss_g)
        if self.ema_losses is None or int(self.ema_losses.numel()) != len(self.domain_names):
            return loss_g.detach().clone().to(dtype=torch.float32)
        baseline = self.ema_losses.to(device=loss_g.device, dtype=torch.float32)
        return normalize_group_losses(loss_g.detach().to(dtype=torch.float32), baseline, eps=self.eps)

    def update(self, loss_g: torch.Tensor) -> torch.Tensor:
        ema_losses = self.ensure_initialized(loss_g)
        updated = self.momentum * ema_losses + (1.0 - self.momentum) * loss_g.detach().to(dtype=torch.float32)
        self.ema_losses = updated
        return self.ema_losses

    def state_dict(self) -> Dict[str, Any]:
        return {
            "ema_losses": None if self.ema_losses is None else self.ema_losses.detach().cpu(),
            "domain_names": list(self.domain_names),
            "momentum": float(self.momentum),
            "eps": float(self.eps),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        ema_losses = state.get("ema_losses")
        saved_names = state.get("domain_names")
        if ema_losses is None:
            self.ema_losses = None
            return

        ema_tensor = ema_losses.detach().clone().to(dtype=torch.float32)
        if saved_names:
            saved_names = [str(name) for name in saved_names]
            if set(saved_names) != set(self.domain_names):
                raise ValueError(
                    "Checkpoint normalized-loss domain names do not match current config. "
                    f"saved={saved_names}, current={self.domain_names}"
                )
            reorder = [saved_names.index(name) for name in self.domain_names]
            ema_tensor = ema_tensor[reorder]

        self.set_ema_losses(ema_tensor)
