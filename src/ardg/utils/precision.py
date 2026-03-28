"""Shared precision helpers for train and eval runtimes."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import torch

_ALLOWED_PRECISIONS = {"fp32", "fp16", "bf16"}


@dataclass(frozen=True)
class PrecisionSpec:
    requested: str
    effective: str
    device: str
    device_type: str
    autocast_enabled: bool
    scaler_enabled: bool
    dtype_name: str | None = None
    reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "requested": self.requested,
            "effective": self.effective,
            "device": self.device,
            "device_type": self.device_type,
            "autocast_enabled": self.autocast_enabled,
            "scaler_enabled": self.scaler_enabled,
            "dtype": self.dtype_name,
            "reason": self.reason,
        }


def resolve_precision_spec(requested: str, device: str | torch.device) -> PrecisionSpec:
    requested_name = str(requested or "fp32").strip().lower()
    if requested_name not in _ALLOWED_PRECISIONS:
        raise ValueError(
            f"Unsupported experiment.precision={requested_name!r}. "
            "Use one of ['fp32', 'fp16', 'bf16']."
        )

    device_obj = torch.device(device)
    device_name = str(device_obj)
    device_type = str(device_obj.type)
    if requested_name == "fp32":
        return PrecisionSpec(
            requested="fp32",
            effective="fp32",
            device=device_name,
            device_type=device_type,
            autocast_enabled=False,
            scaler_enabled=False,
        )

    if device_type != "cuda" or not torch.cuda.is_available():
        return PrecisionSpec(
            requested=requested_name,
            effective="fp32",
            device=device_name,
            device_type=device_type,
            autocast_enabled=False,
            scaler_enabled=False,
            reason="mixed precision is only enabled on CUDA; using fp32 fallback",
        )

    if requested_name == "bf16":
        is_bf16_supported = getattr(torch.cuda, "is_bf16_supported", None)
        if callable(is_bf16_supported) and not bool(is_bf16_supported()):
            return PrecisionSpec(
                requested=requested_name,
                effective="fp32",
                device=device_name,
                device_type=device_type,
                autocast_enabled=False,
                scaler_enabled=False,
                reason="current CUDA device does not report bf16 support; using fp32 fallback",
            )
        return PrecisionSpec(
            requested=requested_name,
            effective="bf16",
            device=device_name,
            device_type=device_type,
            autocast_enabled=True,
            scaler_enabled=False,
            dtype_name="bfloat16",
        )

    return PrecisionSpec(
        requested=requested_name,
        effective="fp16",
        device=device_name,
        device_type=device_type,
        autocast_enabled=True,
        scaler_enabled=True,
        dtype_name="float16",
    )


class PrecisionController:
    """Runtime precision controller shared by train and eval."""

    def __init__(self, requested: str, device: str | torch.device) -> None:
        self.spec = resolve_precision_spec(requested, device)
        self.device = torch.device(device)
        self._autocast_dtype = _resolve_autocast_dtype(self.spec.dtype_name)
        self._scaler = torch.amp.GradScaler('cuda',enabled=self.spec.scaler_enabled)

    @classmethod
    def from_config(cls, cfg: dict[str, Any], device: str | torch.device) -> "PrecisionController":
        requested = str(cfg.get("experiment", {}).get("precision", "fp32"))
        return cls(requested, device=device)

    @property
    def requested(self) -> str:
        return self.spec.requested

    @property
    def effective(self) -> str:
        return self.spec.effective

    @property
    def autocast_enabled(self) -> bool:
        return self.spec.autocast_enabled

    @property
    def scaler_enabled(self) -> bool:
        return self.spec.scaler_enabled

    def autocast_context(self) -> Any:
        if not self.spec.autocast_enabled or self._autocast_dtype is None:
            return nullcontext()
        return torch.autocast(
            device_type=self.spec.device_type,
            dtype=self._autocast_dtype,
            enabled=True,
        )

    def full_precision_context(self) -> Any:
        if not self.spec.autocast_enabled:
            return nullcontext()
        return torch.autocast(device_type=self.spec.device_type, enabled=False)

    def backward(self, loss: torch.Tensor) -> None:
        if self.spec.scaler_enabled:
            self._scaler.scale(loss).backward()
            return
        loss.backward()

    def step_optimizer(self, optimizer: torch.optim.Optimizer) -> None:
        if self.spec.scaler_enabled:
            self._scaler.step(optimizer)
            return
        optimizer.step()

    def update(self) -> None:
        if self.spec.scaler_enabled:
            self._scaler.update()

    def state_dict(self) -> dict[str, Any] | None:
        if not self.spec.scaler_enabled:
            return None
        return self._scaler.state_dict()

    def load_state_dict(self, state: dict[str, Any] | None) -> None:
        if self.spec.scaler_enabled and isinstance(state, dict):
            self._scaler.load_state_dict(state)

    def as_dict(self) -> dict[str, Any]:
        return self.spec.as_dict()

    def describe(self) -> str:
        detail = (
            f"requested={self.spec.requested} effective={self.spec.effective} "
            f"device={self.spec.device}"
        )
        if self.spec.autocast_enabled and self.spec.dtype_name:
            detail += f" autocast={self.spec.dtype_name}"
        if self.spec.scaler_enabled:
            detail += " grad_scaler=enabled"
        if self.spec.reason:
            detail += f" reason={self.spec.reason}"
        return detail


def _resolve_autocast_dtype(dtype_name: str | None) -> torch.dtype | None:
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    return None
