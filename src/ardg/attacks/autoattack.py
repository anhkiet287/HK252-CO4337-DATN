"""AutoAttack evaluation runner."""

from __future__ import annotations

import time
from typing import Any, Dict

import torch
from torch import nn

from ardg.data.transforms import get_dataset_stats


def _require_autoattack() -> Any:
    try:
        from autoattack import AutoAttack  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "attack.autoattack.enabled=true but AutoAttack is not installed. "
            "Install with `pip install autoattack` (or "
            "`pip install git+https://github.com/fra31/auto-attack.git`) "
            "or disable via `attack.autoattack.enabled: false`."
        ) from exc
    return AutoAttack


class _PixelToNormalizedAdapter(nn.Module):
    """Wrap a normalized-input model to accept [0,1] pixel inputs."""

    def __init__(self, model: Any, mean: tuple[float, float, float], std: tuple[float, float, float]) -> None:
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1))

    def forward(self, x_pixel: torch.Tensor) -> torch.Tensor:
        x_norm = (x_pixel - self.mean) / self.std
        return self.model(x_norm)


def run_autoattack(
    model: Any,
    loader: Any,
    eps: float,
    device: str,
    *,
    dataset_name: str = "cifar10",
    norm: str = "Linf",
    version: str = "standard",
    max_batches: int | None = None,
) -> Dict[str, float]:
    """Run AutoAttack for evaluation.

    Args:
        model: Classification model that expects normalized inputs.
        loader: Dataloader yielding normalized images and labels.
        eps: Pixel-space attack epsilon value (e.g., 8/255 for CIFAR-10).
        device: Device string (e.g., "cuda", "cpu").
        dataset_name: Dataset name used to resolve mean/std normalization stats.
        norm: Threat norm. Default is Linf.
        version: AutoAttack version. Default is standard.
        max_batches: Optional cap for quick runs.

    Returns:
        Dictionary with robust accuracy and runtime stats.
    """
    AutoAttack = _require_autoattack()
    model.eval()
    device_t = torch.device(device)
    eps_f = float(eps)
    bs_default = int(getattr(loader, "batch_size", 128) or 128)
    mean, std = get_dataset_stats(dataset_name)
    adapted_model = _PixelToNormalizedAdapter(model, mean, std).to(device_t)
    adapted_model.eval()
    mean_t = adapted_model.mean
    std_t = adapted_model.std

    adversary = AutoAttack(
        adapted_model,
        norm=str(norm),
        eps=eps_f,
        version=str(version),
        device=str(device_t),
    )

    total_correct = 0
    total_seen = 0
    start = time.perf_counter()
    for batch_idx, (images, labels) in enumerate(loader, start=1):
        images = images.to(device_t)
        labels = labels.to(device_t)
        bs = min(bs_default, int(images.size(0)))
        images_pixel = torch.clamp(images * std_t + mean_t, 0.0, 1.0)
        x_adv = adversary.run_standard_evaluation(images_pixel, labels, bs=bs)
        x_adv = x_adv.to(device_t)
        with torch.no_grad():
            logits = adapted_model(x_adv)
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_seen += int(images.size(0))
        if max_batches and batch_idx >= int(max_batches):
            break

    runtime = time.perf_counter() - start
    return {
        "acc": total_correct / max(total_seen, 1),
        "n_samples": total_seen,
        "eps": eps_f,
        "eps_space": "pixel",
        "aa_runtime_sec": runtime,
        "aa_n_samples": total_seen,
        "aa_norm": str(norm),
        "aa_version": str(version),
    }
