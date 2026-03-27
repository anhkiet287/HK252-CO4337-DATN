"""Fast preflight checks for data/model/attack IO consistency."""

from __future__ import annotations

import argparse
from typing import Any, Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from ardg.attacks.pgd import build_pgd_attack
from ardg.config import DEFAULT_CONFIG_PATH
from ardg.data.datasets import load_dataset
from ardg.data.transforms import build_transforms, get_dataset_stats
from ardg.experiments.common import load_runtime_config, setup_run
from ardg.models.factory import build_model
from ardg.utils.data import normalize_dataset_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fast preflight checks.")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="Path to YAML config.")
    parser.add_argument(
        "--profile",
        default=None,
        help="Optional runtime profile overlay YAML (e.g. configs/profiles/local_gpu.yaml).",
    )
    parser.add_argument("--io_mode", choices=["pixel", "normalized"], required=True, help="Input mode to verify.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for one-batch preflight.")
    parser.add_argument(
        "--platform",
        choices=("local", "colab"),
        default=None,
        help="Legacy platform override. Prefer --profile for thesis workflows.",
    )
    parser.add_argument(
        "--check-wandb",
        action="store_true",
        help="Also perform W&B initialization as a preflight stage.",
    )
    return parser.parse_args()


def _pixel_transform(dataset_name: str) -> Any:
    key = normalize_dataset_name(dataset_name)
    if key in {"cifar10", "cifar100"}:
        return transforms.ToTensor()
    if key in {"mnist", "fashion-mnist"}:
        return transforms.Compose(
            [
                transforms.Resize(32),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
            ]
        )
    if key == "color-mnist":
        return transforms.Compose([transforms.Resize(32)])
    raise ValueError(f"Unsupported dataset for preflight: {dataset_name}")


def _get_batch(cfg: dict, io_mode: str, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    dataset_name = cfg["dataset"]["name"]
    data_dir = cfg["dataset"].get("data_dir", "data")
    if io_mode == "normalized":
        transform = build_transforms(cfg, "val")
    else:
        transform = _pixel_transform(dataset_name)
    dataset = load_dataset(dataset_name, data_dir, train=True, transform=transform)
    loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=False, num_workers=0)
    images, labels = next(iter(loader))
    return images, labels


def _stats_tensors(dataset_name: str, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    mean, std = get_dataset_stats(dataset_name)
    mean_t = torch.tensor(mean, device=device).view(1, 3, 1, 1)
    std_t = torch.tensor(std, device=device).view(1, 3, 1, 1)
    return mean_t, std_t


def _normalize(x_pixel: torch.Tensor, mean_t: torch.Tensor, std_t: torch.Tensor) -> torch.Tensor:
    return (x_pixel - mean_t) / std_t


def _denormalize(x_norm: torch.Tensor, mean_t: torch.Tensor, std_t: torch.Tensor) -> torch.Tensor:
    return x_norm * std_t + mean_t


def _linf_max(delta: torch.Tensor) -> float:
    return float(delta.abs().view(delta.size(0), -1).max(dim=1).values.max().item())


def main() -> None:
    args = parse_args()
    cfg = load_runtime_config(
        args.config,
        profile_path=args.profile,
        platform_override=args.platform,
    )
    cfg_device = str(cfg.get("experiment", {}).get("device", "cpu"))
    device = torch.device(cfg_device if (cfg_device.startswith("cuda") and torch.cuda.is_available()) else "cpu")

    if args.check_wandb:
        _, _, run, _ = setup_run(
            args.config,
            stage="preflight",
            run_name_suffix="preflight",
            profile_path=args.profile,
            require_wandb=True,
            platform_override=args.platform,
        )
        if run is not None:
            run.finish()

    images, labels = _get_batch(cfg, args.io_mode, args.batch_size)
    images = images.to(device)
    labels = labels.to(device)

    mean_t, std_t = _stats_tensors(cfg["dataset"]["name"], device)
    if args.io_mode == "pixel":
        x_pixel = images
        x_norm = _normalize(x_pixel, mean_t, std_t)
    else:
        x_norm = images
        x_pixel = _denormalize(x_norm, mean_t, std_t)

    fails = []
    warns = []

    in_min = float(images.min().item())
    in_max = float(images.max().item())
    print(f"[INFO] io_mode={args.io_mode} input_min={in_min:.6f} input_max={in_max:.6f}")
    if args.io_mode == "pixel" and (in_min < -1e-4 or in_max > 1.0001):
        fails.append("pixel input is out of [0,1]")

    px_min = float(x_pixel.min().item())
    px_max = float(x_pixel.max().item())
    print(f"[INFO] pixel_view_min={px_min:.6f} pixel_view_max={px_max:.6f}")
    if px_min < -0.05 or px_max > 1.05:
        warns.append("pixel view has wide range; check transform/stat settings")

    if args.io_mode == "normalized":
        roundtrip = _normalize(_denormalize(x_norm, mean_t, std_t), mean_t, std_t)
        rt_err = float((roundtrip - x_norm).abs().max().item())
        print(f"[INFO] normalized_roundtrip_max_abs_err={rt_err:.8f}")

    eps = float(cfg.get("attack", {}).get("train", {}).get("eps", 8.0 / 255.0))
    step_size = float(cfg.get("attack", {}).get("train", {}).get("step_size", 2.0 / 255.0))
    steps_cfg = int(cfg.get("attack", {}).get("train", {}).get("num_steps", 5))
    steps = max(3, min(5, steps_cfg))

    if args.io_mode == "pixel":
        noisy = x_pixel + torch.randn_like(x_pixel) * (2.0 * eps)
        clamped = torch.clamp(noisy, 0.0, 1.0)
        clamp_ok = float(clamped.min().item()) >= -1e-6 and float(clamped.max().item()) <= 1.000001
    else:
        lo = (0.0 - mean_t) / std_t
        hi = (1.0 - mean_t) / std_t
        noisy = x_norm + torch.randn_like(x_norm) * 2.0
        clamped = torch.max(torch.min(noisy, hi), lo)
        clamp_ok = float(clamped.min().item()) >= float(lo.min().item()) - 1e-6 and float(
            clamped.max().item()
        ) <= float(hi.max().item()) + 1e-6
    print(f"[INFO] clamp_check={'PASS' if clamp_ok else 'FAIL'}")
    if not clamp_ok:
        fails.append("clamp check failed")

    model = build_model(cfg).to(device)
    model.eval()
    with torch.no_grad():
        logits = model(x_norm)
    forward_ok = logits.ndim == 2 and logits.size(0) == x_norm.size(0)
    print(f"[INFO] forward_check={'PASS' if forward_ok else 'FAIL'} logits_shape={tuple(logits.shape)}")
    if not forward_ok:
        fails.append("model forward failed on one batch")

    attack = build_pgd_attack(
        {
            "dataset_name": cfg["dataset"]["name"],
            "eps": eps,
            "step_size": step_size,
            "num_steps": steps,
            "restarts": 1,
        },
        model,
    )
    x_adv_norm = attack(x_norm, labels)
    x_adv_pixel = _denormalize(x_adv_norm, mean_t, std_t)
    delta_linf_max = _linf_max(x_adv_pixel - x_pixel)
    print(
        f"[INFO] pgd_small_steps={steps} delta_linf_max={delta_linf_max:.6f} "
        f"eps={eps:.6f} eps_ok={delta_linf_max <= (eps + 1e-3)}"
    )
    if delta_linf_max > (eps + 1e-3):
        fails.append("PGD delta exceeds epsilon bound in pixel space")

    if fails:
        for item in fails:
            print(f"[FAIL] {item}")
        raise SystemExit(1)
    if warns:
        for item in warns:
            print(f"[WARN] {item}")
        print("[WARN] preflight finished with warnings")
        return
    print("[PASS] preflight checks passed")


if __name__ == "__main__":
    main()
