"""Visualize clean vs adversarial examples and perturbations for a small batch.

Usage:
    python visualize/adv_inspect.py --config configs/full.yaml --checkpoint path/to/ckpt.pt --split val --num-samples 8

Outputs (in logging.output_dir/<run_name>_adv_inspect/):
    clean.png       -- grid of clean images
    adv.png         -- grid of adversarial images
    delta.png       -- grid of (adv - clean) magnified for visibility
    stats.json      -- max|delta| in pixel space and normalized space
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Tuple

import torch
from torchvision.utils import make_grid, save_image

from ardg.attacks.attack_suite import build_val_attack, build_train_attack
from ardg.config import load_config
from ardg.experiments.common import build_loaders, load_model_from_checkpoint, setup_run
from ardg.models.factory import build_model
from ardg.data.transforms import get_dataset_stats


def denorm(x: torch.Tensor, mean: Tuple[float, float, float], std: Tuple[float, float, float]) -> torch.Tensor:
    mean_t = torch.tensor(mean, device=x.device).view(1, 3, 1, 1)
    std_t = torch.tensor(std, device=x.device).view(1, 3, 1, 1)
    return x * std_t + mean_t


def save_grids(clean: torch.Tensor, adv: torch.Tensor, out_dir: Path, run_name: str, mean, std) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    # Denorm to [0,1] for visualization
    clean_img = torch.clamp(denorm(clean, mean, std), 0.0, 1.0)
    adv_img = torch.clamp(denorm(adv, mean, std), 0.0, 1.0)
    delta = adv - clean
    # Scale delta for visibility
    delta_vis = (delta / (delta.abs().max() + 1e-8)) * 0.5 + 0.5

    save_image(make_grid(clean_img, nrow=min(8, clean.size(0))), out_dir / f"{run_name}_clean.png")
    save_image(make_grid(adv_img, nrow=min(8, adv.size(0))), out_dir / f"{run_name}_adv.png")
    save_image(make_grid(delta_vis, nrow=min(8, delta_vis.size(0))), out_dir / f"{run_name}_delta.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--checkpoint", default=None, help="Optional checkpoint path.")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"], help="Which split to draw a batch from.")
    parser.add_argument("--num-samples", type=int, default=8, help="How many samples to visualize.")
    args = parser.parse_args()

    cfg, logger, _, device = setup_run(args.config)
    device = torch.device(device)
    train_loader, val_loader, test_loader = build_loaders(cfg)
    loaders = {"train": train_loader, "val": val_loader, "test": test_loader}
    loader = loaders[args.split]

    if args.checkpoint:
        model = load_model_from_checkpoint(cfg, args.checkpoint, str(device))
    else:
        model = build_model(cfg)
        model.to(device)
        model.eval()

    # Pick attack matching split
    if args.split == "train":
        attack = build_train_attack(cfg, model)
    else:
        attack = build_val_attack(cfg, model)

    mean, std = get_dataset_stats(cfg["dataset"]["name"])

    images, labels = next(iter(loader))
    if isinstance(images, dict):  # not expected, but guard
        images, labels = images["x"], images["y"]
    images = images[: args.num_samples].to(device)
    labels = labels[: args.num_samples].to(device)

    with torch.no_grad():
        clean_logits = model(images)
        clean_pred = clean_logits.argmax(1)

    adv = attack(images, labels)
    with torch.no_grad():
        adv_logits = model(adv)
        adv_pred = adv_logits.argmax(1)

    out_root = Path(cfg["logging"]["output_dir"])
    run_name = cfg["logging"].get("run_name", "adv_inspect")
    out_dir = out_root / f"{run_name}_adv_inspect"
    save_grids(images, adv, out_dir, run_name, mean, std)

    # Compute norms in both normalized and pixel space
    delta = adv - images
    std_t = torch.tensor(std, device=device).view(1, 3, 1, 1)
    max_delta_norm = delta.abs().amax().item()
    max_delta_pix = (delta * std_t).abs().amax().item()
    stats = {
        "split": args.split,
        "num_samples": int(images.size(0)),
        "max_delta_norm": float(max_delta_norm),
        "max_delta_pix": float(max_delta_pix),
        "eps_cfg": float(cfg["attack"][args.split if args.split != "train" else "train"]["eps"]),
        "clean_correct": int((clean_pred == labels).sum().item()),
        "adv_correct": int((adv_pred == labels).sum().item()),
    }
    with open(out_dir / f"{run_name}_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    logger.info("Saved clean/adv/delta grids and stats to %s", out_dir)


if __name__ == "__main__":
    main()
