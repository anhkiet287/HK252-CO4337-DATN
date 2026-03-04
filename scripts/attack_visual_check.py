"""Validate and visualize adversarial attack outputs in pixel and normalized spaces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
from torchvision.utils import make_grid, save_image

from ardg.attacks.cw import build_cw_attack
from ardg.attacks.deepfool import build_deepfool_attack
from ardg.attacks.fab import build_fab_attack
from ardg.attacks.fgsm import build_fgsm_attack
from ardg.attacks.attack_suite import build_eval_attacks, build_train_attack, build_val_attack
from ardg.attacks.pgd import build_pgd_attack
from ardg.attacks.square import build_square_attack
from ardg.config import DEFAULT_CONFIG_PATH, load_config
from ardg.data.transforms import get_dataset_stats
from ardg.experiments.common import build_loaders, load_model_from_checkpoint
from ardg.models.factory import build_model
from ardg.utils.paths import get_run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visual attack sanity check and visualization.")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="Path to YAML config.")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional checkpoint path. If omitted, uses randomly initialized model.",
    )
    parser.add_argument("--split", default="val", choices=["train", "val", "test"], help="Data split to sample from.")
    parser.add_argument(
        "--attack-source",
        default="train",
        choices=["train", "val", "eval_pgd20"],
        help="Which configured attack definition to apply.",
    )
    parser.add_argument(
        "--attack",
        default="from_source",
        choices=["from_source", "pgd", "fgsm", "cw", "deepfool", "square", "fab"],
        help="Attack type override. `from_source` uses the attack config from --attack-source.",
    )
    parser.add_argument("--num-samples", type=int, default=8, help="Number of examples to visualize.")
    parser.add_argument(
        "--class-id",
        type=int,
        default=None,
        help="Optional class id filter. If set, only visualize this class.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory. Defaults to <run_dir>/attack_visual_check.",
    )
    parser.add_argument(
        "--strict-eps",
        action="store_true",
        help="Fail if Linf(pixel) exceeds configured eps (when eps is available).",
    )
    return parser.parse_args()


def _resolve_device(cfg: Dict[str, Any]) -> str:
    requested = str(cfg.get("experiment", {}).get("device", "cpu"))
    if requested.startswith("cuda") and torch.cuda.is_available():
        return requested
    return "cpu"


def _build_attack_from_cfg(atk_cfg: Dict[str, Any], model: Any) -> Any:
    name = str(atk_cfg.get("name", "pgd")).lower()
    if name in {"pgd", "pgd_linf"}:
        return build_pgd_attack(atk_cfg, model)
    if name == "fgsm":
        return build_fgsm_attack(atk_cfg, model)
    if name in {"cw", "carlini-wagner", "carlini_wagner"}:
        return build_cw_attack(atk_cfg, model)
    if name == "deepfool":
        return build_deepfool_attack(atk_cfg, model)
    if name == "square":
        return build_square_attack(atk_cfg, model)
    if name == "fab":
        return build_fab_attack(atk_cfg, model)
    raise ValueError(f"Unsupported attack: {name}")


def _resolve_attack(
    model: Any,
    cfg: Dict[str, Any],
    source: str,
    attack_override: str,
) -> Tuple[Any, float | None, str]:
    if source == "train":
        if attack_override == "from_source":
            atk_cfg = dict(cfg["attack"]["train"])
            atk_cfg["dataset_name"] = cfg["dataset"]["name"]
            eps = atk_cfg.get("eps")
            return build_train_attack(cfg, model), (float(eps) if eps is not None else None), str(atk_cfg.get("name", "pgd"))
        atk_cfg = dict(cfg["attack"]["train"])
    elif source == "val":
        if attack_override == "from_source":
            atk_cfg = dict(cfg["attack"]["val"])
            atk_cfg["dataset_name"] = cfg["dataset"]["name"]
            eps = atk_cfg.get("eps")
            return build_val_attack(cfg, model), (float(eps) if eps is not None else None), str(atk_cfg.get("name", "pgd"))
        atk_cfg = dict(cfg["attack"]["val"])
    else:
        if attack_override == "from_source":
            attacks = build_eval_attacks(cfg, model)
            if "pgd20" not in attacks:
                raise RuntimeError("eval_pgd20 attack is not configured. Provide attack.eval.pgd20 in config.")
            eval_cfg = cfg.get("attack", {}).get("eval", {}).get("pgd20", {})
            eps = eval_cfg.get("eps", cfg.get("attack", {}).get("val", {}).get("eps"))
            return attacks["pgd20"], (float(eps) if eps is not None else None), "pgd"
        eval_cfg = cfg.get("attack", {}).get("eval", {}).get("pgd20", {})
        atk_cfg = {
            "eps": eval_cfg.get("eps", cfg.get("attack", {}).get("val", {}).get("eps", 8.0 / 255.0)),
            "step_size": eval_cfg.get(
                "step_size",
                eval_cfg.get("alpha", cfg.get("attack", {}).get("val", {}).get("step_size", 2.0 / 255.0)),
            ),
            "num_steps": int(eval_cfg.get("num_steps", eval_cfg.get("steps", 20))),
            "restarts": int(eval_cfg.get("restarts", 5)),
        }

    atk_cfg = dict(atk_cfg)
    atk_cfg["dataset_name"] = cfg["dataset"]["name"]
    atk_cfg["name"] = attack_override
    attack = _build_attack_from_cfg(atk_cfg, model)
    eps = atk_cfg.get("eps")
    return attack, (float(eps) if eps is not None else None), str(atk_cfg["name"])


def _as_tuple_batch(batch: Any) -> Tuple[torch.Tensor, torch.Tensor]:
    if isinstance(batch, dict):
        return batch["x"], batch["y"]
    images, labels = batch
    return images, labels


def _collect_samples(loader: Any, num_samples: int, class_id: int | None) -> Tuple[torch.Tensor, torch.Tensor]:
    xs = []
    ys = []
    collected = 0
    for batch in loader:
        images, labels = _as_tuple_batch(batch)
        if class_id is not None:
            mask = labels == int(class_id)
            if not mask.any():
                continue
            images = images[mask]
            labels = labels[mask]
        if images.size(0) == 0:
            continue
        take = min(int(images.size(0)), int(num_samples - collected))
        xs.append(images[:take])
        ys.append(labels[:take])
        collected += int(take)
        if collected >= int(num_samples):
            break

    if collected == 0:
        if class_id is None:
            raise RuntimeError("No samples found in loader.")
        raise RuntimeError(f"No samples found for class_id={class_id}.")
    if collected < int(num_samples):
        print(f"[WARN] Requested {num_samples} samples but collected only {collected}.")
    return torch.cat(xs, dim=0), torch.cat(ys, dim=0)


def _to_pixel(x_norm: torch.Tensor, mean_t: torch.Tensor, std_t: torch.Tensor) -> torch.Tensor:
    return x_norm * std_t + mean_t


def _scale_for_view(x: torch.Tensor) -> torch.Tensor:
    """Per-image min-max scale to [0,1] for visualization."""
    flat = x.view(x.size(0), -1)
    lo = flat.min(dim=1).values.view(-1, 1, 1, 1)
    hi = flat.max(dim=1).values.view(-1, 1, 1, 1)
    return (x - lo) / (hi - lo + 1e-8)


def _delta_signed_vis(delta: torch.Tensor) -> torch.Tensor:
    """Map signed perturbation to [0,1] where 0.5 means zero."""
    m = delta.abs().view(delta.size(0), -1).max(dim=1).values.view(-1, 1, 1, 1).clamp_min(1e-8)
    return delta / m * 0.5 + 0.5


def _save_grid(x: torch.Tensor, path: Path) -> None:
    grid = make_grid(x, nrow=min(8, int(x.size(0))))
    save_image(grid, path)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    device = torch.device(_resolve_device(cfg))

    train_loader, val_loader, test_loader = build_loaders(cfg)
    loader_map = {"train": train_loader, "val": val_loader, "test": test_loader}
    loader = loader_map[args.split]

    if args.checkpoint:
        model = load_model_from_checkpoint(cfg, args.checkpoint, str(device))
    else:
        model = build_model(cfg).to(device)
    model.eval()

    attack, eps_cfg, attack_name = _resolve_attack(model, cfg, args.attack_source, args.attack)

    images, labels = _collect_samples(loader, args.num_samples, args.class_id)
    images = images.to(device)
    labels = labels.to(device)

    with torch.no_grad():
        clean_logits = model(images)
        clean_pred = clean_logits.argmax(dim=1)

    adv = attack(images, labels).detach()
    with torch.no_grad():
        adv_logits = model(adv)
        adv_pred = adv_logits.argmax(dim=1)

    mean, std = get_dataset_stats(cfg["dataset"]["name"])
    mean_t = torch.tensor(mean, device=device).view(1, 3, 1, 1)
    std_t = torch.tensor(std, device=device).view(1, 3, 1, 1)

    clean_norm = images
    adv_norm = adv
    delta_norm = adv_norm - clean_norm

    clean_pixel_raw = _to_pixel(clean_norm, mean_t, std_t)
    adv_pixel_raw = _to_pixel(adv_norm, mean_t, std_t)
    delta_pixel = adv_pixel_raw - clean_pixel_raw

    clean_pixel = torch.clamp(clean_pixel_raw, 0.0, 1.0)
    adv_pixel = torch.clamp(adv_pixel_raw, 0.0, 1.0)

    # Save views required for quick sanity checks.
    out_dir = Path(args.output_dir) if args.output_dir else Path(get_run_dir(cfg)) / "attack_visual_check"
    out_dir.mkdir(parents=True, exist_ok=True)
    _save_grid(clean_pixel, out_dir / "clean_pixel.png")
    _save_grid(adv_pixel, out_dir / "adv_pixel.png")
    _save_grid(_delta_signed_vis(delta_pixel), out_dir / "perturbation_pixel_signed.png")
    _save_grid(_scale_for_view(clean_norm), out_dir / "clean_normalized_scaled.png")
    _save_grid(_scale_for_view(adv_norm), out_dir / "adv_normalized_scaled.png")
    _save_grid(_delta_signed_vis(delta_norm), out_dir / "perturbation_normalized_signed.png")

    linf_pixel = delta_pixel.abs().view(delta_pixel.size(0), -1).max(dim=1).values
    linf_norm = delta_norm.abs().view(delta_norm.size(0), -1).max(dim=1).values
    changed = (clean_pred != adv_pred).sum().item()

    eps_ok = None
    if eps_cfg is not None:
        eps_ok = bool(float(linf_pixel.max().item()) <= float(eps_cfg + 1e-3))
        if args.strict_eps and not eps_ok:
            raise RuntimeError(
                f"Linf(pixel) exceeded eps: max={float(linf_pixel.max().item()):.6f} eps={float(eps_cfg):.6f}"
            )

    stats = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "split": args.split,
        "attack_source": args.attack_source,
        "attack_name": attack_name,
        "num_samples": int(images.size(0)),
        "class_id": args.class_id,
        "eps_config": eps_cfg,
        "linf_pixel_max": float(linf_pixel.max().item()),
        "linf_pixel_mean": float(linf_pixel.mean().item()),
        "linf_norm_max": float(linf_norm.max().item()),
        "linf_norm_mean": float(linf_norm.mean().item()),
        "eps_ok": eps_ok,
        "clean_acc_batch": float((clean_pred == labels).float().mean().item()),
        "adv_acc_batch": float((adv_pred == labels).float().mean().item()),
        "prediction_changed_count": int(changed),
        "prediction_changed_rate": float(changed / max(int(images.size(0)), 1)),
        "files": {
            "clean_pixel": str(out_dir / "clean_pixel.png"),
            "adv_pixel": str(out_dir / "adv_pixel.png"),
            "perturbation_pixel_signed": str(out_dir / "perturbation_pixel_signed.png"),
            "clean_normalized_scaled": str(out_dir / "clean_normalized_scaled.png"),
            "adv_normalized_scaled": str(out_dir / "adv_normalized_scaled.png"),
            "perturbation_normalized_signed": str(out_dir / "perturbation_normalized_signed.png"),
        },
    }

    with (out_dir / "stats.json").open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2)

    print(f"[PASS] Attack visualization saved to: {out_dir}")
    print(
        "[INFO] "
        f"linf_pixel_max={stats['linf_pixel_max']:.6f}, eps={stats['eps_config']}, "
        f"eps_ok={stats['eps_ok']}, clean_acc_batch={stats['clean_acc_batch']:.3f}, "
        f"adv_acc_batch={stats['adv_acc_batch']:.3f}"
    )


if __name__ == "__main__":
    main()
