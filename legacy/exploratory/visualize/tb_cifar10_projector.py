import argparse
import os
import random
from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights

# CIFAR-10 label names (TensorBoard can color points by this metadata column)
CLASSES: List[str] = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

# CIFAR normalization (common for CIFAR-trained checkpoints)
CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_STD = [0.2023, 0.1994, 0.2010]


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(device: torch.device, imagenet_stem: bool, weights: ResNet18_Weights | None, num_classes: int | None = None) -> nn.Module:
    """
    ResNet18; default: ImageNet weights. Keeps FC head so we can log both logits (for predicted labels) and penultimate features.
    `imagenet_stem=False` switches to CIFAR-style stem (3x3 stride1, no maxpool) for CIFAR checkpoints.
    """
    model = resnet18(weights=weights, num_classes=num_classes if num_classes is not None else (1000 if weights else 10))
    # swap stem to CIFAR style if requested
    if not imagenet_stem:
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
    model.to(device).eval()
    return model


def forward_penultimate_and_logits(model: nn.Module, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns (penultimate [B,512], logits [B,C]) using an avgpool hook to capture the 512-d vector.
    """
    feats = {}

    def hook_avgpool(m, inp, out):
        feats["penultimate"] = out.flatten(1).detach()

    h = model.avgpool.register_forward_hook(hook_avgpool)
    logits = model(x)
    h.remove()

    if "penultimate" not in feats:
        raise RuntimeError("Failed to capture penultimate features. Check model architecture.")
    return feats["penultimate"], logits.detach()


@torch.no_grad()
def clamp01(x: torch.Tensor) -> torch.Tensor:
    return x.clamp(0.0, 1.0)


def pgd_attack(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    eps: float,
    alpha: float,
    steps: int,
    random_start: bool,
) -> torch.Tensor:
    """
    Basic Linf PGD on pixels in [0,1], model expects normalized inputs.
    Returns final adversarial images in [0,1].
    """
    if steps <= 0:
        return x

    x_clean = x.detach()
    x_adv = x_clean.clone()
    if random_start:
        x_adv = x_adv + torch.empty_like(x_adv).uniform_(-eps, eps)
        x_adv = clamp01(x_adv)

    for _ in range(steps):
        x_adv.requires_grad_(True)
        logits = model((x_adv - mean) / std)
        loss = torch.nn.functional.cross_entropy(logits, y)
        grad = torch.autograd.grad(loss, x_adv, only_inputs=True)[0]

        with torch.no_grad():
            x_adv = x_adv + alpha * grad.sign()
            x_adv = torch.max(torch.min(x_adv, x_clean + eps), x_clean - eps)
            x_adv = clamp01(x_adv)

    return x_adv.detach()


def resolve_imagenet_stats(device: torch.device) -> (torch.Tensor, torch.Tensor):
    """
    TorchVision versions differ in whether weights.meta includes mean/std; fall back to canonical values.
    """
    default_mean = [0.485, 0.456, 0.406]
    default_std = [0.229, 0.224, 0.225]

    weights = ResNet18_Weights.IMAGENET1K_V1
    meta = getattr(weights, "meta", {}) or {}
    mean = meta.get("mean", default_mean)
    std = meta.get("std", default_std)

    mean_t = torch.tensor(mean, device=device).view(1, 3, 1, 1)
    std_t = torch.tensor(std, device=device).view(1, 3, 1, 1)
    return mean_t, std_t


def resolve_cifar_stats(device: torch.device) -> (torch.Tensor, torch.Tensor):
    mean_t = torch.tensor(CIFAR10_MEAN, device=device).view(1, 3, 1, 1)
    std_t = torch.tensor(CIFAR10_STD, device=device).view(1, 3, 1, 1)
    return mean_t, std_t


def parse_vector3(s: str) -> List[float]:
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 3:
        raise ValueError("Expected 3 comma-separated values")
    return [float(p) for p in parts]


def _strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in state_dict.items():
        nk = k[len("module."):] if k.startswith("module.") else k
        out[nk] = v
    return out


def _is_state_dict_like(d):
    if not isinstance(d, dict) or len(d) == 0:
        return False
    tensor_cnt = sum(torch.is_tensor(v) for v in d.values())
    return tensor_cnt >= max(5, int(0.5 * len(d)))


def extract_state_dict(ckpt_obj):
    if _is_state_dict_like(ckpt_obj):
        return ckpt_obj
    if isinstance(ckpt_obj, dict):
        candidates = [
            "state_dict",
            "model_state_dict",
            "model",
            "net",
            "network",
            "encoder",
            "student",
            "ema",
        ]
        for k in candidates:
            if k in ckpt_obj:
                v = ckpt_obj[k]
                if _is_state_dict_like(v):
                    return v
                if isinstance(v, dict):
                    for kk in ["state_dict", "model_state_dict"]:
                        if kk in v and _is_state_dict_like(v[kk]):
                            return v[kk]
        for _, v in ckpt_obj.items():
            if _is_state_dict_like(v):
                return v
        raise KeyError(f"Can't find a state_dict in checkpoint. Keys: {list(ckpt_obj.keys())[:20]}")
    raise TypeError(f"Unsupported checkpoint type: {type(ckpt_obj)}")


def load_state_dict_flexible(model: nn.Module, ckpt_path: str, strict: bool = False):
    obj = torch.load(ckpt_path, map_location="cpu")
    state = extract_state_dict(obj)
    state = _strip_module_prefix(state)
    missing, unexpected = model.load_state_dict(state, strict=strict)
    if missing:
        print(f"[WARN] Missing keys ({len(missing)}): {missing[:12]}{' ...' if len(missing)>12 else ''}")
    if unexpected:
        print(f"[WARN] Unexpected keys ({len(unexpected)}): {unexpected[:12]}{' ...' if len(unexpected)>12 else ''}")
    return obj


def main() -> None:
    ap = argparse.ArgumentParser(description="Log CIFAR-10 embeddings to TensorBoard Projector.")
    ap.add_argument("--data_root", type=str, default="./data", help="Path to download/load CIFAR-10.")
    ap.add_argument("--run_dir", type=str, default="runs/cifar10_projector", help="TensorBoard logdir.")
    ap.add_argument("--splits", type=str, default="test", help="Comma-separated splits to log: train,test or both.")
    ap.add_argument("--max_samples_per_split", type=int, default=5000, help="Limit per split (0 = all).")
    ap.add_argument("--batch_size", type=int, default=256, help="DataLoader batch size.")
    ap.add_argument("--img_size", type=int, default=32, help="Resize shorter side before embedding (use 32 for native CIFAR).")
    ap.add_argument("--tag", type=str, default="cifar10_resnet18_penultimate", help="Projector tag name.")
    ap.add_argument("--seed", type=int, default=0, help="Random seed for shuffling.")
    ap.add_argument("--ckpt", type=str, default=None, help="Path to a CIFAR-trained ResNet18 checkpoint (model A).")
    ap.add_argument("--ckpt2", type=str, default=None, help="Optional second checkpoint (model B) for side-by-side comparison.")
    ap.add_argument("--model_names", type=str, default="modelA,modelB", help="Comma-separated names for ckpt / ckpt2.")
    ap.add_argument("--imagenet_stem", action="store_true", help="Use ImageNet stem (conv7x7 stride2 + maxpool).")
    ap.add_argument("--norm", choices=["cifar", "imagenet"], default="cifar", help="Normalization to match the checkpoint.")
    ap.add_argument("--no_sprites", action="store_true", help="If set, skip thumbnail sprites and render points only.")
    ap.add_argument("--mean", type=str, default=None, help="Override mean, comma-separated (e.g., 0.4914,0.4822,0.4465).")
    ap.add_argument("--std", type=str, default=None, help="Override std, comma-separated (e.g., 0.2023,0.1994,0.2010).")
    ap.add_argument("--pgd_steps", type=int, default=0, help="If >0, generate PGD adversarial examples for each batch.")
    ap.add_argument("--pgd_eps", type=float, default=8/255, help="Linf epsilon for PGD.")
    ap.add_argument("--pgd_alpha", type=float, default=2/255, help="PGD step size.")
    ap.add_argument("--pgd_random_start", action="store_true", help="Random start within the Linf ball.")
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.run_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_names = [m.strip() for m in args.model_names.split(",") if m.strip()]
    # Ensure we have two names when ckpt2 is provided
    if args.ckpt2 and len(model_names) == 1:
        model_names.append("modelB")

    models: List[tuple[str, nn.Module]] = []
    # Model A
    weights_a = ResNet18_Weights.IMAGENET1K_V1 if args.ckpt is None else None
    model_a = build_model(device, imagenet_stem=args.imagenet_stem, weights=weights_a)
    if args.ckpt:
        print(f"[INFO] Loading checkpoint A: {args.ckpt}")
        load_state_dict_flexible(model_a, args.ckpt)
    models.append((model_names[0] if model_names else "modelA", model_a))

    # Model B (optional)
    if args.ckpt2:
        weights_b = None  # when ckpt2 provided, don't load ImageNet weights
        model_b = build_model(device, imagenet_stem=args.imagenet_stem, weights=weights_b)
        print(f"[INFO] Loading checkpoint B: {args.ckpt2}")
        load_state_dict_flexible(model_b, args.ckpt2)
        name_b = model_names[1] if len(model_names) > 1 else "modelB"
        models.append((name_b, model_b))

    # Prepare transforms: raw tensors for label_img, normalized tensors for the model
    resize = transforms.Resize((args.img_size, args.img_size))
    to_tensor = transforms.ToTensor()

    # Normalization (matches your checkpoint)
    if args.mean and args.std:
        mean_list = parse_vector3(args.mean)
        std_list = parse_vector3(args.std)
        mean = torch.tensor(mean_list, device=device).view(1, 3, 1, 1)
        std = torch.tensor(std_list, device=device).view(1, 3, 1, 1)
        print(f"[INFO] Using custom norm mean={mean_list}, std={std_list}")
    elif args.norm == "imagenet":
        mean, std = resolve_imagenet_stats(device)
        print("[INFO] Using ImageNet normalization")
    else:
        mean, std = resolve_cifar_stats(device)
        print("[INFO] Using CIFAR normalization")

    feats: List[torch.Tensor] = []
    images: List[torch.Tensor] = []
    metadata: List[List[str]] = []
    eval_stats_clean: Dict[tuple[str, str], Dict[str, int]] = {}
    eval_stats_adv: Dict[tuple[str, str], Dict[str, int]] = {}

    splits = [s.strip().lower() for s in args.splits.split(",") if s.strip()]
    if not splits:
        splits = ["test"]

    max_per_split = None if args.max_samples_per_split <= 0 else args.max_samples_per_split

    with torch.no_grad():
        for model_name, model in models:
            for split in splits:
                is_train = split == "train"
                dataset = datasets.CIFAR10(
                    root=args.data_root,
                    train=is_train,
                    download=True,
                    transform=transforms.Compose([resize, to_tensor]),
                )
                loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

                seen = 0
                for batch_imgs, batch_labels in loader:
                    if max_per_split is not None and seen >= max_per_split:
                        break

                    raw = batch_imgs  # [B,3,H,W] in [0,1]
                    inputs = (raw.to(device) - mean) / std

                    penultimate, logits = forward_penultimate_and_logits(model, inputs)
                    preds = logits.argmax(dim=1)

                    feats.append(penultimate.cpu())  # [B,512]
                    if not args.no_sprites:
                        images.append(raw)  # keep unnormalized for nicer thumbnails
                    for lbl, pred in zip(batch_labels.tolist(), preds.tolist()):
                        # step=0 since this script logs clean images only; included for Projector "Color by step" convenience
                        metadata.append([model_name, split, "0", str(lbl), CLASSES[lbl], str(pred), CLASSES[pred]])

                    # Eval stats (clean)
                    key = (model_name, split)
                    if key not in eval_stats_clean:
                        eval_stats_clean[key] = {"correct": 0, "total": 0}
                    eval_stats_clean[key]["correct"] += int((preds.cpu() == batch_labels).sum().item())
                    eval_stats_clean[key]["total"] += batch_labels.size(0)

                    # PGD adversarial pass (final step only)
                    if args.pgd_steps > 0:
                        adv_images = pgd_attack(
                            model=model,
                            x=raw.to(device),
                            y=batch_labels.to(device),
                            mean=mean,
                            std=std,
                            eps=args.pgd_eps,
                            alpha=args.pgd_alpha,
                            steps=args.pgd_steps,
                            random_start=args.pgd_random_start,
                        )
                        pen_adv, logits_adv = forward_penultimate_and_logits(model, (adv_images - mean) / std)
                        preds_adv = logits_adv.argmax(dim=1)

                        feats.append(pen_adv.cpu())
                        if not args.no_sprites:
                            images.append(adv_images.cpu())
                        for lbl, pred in zip(batch_labels.tolist(), preds_adv.tolist()):
                            metadata.append([model_name, split, str(args.pgd_steps), str(lbl), CLASSES[lbl], str(pred), CLASSES[pred]])

                        # Eval stats (adv)
                        if key not in eval_stats_adv:
                            eval_stats_adv[key] = {"correct": 0, "total": 0}
                        eval_stats_adv[key]["correct"] += int((preds_adv.cpu() == batch_labels).sum().item())
                        eval_stats_adv[key]["total"] += batch_labels.size(0)

                    seen += batch_labels.shape[0]

    mat = torch.cat(feats, dim=0)
    imgs = torch.cat(images, dim=0) if (images and not args.no_sprites) else None

    writer = SummaryWriter(args.run_dir)
    writer.add_embedding(
        mat=mat,
        metadata=metadata,
        metadata_header=["model", "split", "step", "label_id", "label", "pred_id", "pred_label"],
        label_img=imgs,
        tag=args.tag,
    )
    writer.flush()
    writer.close()

    # Report accuracies on the (possibly truncated) subsets used for logging
    print("\nEval on logged samples (clean):")
    for key in sorted(eval_stats_clean.keys()):
        correct = eval_stats_clean[key]["correct"]
        total = eval_stats_clean[key]["total"]
        acc = 100.0 * correct / max(total, 1)
        model_name, split = key
        print(f"  {model_name} [{split}]: {correct}/{total} = {acc:.2f}%")

    if args.pgd_steps > 0 and eval_stats_adv:
        print(f"\nEval on logged samples (PGD, {args.pgd_steps} steps, eps={args.pgd_eps}, alpha={args.pgd_alpha}):")
        for key in sorted(eval_stats_adv.keys()):
            correct = eval_stats_adv[key]["correct"]
            total = eval_stats_adv[key]["total"]
            acc = 100.0 * correct / max(total, 1)
            model_name, split = key
            print(f"  {model_name} [{split}]: {correct}/{total} = {acc:.2f}%")

    if args.no_sprites:
        print("\nSprites disabled (--no_sprites); Projector will render points only.")

    print("\nLogged embeddings to TensorBoard.")
    print(f"Logdir: {args.run_dir}")
    print("Launch projector:")
    print(f"  tensorboard --logdir {args.run_dir}")
    print("In the Projector tab: set 'Color by' = label to color points by CIFAR-10 class.\n")


if __name__ == "__main__":
    main()
