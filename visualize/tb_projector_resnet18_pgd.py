import os
import argparse
import random
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torchvision.models import resnet18

# -------------------------
# CIFAR10 constants
# -------------------------
CIFAR10_MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
CIFAR10_STD  = torch.tensor([0.2023, 0.1994, 0.2010]).view(1, 3, 1, 1)

CLASSES = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize(x: torch.Tensor) -> torch.Tensor:
    mean = CIFAR10_MEAN.to(x.device)
    std  = CIFAR10_STD.to(x.device)
    return (x - mean) / std


def build_resnet18_cifar(num_classes: int = 10, imagenet_stem: bool = False) -> nn.Module:
    """
    CIFAR-style stem is typical for CIFAR10 ResNet18:
      - conv1: 3x3 stride 1
      - remove maxpool
    If your checkpoints used default ImageNet stem, pass imagenet_stem=True.
    """
    m = resnet18(num_classes=num_classes)
    if not imagenet_stem:
        m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        m.maxpool = nn.Identity()
    return m


def parse_float_expr(x):
    """
    Accept plain floats or simple ratio strings like '8/255'.
    """
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str) and "/" in x:
        num, den = x.split("/", 1)
        return float(num) / float(den)
    return float(x)


import torch
import torch.nn as nn

def _strip_module_prefix(state_dict):
    out = {}
    for k, v in state_dict.items():
        nk = k[len("module."):] if k.startswith("module.") else k
        out[nk] = v
    return out

def _is_state_dict_like(d):
    # Heuristic: state_dict thường có rất nhiều tensor params
    if not isinstance(d, dict) or len(d) == 0:
        return False
    tensor_cnt = sum(torch.is_tensor(v) for v in d.values())
    return tensor_cnt >= max(5, int(0.5 * len(d)))  # khá an toàn

def extract_state_dict(ckpt_obj):
    """
    Support common checkpoint formats:
    - raw state_dict (dict of tensors)
    - dict with keys: state_dict / model_state_dict / model / net / network / encoder / ...
    - nested dict: {'model': {'state_dict': ...}}
    """
    if _is_state_dict_like(ckpt_obj):
        return ckpt_obj

    if isinstance(ckpt_obj, dict):
        # 1) common top-level keys
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
                # nested: model -> state_dict
                if isinstance(v, dict):
                    for kk in ["state_dict", "model_state_dict"]:
                        if kk in v and _is_state_dict_like(v[kk]):
                            return v[kk]

        # 2) fallback: try to find first dict that looks like state_dict
        for _, v in ckpt_obj.items():
            if _is_state_dict_like(v):
                return v

        raise KeyError(
            f"Can't find a state_dict in checkpoint. Available keys: {list(ckpt_obj.keys())[:30]}"
        )

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

    return obj  # trả về để bạn có thể đọc epoch/acc nếu cần

@torch.no_grad()
def clamp01(x: torch.Tensor) -> torch.Tensor:
    return x.clamp(0.0, 1.0)


def pgd_trajectory(
    model: nn.Module,
    x0: torch.Tensor,
    y: torch.Tensor,
    eps: float,
    alpha: float,
    steps: int,
    random_start: bool
) -> List[torch.Tensor]:
    """
    Returns [x_clean, x_1, ..., x_steps], all in [0,1].
    PGD is done in pixel space; normalization is applied only when feeding the model.
    """
    model.eval()
    x_clean = x0.detach()

    # init for adversarial
    x_adv = x_clean.detach()
    if random_start:
        x_adv = x_adv + torch.empty_like(x_adv).uniform_(-eps, eps)
        x_adv = clamp01(x_adv)

    traj = [x_clean]  # step 0 = clean

    for _ in range(steps):
        x_adv.requires_grad_(True)
        logits = model(normalize(x_adv))
        loss = F.cross_entropy(logits, y)
        grad = torch.autograd.grad(loss, x_adv, only_inputs=True)[0]

        with torch.no_grad():
            x_adv = x_adv + alpha * grad.sign()
            # project to Linf ball around clean
            x_adv = torch.max(torch.min(x_adv, x_clean + eps), x_clean - eps)
            x_adv = clamp01(x_adv)

        traj.append(x_adv.detach())

    return traj


def forward_collect_features(model: nn.Module, x: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """
    Collects:
      - layer3_gap: GAP(layer3 output) -> [B, 256]
      - layer4_gap: GAP(layer4 output) -> [B, 512]
      - penultimate: flatten(avgpool output) -> [B, 512]
      - logits: [B, 10]
    One forward pass, multiple hooks.
    """
    feats: Dict[str, torch.Tensor] = {}
    hooks = []

    def hook_layer3(m, inp, out):
        feats["layer3_gap"] = out.mean(dim=(2, 3)).detach()

    def hook_layer4(m, inp, out):
        feats["layer4_gap"] = out.mean(dim=(2, 3)).detach()

    def hook_avgpool(m, inp, out):
        feats["penultimate"] = out.flatten(1).detach()

    hooks.append(model.layer3.register_forward_hook(hook_layer3))
    hooks.append(model.layer4.register_forward_hook(hook_layer4))
    hooks.append(model.avgpool.register_forward_hook(hook_avgpool))

    logits = model(normalize(x)).detach()
    feats["logits"] = logits

    for h in hooks:
        h.remove()

    # sanity
    for k in ["layer3_gap", "layer4_gap", "penultimate", "logits"]:
        if k not in feats:
            raise RuntimeError(f"Missing feature '{k}'. Check your ResNet definition.")
    return feats, logits


def parse_label_arg(arg: str) -> int:
    """
    Accepts either class name or numeric id string. Returns label id.
    """
    if arg.isdigit():
        v = int(arg)
        if 0 <= v < len(CLASSES):
            return v
    else:
        if arg in CLASSES:
            return CLASSES.index(arg)
    raise ValueError(f"Invalid label '{arg}'. Use 0-9 or one of: {CLASSES}")


def pick_samples(ds, count: int, fixed_label: int | None = None) -> List[Tuple[str, int]]:
    """
    If fixed_label is provided, pick `count` samples of that label.
    If fixed_label is None and count==4, mimic the old A/B/C/D diverse pick.
    Else, pick the first `count` samples sequentially.
    """
    if fixed_label is not None:
        idxs = [i for i in range(len(ds)) if ds[i][1] == fixed_label]
        if len(idxs) < count:
            raise RuntimeError(f"Requested {count} samples of label {CLASSES[fixed_label]} but only found {len(idxs)}.")
        return [(f"{CLASSES[fixed_label]}_{k}", idxs[k]) for k in range(count)]

    if count == 4:
        # Original A/B/C/D selection for variety
        from collections import defaultdict
        idxs_by_label = defaultdict(list)
        for i in range(len(ds)):
            _, lab = ds[i]
            idxs_by_label[lab].append(i)
            if any(len(v) >= 2 for v in idxs_by_label.values()) and len(idxs_by_label) >= 3 and i > 2000:
                break
        base_label = None
        for lab, idxs in idxs_by_label.items():
            if len(idxs) >= 2:
                base_label = lab
                break
        if base_label is None:
            raise RuntimeError("Could not find a label with >=2 samples (unexpected for CIFAR10).")
        a_idx, b_idx = idxs_by_label[base_label][0], idxs_by_label[base_label][1]
        other_labels = [l for l in idxs_by_label.keys() if l != base_label]
        if len(other_labels) < 2:
            other_labels = [l for l in range(10) if l != base_label]
        c_label = other_labels[0]
        d_label = other_labels[1]
        c_idx = idxs_by_label[c_label][0] if len(idxs_by_label[c_label]) > 0 else 0
        d_idx = idxs_by_label[d_label][1] if len(idxs_by_label[d_label]) > 1 else idxs_by_label[d_label][0]
        return [("A", a_idx), ("B", b_idx), ("C", c_idx), ("D", d_idx)]

    # Fallback: first `count` samples
    return [(f"sample_{k}", k) for k in range(min(count, len(ds)))]


def margin_from_logits(logits: torch.Tensor, y: int) -> float:
    """
    margin = logit_true - max(logit_other)
    logits: [10]
    """
    true = logits[y].item()
    other = torch.cat([logits[:y], logits[y+1:]]).max().item()
    return float(true - other)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean_ckpt", type=str, required=True)
    ap.add_argument("--adv_ckpt", type=str, required=True)
    ap.add_argument("--run_dir", type=str, default="runs/projector_pgd_resnet18")
    ap.add_argument("--data_root", type=str, default="./data")
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--eps", type=str, default="8/255")
    ap.add_argument("--alpha", type=str, default="2/255")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--random_start", action="store_true")
    ap.add_argument("--imagenet_stem", action="store_true", help="Use default torchvision stem if your ckpt was trained that way.")
    ap.add_argument("--sample_label", type=str, default=None, help="Optional: pick samples from this label (name or id).")
    ap.add_argument("--num_samples", type=int, default=4, help="Number of samples to visualize.")

    args = ap.parse_args()
    args.eps = parse_float_expr(args.eps)
    args.alpha = parse_float_expr(args.alpha)
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.run_dir, exist_ok=True)
    writer = SummaryWriter(args.run_dir)

    # Dataset in [0,1]
    ds = datasets.CIFAR10(root=args.data_root, train=False, download=True, transform=transforms.ToTensor())
    fixed_label = parse_label_arg(args.sample_label) if args.sample_label is not None else None
    picked = pick_samples(ds, count=args.num_samples, fixed_label=fixed_label)

    # Load models
    clean_model = build_resnet18_cifar(num_classes=10, imagenet_stem=args.imagenet_stem)
    adv_model   = build_resnet18_cifar(num_classes=10, imagenet_stem=args.imagenet_stem)

    load_state_dict_flexible(clean_model, args.clean_ckpt)
    load_state_dict_flexible(adv_model, args.adv_ckpt)

    clean_model.to(device).eval()
    adv_model.to(device).eval()

    models = [("clean", clean_model), ("adv", adv_model)]

    # Metadata columns (include attack config to filter runs in Projector)
    attack_cfg = f"pgd_eps={args.eps}_alpha={args.alpha}_steps={args.steps}_rs={int(args.random_start)}"
    header = ["model","image_id","step","true_label","pred","prob_true","margin","loss","linf","attack"]

    # Buffers per feature-space for embeddings
    feats_buffers = {
        "penultimate_xt": [],
        "layer4_xt": [],
        "layer3_xt": [],
        "logits_xt": [],
        "penultimate_delta": [],  # same feature as penultimate_xt, different label_img (delta visualization)
    }
    imgs_xt: List[torch.Tensor] = []
    imgs_delta: List[torch.Tensor] = []
    metadata_rows: List[List[str]] = []

    # For drift scalar logs: keep feature_0 per (model,image_id,space)
    # We'll compute drift online inside the loop.

    for model_name, model in models:
        for image_id, idx in picked:
            x0, y0 = ds[idx]
            x0 = x0.unsqueeze(0).to(device)  # [1,3,32,32]
            y = torch.tensor([y0], device=device)

            traj = pgd_trajectory(
                model=model,
                x0=x0,
                y=y,
                eps=args.eps,
                alpha=args.alpha,
                steps=args.steps,
                random_start=args.random_start
            )

            # step 0 features baseline
            base_feats, base_logits = forward_collect_features(model, traj[0])
            base = {
                "penultimate": base_feats["penultimate"].squeeze(0),
                "layer4_gap": base_feats["layer4_gap"].squeeze(0),
                "layer3_gap": base_feats["layer3_gap"].squeeze(0),
                "logits": base_feats["logits"].squeeze(0),
            }

            for t, xt in enumerate(traj):
                feats, logits = forward_collect_features(model, xt)
                logits1 = logits.squeeze(0)
                probs = logits1.softmax(dim=0)

                pred = int(probs.argmax().item())
                prob_true = float(probs[y0].item())
                loss = float(F.cross_entropy(logits, y).item())
                linf = float((xt - x0).abs().max().item())
                margin = margin_from_logits(logits1, y0)

                # feature drift per space
                pen = feats["penultimate"].squeeze(0)
                l4  = feats["layer4_gap"].squeeze(0)
                l3  = feats["layer3_gap"].squeeze(0)
                log = feats["logits"].squeeze(0)

                drift_pen = float(torch.norm(pen - base["penultimate"], p=2).item())
                drift_l4  = float(torch.norm(l4  - base["layer4_gap"], p=2).item())
                drift_l3  = float(torch.norm(l3  - base["layer3_gap"], p=2).item())
                drift_log = float(torch.norm(log - base["logits"], p=2).item())

                # Scalars for sanity checks
                writer.add_scalar(f"linf/{model_name}_{image_id}", linf, t)
                writer.add_scalar(f"prob_true/{model_name}_{image_id}", prob_true, t)
                writer.add_scalar(f"margin/{model_name}_{image_id}", margin, t)

                writer.add_scalar(f"feat_drift_L2_penultimate/{model_name}_{image_id}", drift_pen, t)
                writer.add_scalar(f"feat_drift_L2_layer4/{model_name}_{image_id}", drift_l4, t)
                writer.add_scalar(f"feat_drift_L2_layer3/{model_name}_{image_id}", drift_l3, t)
                writer.add_scalar(f"logit_drift_L2/{model_name}_{image_id}", drift_log, t)

                # Metadata row (strings)
                row = [
                    model_name,
                    image_id,
                    str(t),
                    CLASSES[y0],
                    CLASSES[pred],
                    f"{prob_true:.4f}",
                    f"{margin:.4f}",
                    f"{loss:.4f}",
                    f"{linf:.5f}",
                    attack_cfg,
                ]
                metadata_rows.append(row)

                # Images for sprite: xt
                imgs_xt.append(xt.squeeze(0).detach().cpu())

                # Delta visualization for sprite: (xt-x0)/eps mapped to [0,1]
                with torch.no_grad():
                    delta = (xt - x0) / max(args.eps, 1e-12)
                    delta = delta.clamp(-1.0, 1.0)
                    delta_vis = 0.5 + 0.5 * delta
                imgs_delta.append(delta_vis.squeeze(0).detach().cpu())

                # Append features to buffers
                feats_buffers["penultimate_xt"].append(pen.detach().cpu())
                feats_buffers["layer4_xt"].append(l4.detach().cpu())
                feats_buffers["layer3_xt"].append(l3.detach().cpu())
                feats_buffers["logits_xt"].append(log.detach().cpu())
                feats_buffers["penultimate_delta"].append(pen.detach().cpu())  # same penultimate vectors

    # Stack and write embeddings
    imgs_xt_t = torch.stack(imgs_xt, dim=0)      # [N,3,32,32]
    imgs_dl_t = torch.stack(imgs_delta, dim=0)   # [N,3,32,32]

    def write_embedding(tag: str, mat_list: List[torch.Tensor], label_img: torch.Tensor):
        mat = torch.stack(mat_list, dim=0)  # [N,D]
        writer.add_embedding(
            mat=mat,
            metadata=metadata_rows,
            metadata_header=header,
            label_img=label_img,
            tag=tag
        )

    write_embedding("penultimate_xt", feats_buffers["penultimate_xt"], imgs_xt_t)
    write_embedding("penultimate_delta", feats_buffers["penultimate_delta"], imgs_dl_t)
    write_embedding("layer4_xt", feats_buffers["layer4_xt"], imgs_xt_t)
    write_embedding("layer3_xt", feats_buffers["layer3_xt"], imgs_xt_t)
    write_embedding("logits_xt", feats_buffers["logits_xt"], imgs_xt_t)

    writer.flush()
    writer.close()

    print("\nDone.")
    print(f"Run dir: {args.run_dir}")
    print("Launch TensorBoard:")
    print(f"  tensorboard --logdir {args.run_dir}\n")
    print("Projector tips:")
    print("  - Color by: 'step' or 'model'")
    print("  - Filter/search by image_id: A/B/C/D to isolate a trajectory")
    print("  - Click point to see thumbnail; use penultimate_delta to see perturb pattern clearly.")


if __name__ == "__main__":
    main()
