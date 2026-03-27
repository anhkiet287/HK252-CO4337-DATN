"""Visualize PGD step trajectories in a model's hidden space."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from ardg.experiments.common import build_loaders, load_model_from_checkpoint, setup_run
from ardg.data.transforms import get_dataset_stats
from ardg.models.factory import build_model
from ardg.utils.paths import ensure_dir


def _get_module_by_name(model: torch.nn.Module, name: str) -> torch.nn.Module:
    """Resolve a module by its dotted name.

    Args:
        model: Model instance to search.
        name: Module name (e.g., "layer4", "layer4.1", "fc").

    Returns:
        Resolved module.

    Raises:
        ValueError: If the module name is not found.
    """
    for module_name, module in model.named_modules():
        if module_name == name:
            return module
    raise ValueError(f"Module '{name}' not found in model.")


def _forward_with_embedding(
    model: torch.nn.Module,
    layer: torch.nn.Module,
    images: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run a forward pass and capture a layer's activations.

    Args:
        model: Model instance.
        layer: Module to hook for embeddings.
        images: Input tensor.

    Returns:
        Tuple of (logits, embeddings).
    """
    bucket: Dict[str, torch.Tensor] = {}

    def _hook(_: torch.nn.Module, __: Tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        bucket["emb"] = output.detach()

    handle = layer.register_forward_hook(_hook)
    logits = model(images)
    handle.remove()
    if "emb" not in bucket:
        raise RuntimeError("Embedding hook did not capture activations.")
    return logits, bucket["emb"]


def _collect_pgd_embeddings(
    model: torch.nn.Module,
    layer: torch.nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    attack_cfg: Dict[str, Any],
    include_clean: bool,
) -> Tuple[np.ndarray, np.ndarray, List[str], torch.Tensor]:
    """Collect embeddings across PGD steps for one batch.

    Args:
        model: Model instance.
        layer: Module to hook for embeddings.
        images: Input batch.
        labels: Label batch.
        attack_cfg: Attack config with eps/step_size/num_steps.
        include_clean: Whether to include clean embeddings at step 0.

    Returns:
        Tuple of (embeddings, steps, step_kinds).
    """
    eps = float(attack_cfg["eps"])
    alpha = float(attack_cfg["step_size"])
    steps = int(attack_cfg["num_steps"])

    embeddings: List[torch.Tensor] = []
    step_ids: List[int] = []
    step_kinds: List[str] = []
    images_by_step: List[torch.Tensor] = []

    model.eval()
    if include_clean:
        _, emb = _forward_with_embedding(model, layer, images)
        embeddings.append(emb)
        step_ids.append(0)
        step_kinds.append("clean")
        images_by_step.append(images.detach())

    x = images.detach()
    x_adv = x + torch.empty_like(x).uniform_(-eps, eps)
    for step in range(1, steps + 1):
        x_adv.requires_grad_(True)
        logits, emb = _forward_with_embedding(model, layer, x_adv)
        loss = F.cross_entropy(logits, labels)
        grad = torch.autograd.grad(loss, x_adv, retain_graph=False, create_graph=False)[0]
        x_adv = x_adv.detach() + alpha * torch.sign(grad.detach())
        delta = torch.clamp(x_adv - x, min=-eps, max=eps)
        x_adv = (x + delta).detach()
        embeddings.append(emb)
        step_ids.append(step)
        step_kinds.append("pgd")
        images_by_step.append(x_adv.detach())

    emb_array = torch.stack(embeddings, dim=0).cpu().numpy()
    step_array = np.array(step_ids, dtype=np.int64)
    images_tensor = torch.stack(images_by_step, dim=0)
    return emb_array, step_array, step_kinds, images_tensor


def _save_outputs(
    output_dir: str,
    run_name: str,
    embeddings: np.ndarray,
    steps: np.ndarray,
    step_kinds: List[str],
    labels: np.ndarray,
    meta: Dict[str, Any],
) -> None:
    """Save embeddings and metadata to disk.

    Args:
        output_dir: Output directory.
        run_name: Run name prefix for files.
        embeddings: Embedding array of shape (S, B, ...).
        steps: Step index array.
        step_kinds: Step labels.
        labels: Ground-truth labels.
        meta: Metadata dictionary.

    Returns:
        None.
    """
    ensure_dir(output_dir)
    out_npz = Path(output_dir) / f"{run_name}_pgd_steps.npz"
    out_json = Path(output_dir) / f"{run_name}_pgd_steps.json"
    np.savez_compressed(
        out_npz,
        embeddings=embeddings,
        steps=steps,
        step_kinds=np.array(step_kinds),
        labels=labels,
    )
    with out_json.open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)


def _plot_pca(output_dir: str, run_name: str, embeddings: np.ndarray, steps: np.ndarray) -> None:
    """Plot a 2D PCA projection if optional deps are available.

    Args:
        output_dir: Output directory.
        run_name: Run name prefix for files.
        embeddings: Embedding array of shape (S, B, ...).
        steps: Step indices for each slice.

    Returns:
        None.
    """
    try:
        import matplotlib.pyplot as plt  # type: ignore
        from sklearn.decomposition import PCA  # type: ignore
    except Exception:
        return

    flat = embeddings.reshape(embeddings.shape[0] * embeddings.shape[1], -1)
    pca = PCA(n_components=2, random_state=0)
    proj = pca.fit_transform(flat)

    step_rep = np.repeat(steps, embeddings.shape[1])
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(proj[:, 0], proj[:, 1], c=step_rep, s=8, cmap="viridis")
    plt.colorbar(scatter, label="PGD step")
    plt.title("PCA projection of PGD steps")
    plt.tight_layout()
    out_path = Path(output_dir) / f"{run_name}_pgd_steps_pca.png"
    plt.savefig(out_path, dpi=150)
    plt.close()


def _log_wandb_steps(
    images: torch.Tensor,
    embeddings: np.ndarray,
    steps: np.ndarray,
    labels: np.ndarray,
    step_kinds: List[str],
    max_images: int,
    projection: str,
) -> None:
    """Log per-step images and embeddings to W&B.

    Args:
        images: Tensor of shape (S, B, C, H, W).
        embeddings: Array of shape (S, B, ...).
        steps: Step indices.
        labels: Label array for the batch.
        step_kinds: Step labels.
        max_images: Max images per step to log.

    Returns:
        None.
    """
    try:
        import wandb  # type: ignore
    except Exception:
        return

    if getattr(wandb, "run", None) is None:
        return

    step_count, batch_size = images.shape[0], images.shape[1]
    max_images = min(max_images, batch_size)
    pca_xy = _maybe_project_embeddings(embeddings, projection)
    table_columns = ["step", "kind", "label", "image", "embedding"]
    if pca_xy is not None:
        table_columns.extend(["x", "y"])
    table = wandb.Table(columns=table_columns)
    for step_idx in range(step_count):
        step_id = int(steps[step_idx])
        kind = step_kinds[step_idx]
        for img_idx in range(max_images):
            row = [
                step_id,
                kind,
                int(labels[img_idx]),
                wandb.Image(images[step_idx, img_idx].cpu()),
                embeddings[step_idx, img_idx].reshape(-1),
            ]
            if pca_xy is not None:
                point_idx = step_idx * batch_size + img_idx
                row.extend([float(pca_xy[point_idx, 0]), float(pca_xy[point_idx, 1])])
            table.add_data(
                *row,
            )
    wandb.log({"pgd_steps": table})
    if pca_xy is not None:
        wandb.log(
            {
                "pgd_steps_scatter_step": wandb.plot.scatter(table, "x", "y", "step"),
                "pgd_steps_scatter_kind": wandb.plot.scatter(table, "x", "y", "kind"),
            }
        )


def _maybe_project_embeddings(embeddings: np.ndarray, projection: str) -> np.ndarray | None:
    """Project embeddings to 2D for scatter plots.

    Args:
        embeddings: Embedding array of shape (S, B, ...).
        projection: Projection name ("pca" or "none").

    Returns:
        Array of shape (S*B, 2) or None.
    """
    if projection.lower() != "pca":
        return None
    try:
        from sklearn.decomposition import PCA  # type: ignore
    except Exception:
        return None
    flat = embeddings.reshape(embeddings.shape[0] * embeddings.shape[1], -1)
    pca = PCA(n_components=2, random_state=0)
    return pca.fit_transform(flat)


def _denormalize_images(images: torch.Tensor, dataset_name: str) -> torch.Tensor:
    """Undo dataset normalization for visualization.

    Args:
        images: Image tensor of shape (S, B, C, H, W).
        dataset_name: Dataset name for mean/std lookup.

    Returns:
        Denormalized image tensor in [0, 1].
    """
    mean, std = get_dataset_stats(dataset_name)
    mean_t = torch.tensor(mean, device=images.device).view(1, 1, 3, 1, 1)
    std_t = torch.tensor(std, device=images.device).view(1, 1, 3, 1, 1)
    images = images * std_t + mean_t
    return images.clamp(0.0, 1.0)


def _plotly_scatter(
    output_dir: str,
    run_name: str,
    embeddings: np.ndarray,
    steps: np.ndarray,
    labels: np.ndarray,
    step_kinds: List[str],
) -> None:
    """Create a Plotly scatter with marker shapes per step kind.

    Args:
        output_dir: Output directory.
        run_name: Run name prefix for files.
        embeddings: Embedding array of shape (S, B, ...).
        steps: Step indices for each slice.
        labels: Label array for the batch.
        step_kinds: Step labels.

    Returns:
        None.
    """
    try:
        import plotly.express as px  # type: ignore
    except Exception:
        return

    proj = _maybe_project_embeddings(embeddings, "pca")
    if proj is None:
        return

    step_rep = np.repeat(steps, embeddings.shape[1])
    kind_rep = np.repeat(np.array(step_kinds), embeddings.shape[1])
    label_rep = np.tile(labels, embeddings.shape[0])

    data = {
        "x": proj[:, 0],
        "y": proj[:, 1],
        "step": step_rep,
        "kind": kind_rep,
        "label": label_rep,
    }
    fig = px.scatter(
        data_frame=data,
        x="x",
        y="y",
        color="label",
        symbol="kind",
        title="PGD steps in hidden space (PCA)",
    )
    out_path = Path(output_dir) / f"{run_name}_pgd_steps_plotly.html"
    fig.write_html(out_path)


def _plotly_paths(
    output_dir: str,
    run_name: str,
    embeddings: np.ndarray,
    steps: np.ndarray,
    labels: np.ndarray,
) -> None:
    """Create a Plotly path plot from clean to final step.

    Args:
        output_dir: Output directory.
        run_name: Run name prefix for files.
        embeddings: Embedding array of shape (S, B, ...).
        steps: Step indices for each slice.
        labels: Label array for the batch.

    Returns:
        None.
    """
    try:
        import plotly.graph_objects as go  # type: ignore
    except Exception:
        return

    proj = _maybe_project_embeddings(embeddings, "pca")
    if proj is None:
        return

    step_count, batch_size = embeddings.shape[0], embeddings.shape[1]
    fig = go.Figure()
    for sample_idx in range(batch_size):
        start = sample_idx
        stride = batch_size
        points = proj[start::stride]
        if points.shape[0] != step_count:
            continue
        fig.add_trace(
            go.Scatter(
                x=points[:, 0],
                y=points[:, 1],
                mode="lines+markers",
                name=f"label_{int(labels[sample_idx])}_sample_{sample_idx}",
            )
        )
    fig.update_layout(title="PGD path from clean to final step (PCA)")
    out_path = Path(output_dir) / f"{run_name}_pgd_steps_path.html"
    fig.write_html(out_path)


def main() -> None:
    """Run a simple PGD-step visualization pipeline.

    Returns:
        None.
    """
    cfg_path = "configs/visualize_pgd_steps.yaml"
    cfg, logger, run, device = setup_run(cfg_path)
    train_loader, _, _ = build_loaders(cfg)
    viz_cfg = cfg.get("visualize", {})
    max_batches = int(viz_cfg.get("max_batches", 1))
    max_samples = int(viz_cfg.get("max_samples", 0))
    layer_name = str(viz_cfg.get("layer", "layer4"))
    include_clean = bool(viz_cfg.get("include_clean", True))
    output_dir = str(viz_cfg.get("output_dir", "outputs/visualize"))
    checkpoint = viz_cfg.get("checkpoint")
    log_wandb = bool(viz_cfg.get("log_wandb", True))
    max_log_images = int(viz_cfg.get("max_log_images", 8))
    projection = str(viz_cfg.get("projection", "pca"))

    if checkpoint:
        model = load_model_from_checkpoint(cfg, checkpoint, device)
    else:
        model = build_model(cfg)
        model.to(device)

    layer = _get_module_by_name(model, layer_name)
    attack_cfg = cfg["attack"]["train"]
    run_name = cfg.get("logging", {}).get("run_name", "visualize")

    all_embeddings: List[np.ndarray] = []
    all_steps: List[np.ndarray] = []
    all_kinds: List[str] = []
    all_labels: List[np.ndarray] = []

    for batch_idx, (images, labels) in enumerate(train_loader, start=1):
        images = images.to(device)
        labels = labels.to(device)
        if max_samples:
            images = images[:max_samples]
            labels = labels[:max_samples]
        emb, steps, kinds, images_by_step = _collect_pgd_embeddings(
            model,
            layer,
            images,
            labels,
            attack_cfg,
            include_clean=include_clean,
        )
        if log_wandb:
            images_for_log = _denormalize_images(images_by_step, cfg["dataset"]["name"])
            _log_wandb_steps(
                images_for_log,
                emb,
                steps,
                labels.detach().cpu().numpy(),
                kinds,
                max_log_images,
                projection,
            )
        all_embeddings.append(emb)
        all_steps.append(steps)
        all_kinds.extend(kinds)
        all_labels.append(labels.detach().cpu().numpy())
        if batch_idx >= max_batches:
            break

    embeddings = np.concatenate(all_embeddings, axis=1)
    steps = all_steps[0]
    labels = np.concatenate(all_labels, axis=0)
    meta = {
        "dataset": cfg["dataset"]["name"],
        "model": cfg["model"]["name"],
        "layer": layer_name,
        "attack": attack_cfg,
        "platform": cfg.get("experiment", {}).get("platform"),
        "run_name": run_name,
    }

    _save_outputs(output_dir, run_name, embeddings, steps, all_kinds, labels, meta)
    _plot_pca(output_dir, run_name, embeddings, steps)
    _plotly_scatter(output_dir, run_name, embeddings, steps, labels, all_kinds)
    _plotly_paths(output_dir, run_name, embeddings, steps, labels)
    logger.info("Saved PGD embeddings to %s", output_dir)
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
