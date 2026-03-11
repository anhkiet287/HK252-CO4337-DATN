"""Lightweight clustering helpers for GroupDRO++ draft."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import torch


def run_kmeans(embeddings: torch.Tensor, num_clusters: int, num_iters: int = 20) -> torch.Tensor:
    """Run simple k-means on CPU/GPU depending on tensor device."""
    labels, _ = run_kmeans_with_centers(embeddings, num_clusters, num_iters=num_iters)
    return labels


def run_kmeans_with_centers(
    embeddings: torch.Tensor,
    num_clusters: int,
    num_iters: int = 20,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run k-means and return both labels and final centers."""
    n, d = embeddings.shape
    if n < num_clusters:
        # Degenerate case: fewer samples than clusters.
        labels = torch.arange(n, device=embeddings.device) % num_clusters
        centers, _ = compute_cluster_centers(embeddings, labels, num_clusters)
        return labels, centers

    # Initialize centers via random subset.
    perm = torch.randperm(n, device=embeddings.device)
    centers = embeddings[perm[:num_clusters]]
    for _ in range(num_iters):
        # Assign
        distances = torch.cdist(embeddings, centers)
        labels = distances.argmin(dim=1)
        # Update
        new_centers = torch.zeros_like(centers)
        for k in range(num_clusters):
            mask = labels == k
            if mask.any():
                new_centers[k] = embeddings[mask].mean(dim=0)
            else:
                new_centers[k] = centers[k]  # keep old if empty
        if torch.allclose(new_centers, centers):
            break
        centers = new_centers
    return labels, centers


def compute_cluster_centers(
    embeddings: torch.Tensor,
    cluster_ids: torch.Tensor,
    num_clusters: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute cluster centers and counts for one clustered batch."""
    if embeddings.ndim != 2:
        raise ValueError(f"Expected 2D embeddings, got shape={tuple(embeddings.shape)}.")

    centers = torch.zeros(
        (num_clusters, embeddings.size(1)),
        device=embeddings.device,
        dtype=embeddings.dtype,
    )
    counts = torch.zeros(num_clusters, device=embeddings.device, dtype=torch.long)
    for idx in range(num_clusters):
        mask = cluster_ids == idx
        if not mask.any():
            continue
        centers[idx] = embeddings[mask].mean(dim=0)
        counts[idx] = int(mask.sum().item())
    return centers, counts


def assign_to_centers(embeddings: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    """Assign each embedding to the nearest center."""
    if embeddings.ndim != 2:
        raise ValueError(f"Expected 2D embeddings, got shape={tuple(embeddings.shape)}.")
    if centers.ndim != 2:
        raise ValueError(f"Expected 2D centers, got shape={tuple(centers.shape)}.")
    if centers.size(0) == 0:
        raise ValueError("Cannot assign to an empty center set.")
    distances = torch.cdist(embeddings, centers)
    return distances.argmin(dim=1)


def project_embeddings_pca(embeddings: torch.Tensor, n_components: int = 2) -> torch.Tensor:
    """Project embeddings to PCA space using a small CPU SVD."""
    n_components = max(1, int(n_components))
    emb = embeddings.detach().to(dtype=torch.float32, device="cpu")
    if emb.ndim != 2:
        raise ValueError(f"Expected 2D embeddings, got shape={tuple(emb.shape)}.")
    if emb.numel() == 0:
        return torch.zeros((0, n_components), dtype=torch.float32)

    emb = emb - emb.mean(dim=0, keepdim=True)
    rows, cols = emb.shape
    usable = min(rows, cols, n_components)
    if usable <= 0:
        return torch.zeros((rows, n_components), dtype=torch.float32)

    try:
        _, _, vh = torch.linalg.svd(emb, full_matrices=False)
        basis = vh[:usable].T
        proj = emb @ basis
    except Exception:
        proj = emb[:, :usable]

    if proj.size(1) < n_components:
        pad = torch.zeros((rows, n_components - proj.size(1)), dtype=proj.dtype)
        proj = torch.cat([proj, pad], dim=1)
    return proj[:, :n_components]


def save_cluster_snapshot(
    path: str | Path,
    *,
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    cluster_ids: torch.Tensor,
    centers: torch.Tensor,
    counts: torch.Tensor,
    q: torch.Tensor | None,
    epoch: int,
    batch_idx: int,
    split: str,
    projection_2d: torch.Tensor | None = None,
    is_adv: bool = False,
) -> Path:
    """Persist one batch clustering snapshot as a compressed NumPy archive."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "embeddings": embeddings.detach().cpu().numpy().astype(np.float32, copy=False),
        "labels": labels.detach().cpu().numpy(),
        "cluster_ids": cluster_ids.detach().cpu().numpy(),
        "centers": centers.detach().cpu().numpy().astype(np.float32, copy=False),
        "counts": counts.detach().cpu().numpy(),
        "q": (
            np.empty((0,), dtype=np.float32)
            if q is None
            else q.detach().cpu().numpy().astype(np.float32, copy=False)
        ),
        "epoch": np.array([int(epoch)], dtype=np.int64),
        "batch_idx": np.array([int(batch_idx)], dtype=np.int64),
        "split": np.array([str(split)]),
        "is_adv": np.array([int(bool(is_adv))], dtype=np.int64),
    }
    if projection_2d is not None:
        payload["projection_2d"] = projection_2d.detach().cpu().numpy().astype(np.float32, copy=False)

    np.savez_compressed(target, **payload)
    return target


@torch.no_grad()
def extract_embeddings(
    model: torch.nn.Module,
    loader: Iterable,
    device: torch.device,
    max_samples: int | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Collect embeddings (using model penultimate logits) and labels."""
    model.eval()
    feats = []
    labels = []
    for batch in loader:
        if isinstance(batch, dict):
            x, y = batch["x"], batch["y"]
        else:
            x, y = batch[0], batch[1]
        x = x.to(device)
        y = y.to(device)
        out = model(x)
        feats.append(out.detach())
        labels.append(y)
        if max_samples is not None:
            if sum(t.size(0) for t in feats) >= max_samples:
                break
    feats_t = torch.cat(feats, dim=0)
    labels_t = torch.cat(labels, dim=0)
    if max_samples is not None and feats_t.size(0) > max_samples:
        feats_t = feats_t[:max_samples]
        labels_t = labels_t[:max_samples]
    return feats_t, labels_t
