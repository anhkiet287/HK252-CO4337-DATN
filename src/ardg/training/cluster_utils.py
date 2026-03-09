"""Lightweight clustering helpers for GroupDRO++ draft."""

from __future__ import annotations

from typing import Iterable, Tuple

import torch


def run_kmeans(embeddings: torch.Tensor, num_clusters: int, num_iters: int = 20) -> torch.Tensor:
    """Run simple k-means on CPU/GPU depending on tensor device."""
    n, d = embeddings.shape
    if n < num_clusters:
        # Degenerate case: fewer samples than clusters.
        return torch.arange(n, device=embeddings.device) % num_clusters

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
    return labels


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
