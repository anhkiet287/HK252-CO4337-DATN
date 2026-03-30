# Artifact Registry

`artifacts/latest/manifest.yaml` is the repo-local source of truth for canonical config paths and the most recent important outputs produced by this workspace.

Use it for:

- canonical experiment and profile paths
- latest ResNet-18 / ResNet-50 / ViT train and eval outputs
- latest best and last checkpoints
- latest eval summary JSON
- latest export and report outputs
- latest W&B run id and URL when available

Notes:

- Runtime outputs still live under `outputs/`.
- The registry only records pointers to those outputs.
- `artifacts/latest/paths.md` is the human-readable companion view.
