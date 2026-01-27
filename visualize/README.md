# Visualization playground

Simple scenario: capture PGD-step embeddings and project with PCA.

1) Run the script (uses `configs/visualize_pgd_steps.yaml`):
```bash
python visualize/pgd_steps.py
```

2) Outputs:
- `outputs/visualize/visualize_pgd_steps_pgd_steps.npz`
- `outputs/visualize/visualize_pgd_steps_pgd_steps.json`
- `outputs/visualize/visualize_pgd_steps_pgd_steps_pca.png` (if matplotlib + sklearn installed)
- `outputs/visualize/visualize_pgd_steps_pgd_steps_plotly.html` (if plotly installed)
- `outputs/visualize/visualize_pgd_steps_pgd_steps_path.html` (if plotly installed)

3) W&B logging:
- `visualize.log_wandb: true` logs a W&B table named `pgd_steps`
- Each row includes `step`, `kind`, `label`, `image`, and `embedding`
- With `visualize.projection: "pca"`, extra columns `x`, `y` are added
- W&B panels `pgd_steps_scatter_step` and `pgd_steps_scatter_kind` are logged
- Use W&B filters to compare shifts by step/dataset/model

4) Tweak via config:
- `visualize.layer`: module name to hook (e.g. `layer4`, `fc`)
- `attack.train.num_steps`: number of PGD steps to trace
- `visualize.max_samples`: set to 1 to log a single image (11 rows for 10 steps + clean)
- `visualize.checkpoint`: path to a trained `.pt` to visualize
- `visualize.max_log_images`: limit per-step W&B image logging
- `visualize.projection`: `pca` or `none`

## TensorBoard projector for CIFAR-10

Log penultimate features from an ImageNet-pretrained ResNet18 and visualize them with TensorBoard's Projector. Points can be colored by CIFAR-10 label. By default this logs 5k test images to keep memory reasonable; set `--max_samples 0` to log the full split.

```bash
python visualize/tb_cifar10_projector.py \
  --run_dir runs/cifar10_projector \
  --data_root ./data \
  --split test \
  --max_samples 5000 \
  --img_size 64

tensorboard --logdir runs/cifar10_projector
# In the Projector tab: Color by -> label
```

Flags:
- `--split [train|test]`: choose dataset split.
- `--max_samples N`: number of examples to log (0 = all).
- `--img_size`: resize before embedding to trade detail vs. memory.
- `--batch_size`: dataloader batch size.
