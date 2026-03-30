# Experiment Matrix

| Backbone | Category | Config | Typical Profile | Purpose |
| --- | --- | --- | --- | --- |
| ResNet-18 | Baseline | `configs/experiments/cifar10/resnet18/baselines/erm.yaml` | `local_gpu` | Default clean baseline |
| ResNet-18 | Baseline | `configs/experiments/cifar10/resnet18/baselines/pgd_at_linf.yaml` | `local_gpu` | Single-attack AT baseline with explicit Linf train norm |
| ResNet-18 | Baseline | `configs/experiments/cifar10/resnet18/baselines/pgd_at_l2.yaml` | `local_gpu` | Single-attack AT baseline with explicit L2 train norm |
| ResNet-18 | Baseline | `configs/experiments/cifar10/resnet18/baselines/uniform_multi_attack.yaml` | `local_gpu` | Uniform multi-attack baseline |
| ResNet-18 | Ablation | `configs/experiments/cifar10/resnet18/ablations/groupdro.yaml` | `local_gpu` | Domain-weighting ablation |
| ResNet-18 | Ablation | `configs/experiments/cifar10/resnet18/ablations/multi_attack_erm.yaml` | `local_gpu` | Attack-domain baseline ablation |
| ResNet-18 | Eval | `configs/experiments/cifar10/resnet18/eval/baseline_erm_all_attacks.yaml` | `local_gpu` | Full attack suite evaluation |
| ResNet-50 | Baseline | `configs/experiments/cifar10/resnet50/baselines/erm.yaml` | `h100` or `local_gpu` | Scale-up baseline |
| ResNet-50 | Eval | `configs/experiments/cifar10/resnet50/eval/baseline_erm_all_attacks.yaml` | `h100` or `local_gpu` | Scale-up evaluation |
| ViT-B/16 | Baseline | `configs/experiments/cifar10/vit_b16/baselines/erm.yaml` | `h100` | Optional extension baseline |
| ViT-B/16 | Eval | `configs/experiments/cifar10/vit_b16/eval/baseline_erm_all_attacks.yaml` | `h100` | Optional extension evaluation |

## Smoke Coverage

- ResNet-18 smoke train: baseline ERM with `configs/profiles/dev_fast.yaml`
- ResNet-18 smoke eval: `baseline_erm_all_attacks.yaml` with `configs/profiles/dev_fast.yaml`
- ResNet-50 smoke train: baseline ERM with `configs/profiles/dev_fast.yaml`
- ViT-B/16 smoke train: baseline ERM with `configs/profiles/dev_fast.yaml`
