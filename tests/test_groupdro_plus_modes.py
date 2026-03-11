from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchattacks")

from ardg.training.objectives.groupdro_plus import GroupDROPlus


class TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(4, 3, bias=False)
        with torch.no_grad():
            self.proj.weight.copy_(
                torch.tensor(
                    [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 1.0],
                    ],
                    dtype=torch.float32,
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flat = x.view(x.size(0), -1)
        return self.proj(flat)


def _make_epoch_cfg(every_epochs: int = 1) -> dict:
    return {
        "experiment": {"seed": 42},
        "dataset": {"name": "cifar10"},
        "logging": {
            "output_dir": "outputs/test",
            "run_name": "groupdro_plus_modes_test",
            "wandb": {"enabled": False},
        },
        "train": {
            "mode": "groupdro_plus",
            "adv_training": False,
            "groupdro_plus": {
                "cluster_mode": "epoch",
                "num_clusters": 2,
                "eta": 0.02,
                "lambda_reg": 0.5,
                "gamma": 1.0,
                "epoch_cluster": {
                    "every_epochs": every_epochs,
                    "source": "clean",
                    "max_batches": 1,
                },
            },
        },
    }


def _batch(values: list[list[float]], labels: list[int]) -> dict:
    x = torch.tensor(values, dtype=torch.float32).view(len(values), 1, 2, 2)
    y = torch.tensor(labels, dtype=torch.long)
    return {"x": x, "y": y}


def test_epoch_mode_initializes_centers_on_train_start() -> None:
    cfg = _make_epoch_cfg(every_epochs=2)
    objective = GroupDROPlus(cfg, TinyModel())
    loaders = {
        "train": [
            _batch(
                [
                    [0.0, 0.0, 0.0, 0.1],
                    [0.1, 0.0, 0.0, 0.0],
                    [3.0, 3.0, 3.0, 3.1],
                    [3.1, 3.0, 3.0, 3.0],
                ],
                [0, 0, 1, 1],
            )
        ]
    }

    objective.on_train_start(loaders)

    assert objective.cluster_centers is not None
    assert objective.cluster_counts is not None
    assert objective.cluster_centers.shape == (2, 3)
    assert int(objective.cluster_counts.sum().item()) == 4
    assert objective.cluster_centers_epoch == 0


def test_epoch_mode_reclusters_only_on_configured_epochs() -> None:
    cfg = _make_epoch_cfg(every_epochs=2)
    objective = GroupDROPlus(cfg, TinyModel())

    loaders_a = {
        "train": [
            _batch(
                [
                    [0.0, 0.0, 0.0, 0.1],
                    [0.1, 0.0, 0.0, 0.0],
                    [3.0, 3.0, 3.0, 3.1],
                    [3.1, 3.0, 3.0, 3.0],
                ],
                [0, 0, 1, 1],
            )
        ]
    }
    loaders_b = {
        "train": [
            _batch(
                [
                    [10.0, 10.0, 10.0, 10.0],
                    [10.2, 10.1, 10.0, 10.0],
                    [20.0, 20.0, 20.0, 20.0],
                    [20.2, 20.1, 20.0, 20.0],
                ],
                [0, 0, 1, 1],
            )
        ]
    }

    objective.on_train_start(loaders_a)
    assert objective.cluster_centers is not None
    centers_start = objective.cluster_centers.clone()

    objective.on_epoch_end(1, loaders_b)
    assert objective.cluster_centers is not None
    assert torch.allclose(objective.cluster_centers, centers_start)

    objective.on_epoch_end(2, loaders_b)
    assert objective.cluster_centers is not None
    assert not torch.allclose(objective.cluster_centers, centers_start)
    assert objective.cluster_centers_epoch == 2

    loss, metrics = objective.compute_loss(objective.model, loaders_b["train"][0])
    assert torch.isfinite(loss)
    assert metrics["cluster_mode_epoch"] == 1.0
    assert "loss_g0" in metrics
    assert "count_g0" in metrics
