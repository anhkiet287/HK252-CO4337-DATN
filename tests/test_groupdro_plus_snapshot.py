from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchattacks")

from ardg.training.objectives.groupdro_plus import GroupDROPlus


class TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(4, 3, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flat = x.view(x.size(0), -1)
        return self.proj(flat)


def _make_cfg(tmp_path) -> dict:
    return {
        "experiment": {"seed": 42},
        "dataset": {"name": "cifar10"},
        "logging": {
            "output_dir": str(tmp_path),
            "run_name": "groupdro_plus_snapshot_test",
            "wandb": {"enabled": False},
        },
        "train": {
            "mode": "groupdro_plus",
            "adv_training": False,
            "groupdro_plus": {
                "num_clusters": 2,
                "eta": 0.02,
                "lambda_reg": 0.5,
                "gamma": 1.0,
                "snapshot": {
                    "enabled": True,
                    "every_epochs": 2,
                    "max_batches": 1,
                    "splits": ["train"],
                    "save_pca": True,
                },
            },
        },
    }


def test_groupdro_plus_snapshot_saved_every_k_epochs(tmp_path) -> None:
    cfg = _make_cfg(tmp_path)
    model = TinyModel()
    objective = GroupDROPlus(cfg, model)

    batch = {
        "x": torch.tensor(
            [
                [[0.1, 0.2], [0.3, 0.4]],
                [[0.2, 0.1], [0.4, 0.3]],
                [[1.0, 1.1], [0.9, 1.2]],
                [[1.2, 1.0], [1.1, 0.8]],
            ],
            dtype=torch.float32,
        ).unsqueeze(1),
        "y": torch.tensor([0, 0, 1, 1], dtype=torch.long),
    }
    loaders = {"train": [batch]}

    objective.on_epoch_end(1, loaders)
    epoch1_path = tmp_path / "groupdro_plus_snapshot_test" / "cluster_snapshots" / "train" / "epoch_0001_batch_0001.npz"
    assert not epoch1_path.exists()

    objective.on_epoch_end(2, loaders)
    epoch2_path = tmp_path / "groupdro_plus_snapshot_test" / "cluster_snapshots" / "train" / "epoch_0002_batch_0001.npz"
    assert epoch2_path.exists()

    data = np.load(epoch2_path)
    assert data["embeddings"].shape == (4, 3)
    assert data["cluster_ids"].shape == (4,)
    assert data["labels"].tolist() == [0, 0, 1, 1]
    assert data["projection_2d"].shape == (4, 2)
    assert int(data["epoch"][0]) == 2
    assert int(data["batch_idx"][0]) == 1
