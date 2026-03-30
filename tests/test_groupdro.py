from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")
from torch.utils.data import DataLoader, Dataset

from ardg.data.grouping import GroupHomogeneousBatchSampler, RepeatedGroupDataset
import ardg.training.objectives.groupdro as groupdro_module
from ardg.training.objectives.groupdro import GroupDRO


class ZeroLogitModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        del x
        logits = torch.zeros((2, 2), dtype=torch.float32, device=self.bias.device)
        return logits + self.bias


class TinyDataset(Dataset):
    def __len__(self) -> int:
        return 5

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        x = torch.tensor([[[float(idx)]]], dtype=torch.float32)
        y = int(idx % 2)
        return x, y


def _make_groupdro_cfg(eta_q: float = 1.0) -> dict:
    return {
        "dataset": {"name": "cifar10"},
        "train": {
            "mode": "groupdro",
            "groupdro": {
                "eta_q": eta_q,
                "selection_metric": "val/acc_worst",
            },
        },
        "attack": {
            "train_domains": [
                {"label": "clean", "type": "clean"},
                {"label": "shift", "type": "pgd", "eps": 8.0 / 255.0, "step_size": 2.0 / 255.0, "num_steps": 2, "loss": "ce"},
            ]
        },
    }


def _fake_build_attack(spec: dict, model: torch.nn.Module, dataset_name: str):  # noqa: ANN001
    del model, dataset_name
    name = str(spec.get("label", spec.get("name", spec.get("type", "clean"))))
    offset = 0.0 if name == "clean" else 0.25

    def attack(images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        del labels
        return images + offset

    return attack


def test_groupdro_native_step_updates_only_current_group(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(groupdro_module, "build_attack", _fake_build_attack)

    objective = GroupDRO(_make_groupdro_cfg(eta_q=1.0), ZeroLogitModel())
    batch = {
        "x": torch.zeros(2, 1, 2, 2),
        "y": torch.tensor([0, 1], dtype=torch.long),
        "group_id": torch.zeros(2, dtype=torch.long),
    }

    loss, metrics = objective.compute_loss(objective.model, batch)

    expected_group_loss = math.log(2.0)
    expected_q = torch.tensor([2.0 / 3.0, 1.0 / 3.0], dtype=torch.float32)

    assert torch.allclose(objective.q.cpu(), expected_q, atol=1e-6)
    assert torch.isclose(loss.detach().cpu(), torch.tensor((2.0 / 3.0) * expected_group_loss), atol=1e-6)
    assert metrics["loss_group"] == pytest.approx(expected_group_loss, rel=1e-6)
    assert metrics["group_id"] == 0
    assert metrics["q_g"] == pytest.approx(2.0 / 3.0, rel=1e-6)
    assert metrics["q_max"] == pytest.approx(2.0 / 3.0, rel=1e-6)
    assert metrics["q_min"] == pytest.approx(1.0 / 3.0, rel=1e-6)
    assert metrics["q/clean"] == pytest.approx(2.0 / 3.0, rel=1e-6)
    assert metrics["q/shift"] == pytest.approx(1.0 / 3.0, rel=1e-6)
    assert metrics["correct"] == 1
    assert metrics["batch_size"] == 2


def test_groupdro_native_mode_requires_group_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(groupdro_module, "build_attack", _fake_build_attack)

    objective = GroupDRO(_make_groupdro_cfg(), ZeroLogitModel())
    batch = {
        "x": torch.zeros(2, 1, 2, 2),
        "y": torch.tensor([0, 1], dtype=torch.long),
    }

    with pytest.raises(ValueError, match="group ids"):
        objective.compute_loss(objective.model, batch)


def test_groupdro_native_mode_rejects_mixed_group_batches(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(groupdro_module, "build_attack", _fake_build_attack)

    objective = GroupDRO(_make_groupdro_cfg(), ZeroLogitModel())
    batch = {
        "x": torch.zeros(2, 1, 2, 2),
        "y": torch.tensor([0, 1], dtype=torch.long),
        "group_id": torch.tensor([0, 1], dtype=torch.long),
    }

    with pytest.raises(ValueError, match="exactly one group"):
        objective.compute_loss(objective.model, batch)


def test_groupdro_q_roundtrip(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(groupdro_module, "build_attack", _fake_build_attack)

    objective = GroupDRO(_make_groupdro_cfg(), ZeroLogitModel())
    objective.q = torch.tensor([0.8, 0.2], dtype=torch.float32)

    restored = GroupDRO(_make_groupdro_cfg(), ZeroLogitModel())
    restored.load_state_dict(objective.state_dict())

    assert torch.allclose(restored.q, objective.q, atol=1e-6)


def test_group_homogeneous_batch_sampler_emits_single_group_batches() -> None:
    dataset = RepeatedGroupDataset(TinyDataset(), num_groups=3)
    sampler = GroupHomogeneousBatchSampler(
        base_size=5,
        num_groups=3,
        batch_size=2,
        shuffle=False,
        drop_last=False,
    )
    loader = DataLoader(dataset, batch_sampler=sampler)

    batches = list(loader)

    assert len(batches) == 9
    for batch in batches:
        assert "group_id" in batch
        assert int(torch.unique(batch["group_id"]).numel()) == 1
