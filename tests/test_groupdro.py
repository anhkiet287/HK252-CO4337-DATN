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


class MeanLogitModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = x.view(x.size(0), -1).mean(dim=1) + self.bias
        zeros = torch.zeros_like(pooled)
        return torch.stack([pooled, zeros], dim=1)


class LargeLossModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.size(0)
        true_class_bad = torch.full((batch,), -1_000_000.0, dtype=torch.float32, device=x.device)
        other_class = torch.zeros((batch,), dtype=torch.float32, device=x.device)
        return torch.stack([true_class_bad + self.bias, other_class], dim=1)


class TinyDataset(Dataset):
    def __len__(self) -> int:
        return 5

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        x = torch.tensor([[[float(idx)]]], dtype=torch.float32)
        y = int(idx % 2)
        return x, y


def _make_groupdro_cfg(eta_q: float = 1.0, update_mode: str = "online") -> dict:
    return {
        "dataset": {"name": "cifar10"},
        "train": {
            "mode": "groupdro",
            "groupdro": {
                "update_mode": update_mode,
                "eta_q": eta_q,
                "warmup_epochs": 0,
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
    offset = 0.0 if name == "clean" else 1.0

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


def test_groupdro_online_update_stays_finite_for_large_finite_losses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(groupdro_module, "build_attack", _fake_build_attack)

    objective = GroupDRO(_make_groupdro_cfg(eta_q=0.001), LargeLossModel())
    batch = {
        "x": torch.zeros(2, 1, 2, 2),
        "y": torch.zeros(2, dtype=torch.long),
        "group_id": torch.zeros(2, dtype=torch.long),
    }

    loss, metrics = objective.compute_loss(objective.model, batch)

    assert torch.isfinite(loss)
    assert torch.isfinite(objective.q).all()
    assert metrics["q_g"] == pytest.approx(1.0, rel=1e-6)
    assert metrics["q_min"] == pytest.approx(0.0, abs=1e-6)


def test_groupdro_warmup_epochs_freeze_q_updates(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(groupdro_module, "build_attack", _fake_build_attack)

    cfg = _make_groupdro_cfg(eta_q=1.0)
    cfg["train"]["groupdro"]["warmup_epochs"] = 3
    objective = GroupDRO(cfg, ZeroLogitModel())
    batch = {
        "x": torch.zeros(2, 1, 2, 2),
        "y": torch.tensor([0, 1], dtype=torch.long),
        "group_id": torch.zeros(2, dtype=torch.long),
    }

    initial_q = objective.q.clone()
    loss, metrics = objective.compute_loss(objective.model, batch)

    assert torch.isfinite(loss)
    assert torch.allclose(objective.q, initial_q, atol=1e-6)
    assert metrics["q_g"] == pytest.approx(0.5, rel=1e-6)


def test_groupdro_batch_mode_updates_observed_groups(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(groupdro_module, "build_attack", _fake_build_attack)

    objective = GroupDRO(_make_groupdro_cfg(eta_q=1.0, update_mode="batch"), MeanLogitModel())
    batch = {
        "x": torch.zeros(4, 1, 2, 2),
        "y": torch.zeros(4, dtype=torch.long),
        "group_id": torch.tensor([0, 0, 1, 1], dtype=torch.long),
    }

    loss, metrics = objective.compute_loss(objective.model, batch)

    loss_g0 = math.log(2.0)
    loss_g1 = float(torch.nn.functional.cross_entropy(torch.tensor([[1.0, 0.0]]), torch.tensor([0])).item())
    expected_unnormalized = torch.tensor(
        [0.5 * math.exp(loss_g0), 0.5 * math.exp(loss_g1)],
        dtype=torch.float32,
    )
    expected_q = expected_unnormalized / expected_unnormalized.sum()
    expected_loss = float(expected_q[0].item() * loss_g0 + expected_q[1].item() * loss_g1)

    assert torch.allclose(objective.q.cpu(), expected_q, atol=1e-6)
    assert torch.isclose(loss.detach().cpu(), torch.tensor(expected_loss), atol=1e-6)
    assert metrics["domain_name"] == "mixed"
    assert metrics["group_batch_counts"] == {"clean": 2, "shift": 2}
    assert metrics["q/clean"] == pytest.approx(float(expected_q[0].item()), rel=1e-6)
    assert metrics["q/shift"] == pytest.approx(float(expected_q[1].item()), rel=1e-6)


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


def test_groupdro_q_trajectory_figure_uses_group_labels(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("matplotlib")
    monkeypatch.setattr(groupdro_module, "build_attack", _fake_build_attack)

    objective = GroupDRO(_make_groupdro_cfg(), ZeroLogitModel())
    objective.q = torch.tensor([0.7, 0.3], dtype=torch.float32)
    objective.record_q_snapshot(1)
    objective.q = torch.tensor([0.6, 0.4], dtype=torch.float32)
    objective.record_q_snapshot(2)

    fig = objective.build_q_trajectory_figure(max_points=10)
    assert fig is not None
    labels = [line.get_label() for line in fig.axes[0].lines]
    assert labels == ["g=0: clean", "g=1: shift"]
