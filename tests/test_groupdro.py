from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchattacks")

import ardg.training.objectives.multi_attack_erm as multi_attack_module
from ardg.training.objectives.groupdro import GroupDRO
from ardg.training.objectives.groupdro_state import (
    DomainWeightState,
    update_q_from_losses,
    weighted_group_loss,
)


class TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flat = x.view(x.size(0), -1).mean(dim=1, keepdim=True) * self.scale
        return torch.cat([flat, -flat], dim=1)


class DomainAwareModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        offset = float(x.mean().item())
        if offset < 0.05:
            logits = torch.tensor([[0.0, 0.1], [0.1, 0.0]], dtype=torch.float32, device=x.device)
        elif offset < 0.15:
            logits = torch.tensor([[3.0, 0.0], [0.0, 3.0]], dtype=torch.float32, device=x.device)
        elif offset < 0.25:
            logits = torch.tensor([[3.0, 0.0], [3.0, 0.0]], dtype=torch.float32, device=x.device)
        else:
            logits = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32, device=x.device)
        return logits + self.bias


def _make_groupdro_cfg() -> dict:
    return {
        "experiment": {"seed": 42},
        "dataset": {"name": "cifar10"},
        "train": {
            "mode": "groupdro",
            "domain_strategy": "all_domains",
            "groupdro": {
                "eta_q": 0.0,
                "include_clean": True,
                "init_q": "uniform",
                "selection_metric": "val/acc_worst",
            },
        },
        "attack": {
            "train_domains": [
                {"label": "clean", "type": "clean"},
                {"label": "fgsm_rs", "type": "fgsm", "eps": 8.0 / 255.0, "alpha": 8.0 / 255.0, "random_start": True},
                {"label": "pgd_ce", "type": "pgd", "eps": 8.0 / 255.0, "step_size": 2.0 / 255.0, "num_steps": 2, "loss": "ce"},
                {"label": "pgd_dlr", "type": "pgd", "eps": 8.0 / 255.0, "step_size": 2.0 / 255.0, "num_steps": 2, "loss": "dlr"},
            ]
        },
    }


def _fake_build_attack(spec: dict, model: torch.nn.Module, dataset_name: str, shared_norm=None, shared_eps=None):  # noqa: ANN001
    del model, dataset_name, shared_norm, shared_eps
    offsets = {
        "clean": 0.0,
        "fgsm_rs": 0.1,
        "pgd_ce": 0.2,
        "pgd_dlr": 0.3,
        "cw_l2": 0.4,
    }
    name = str(spec.get("name", spec.get("label", spec.get("type", "clean"))))
    offset = float(offsets[name])

    def attack(images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        del labels
        return images + offset

    return attack


def test_q_update_correctness() -> None:
    losses = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    q0 = torch.ones(3, dtype=torch.float32) / 3.0

    q1 = update_q_from_losses(q0, losses, eta_q=0.5)

    assert torch.isclose(q1.sum(), torch.tensor(1.0), atol=1e-6)
    assert q1[2] > q1[1] > q1[0]
    assert torch.all(q1 >= 0)


def test_weighted_loss_correctness() -> None:
    q = torch.tensor([0.2, 0.3, 0.5], dtype=torch.float32)
    losses = torch.tensor([1.0, 2.0, 4.0], dtype=torch.float32)

    total = weighted_group_loss(q, losses)

    assert torch.isclose(total, torch.tensor(2.8), atol=1e-6)


def test_eta_zero_matches_mean_loss() -> None:
    losses = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
    state = DomainWeightState(["a", "b", "c", "d"], eta_q=0.0, init_q="uniform")

    q = state.update(losses)
    total = weighted_group_loss(q, losses)

    assert torch.allclose(q, torch.ones_like(q) / 4.0, atol=1e-6)
    assert torch.isclose(total, losses.mean(), atol=1e-6)


def test_state_dict_roundtrip() -> None:
    state = DomainWeightState(["clean", "fgsm_rs", "pgd_ce", "pgd_dlr"], eta_q=0.1, init_q="uniform")
    state.set_q(torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float32))

    payload = state.state_dict()
    restored = DomainWeightState(["clean", "fgsm_rs", "pgd_ce", "pgd_dlr"], eta_q=0.1, init_q="uniform")
    restored.load_state_dict(payload)

    assert restored.q is not None
    assert torch.allclose(restored.q, state.q, atol=1e-6)


def test_all_domains_coverage(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(multi_attack_module, "build_attack", _fake_build_attack)

    model = TinyModel()
    objective = GroupDRO(_make_groupdro_cfg(), model)
    batch = {
        "x": torch.randn(4, 3, 4, 4),
        "y": torch.tensor([0, 1, 0, 1], dtype=torch.long),
    }

    out = objective.preprocess_batch(batch, model)

    assert out["domain_names"] == ["clean", "fgsm_rs", "pgd_ce", "pgd_dlr"]
    assert len(out["x_domains"]) == 4
    assert all(t.shape == batch["x"].shape for t in out["x_domains"])
    assert all(count == batch["x"].size(0) for count in out["domain_batch_counts"].values())
    assert torch.allclose(out["x_domains"][0], batch["x"])


def test_objective_state_dict_roundtrip(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(multi_attack_module, "build_attack", _fake_build_attack)

    model = TinyModel()
    objective = GroupDRO(_make_groupdro_cfg(), model)
    objective.q_state.set_q(torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float32))
    objective.normalized_loss_state.set_ema_losses(torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32))

    clone = GroupDRO(_make_groupdro_cfg(), TinyModel())
    clone.load_state_dict(objective.state_dict())

    assert clone.q is not None
    assert objective.q is not None
    assert torch.allclose(clone.q, objective.q, atol=1e-6)
    assert clone.normalized_loss_state.ema_losses is not None
    assert objective.normalized_loss_state.ema_losses is not None
    assert torch.allclose(clone.normalized_loss_state.ema_losses, objective.normalized_loss_state.ema_losses, atol=1e-6)


def test_groupdro_tracks_worst_group_by_accuracy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(multi_attack_module, "build_attack", _fake_build_attack)

    model = DomainAwareModel()
    objective = GroupDRO(_make_groupdro_cfg(), model)
    batch = {
        "x": torch.zeros(2, 3, 4, 4),
        "y": torch.tensor([0, 1], dtype=torch.long),
    }

    processed = objective.preprocess_batch(batch, model)
    _, metrics = objective.compute_loss(model, processed)

    assert metrics["worst_group"] == "clean"
    assert metrics["worst_group_mode"] == "acc"
    assert metrics["worst_group_by_acc"] == "clean"
    assert metrics["worst_group_by_loss"] == "pgd_ce"
    assert metrics["loss_pgd_ce"] > metrics["loss_clean"]


def test_groupdro_accepts_cw_train_domain(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(multi_attack_module, "build_attack", _fake_build_attack)

    cfg = _make_groupdro_cfg()
    cfg["attack"]["train_domains"] = [
        {"label": "clean", "type": "clean"},
        {"label": "pgd_ce", "type": "pgd", "eps": 8.0 / 255.0, "step_size": 2.0 / 255.0, "num_steps": 2, "loss": "ce"},
        {"label": "cw_l2", "type": "cw", "steps": 5, "lr": 0.01},
    ]

    model = TinyModel()
    objective = GroupDRO(cfg, model)
    batch = {
        "x": torch.randn(4, 3, 4, 4),
        "y": torch.tensor([0, 1, 0, 1], dtype=torch.long),
    }

    out = objective.preprocess_batch(batch, model)

    assert out["domain_names"] == ["clean", "pgd_ce", "cw_l2"]
    assert len(out["x_domains"]) == 3


def test_groupdro_tracks_worst_group_by_loss_when_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(multi_attack_module, "build_attack", _fake_build_attack)

    cfg = _make_groupdro_cfg()
    cfg["train"]["groupdro"]["worst_group_by"] = "loss"

    model = DomainAwareModel()
    objective = GroupDRO(cfg, model)
    batch = {
        "x": torch.zeros(2, 3, 4, 4),
        "y": torch.tensor([0, 1], dtype=torch.long),
    }

    processed = objective.preprocess_batch(batch, model)
    _, metrics = objective.compute_loss(model, processed)

    assert metrics["worst_group"] == "pgd_ce"
    assert metrics["worst_group_mode"] == "loss"
    assert metrics["worst_group_by_acc"] == "clean"
    assert metrics["worst_group_by_loss"] == "pgd_ce"


def test_groupdro_tracks_worst_group_by_normalized_loss_when_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(multi_attack_module, "build_attack", _fake_build_attack)

    cfg = _make_groupdro_cfg()
    cfg["train"]["groupdro"]["worst_group_by"] = "normalized_loss"

    model = DomainAwareModel()
    objective = GroupDRO(cfg, model)
    objective.normalized_loss_state.set_ema_losses(torch.tensor([0.1, 10.0, 10.0, 10.0], dtype=torch.float32))
    batch = {
        "x": torch.zeros(2, 3, 4, 4),
        "y": torch.tensor([0, 1], dtype=torch.long),
    }

    processed = objective.preprocess_batch(batch, model)
    _, metrics = objective.compute_loss(model, processed)

    assert metrics["worst_group"] == "clean"
    assert metrics["worst_group_mode"] == "normalized_loss"
    assert metrics["worst_group_by_acc"] == "clean"
    assert metrics["worst_group_by_loss"] == "pgd_ce"
    assert metrics["worst_group_by_normalized_loss"] == "clean"
    assert metrics["normalized_loss_clean"] > metrics["normalized_loss_pgd_ce"]


def test_groupdro_rejects_invalid_worst_group_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(multi_attack_module, "build_attack", _fake_build_attack)

    cfg = _make_groupdro_cfg()
    cfg["train"]["groupdro"]["worst_group_by"] = "foo"

    with pytest.raises(ValueError, match="worst_group_by"):
        GroupDRO(cfg, TinyModel())
