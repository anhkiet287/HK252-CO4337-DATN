from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from ardg.training.objectives import build_objective
from ardg.training.objectives.custom_protocol import CustomProtocol


class TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(4, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x.view(x.size(0), -1))


def _cfg() -> dict:
    return {
        "dataset": {"name": "cifar10"},
        "train": {
            "mode": "custom_protocol",
            "adv_training": False,
            "custom_protocol": {
                "use_adv_inputs": False,
                "weight_main": 1.0,
                "weight_aux": 0.0,
            },
        },
    }


def _batch() -> dict:
    return {
        "x": torch.randn(6, 1, 2, 2),
        "y": torch.tensor([0, 1, 2, 1, 0, 2], dtype=torch.long),
    }


def test_registry_builds_custom_protocol() -> None:
    objective = build_objective(_cfg(), TinyModel())
    assert isinstance(objective, CustomProtocol)


def test_custom_protocol_compute_loss_is_runnable() -> None:
    objective = CustomProtocol(_cfg(), TinyModel())
    loss, metrics = objective.compute_loss(objective.model, _batch())

    assert torch.isfinite(loss)
    assert metrics["batch_size"] == 6
    assert "loss_base" in metrics
    assert "loss_aux" in metrics
    assert "loss_clean" in metrics
    assert "acc_clean" in metrics


def test_custom_protocol_state_roundtrip() -> None:
    objective = CustomProtocol(_cfg(), TinyModel())
    objective.protocol_step = 7
    restored = CustomProtocol(_cfg(), TinyModel())

    restored.load_state_dict(objective.state_dict())

    assert restored.protocol_step == 7
