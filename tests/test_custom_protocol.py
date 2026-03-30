from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
from torch.utils.data import DataLoader, Dataset

from ardg.data.checkpoint_cache import CheckpointCacheDataset
from ardg.data.grouping import RepeatedGroupDataset
import ardg.data.datasets as datasets_module
from ardg.training.objectives import build_objective
import ardg.training.objectives.custom_protocol as custom_protocol_module
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


def _checkpoint_base_cfg() -> dict:
    return {
        "dataset": {"name": "cifar10"},
        "experiment": {"seed": 7},
        "train": {
            "mode": "custom_protocol",
            "batch_size": 2,
        },
        "custom_protocol": {
            "name": "checkpoint_base",
            "warmup_epochs": 0,
            "period_epochs": 2,
            "refresh_subset_size": 0.5,
            "score_metric": "ce_loss",
            "update_rule": {"temperature": 1.0, "alpha": 0.5, "beta": 0.1},
            "allocation": {"mode": "quota", "one_attack_per_sample": True},
            "cache": {"save_to_disk": False, "directory": "artifacts/checkpoint_base_cache"},
        },
        "attack": {
            "train_domains": [
                {"label": "clean", "type": "clean"},
                {"label": "hard", "type": "pgd", "eps": 8.0 / 255.0, "step_size": 2.0 / 255.0, "num_steps": 2},
            ]
        },
    }


def _checkpoint_base_v2_cfg(update_mode: str = "online") -> dict:
    return {
        "dataset": {"name": "cifar10"},
        "experiment": {"seed": 7},
        "train": {
            "mode": "custom_protocol",
            "batch_size": 2,
        },
        "custom_protocol": {
            "name": "checkpoint_base",
            "version": "v2",
            "warmup_enabled": True,
            "warmup_epochs": 2,
            "period_epochs": 2,
            "update_mode": update_mode,
            "refresh_subset_size": 0.5,
            "score_metric": "ce_loss",
            "update_rule": {"temperature": 1.0, "alpha": 0.5, "beta": 0.1},
        },
        "attack": {
            "train_domains": [
                {"label": "clean", "type": "clean"},
                {"label": "hard", "type": "pgd", "eps": 8.0 / 255.0, "step_size": 2.0 / 255.0, "num_steps": 2},
            ]
        },
    }


def _batch() -> dict:
    return {
        "x": torch.randn(6, 1, 2, 2),
        "y": torch.tensor([0, 1, 2, 1, 0, 2], dtype=torch.long),
    }


class TinyTrainDataset(Dataset):
    def __len__(self) -> int:
        return 6

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        del idx
        return torch.zeros(1, 2, 2), 0


class TwoClassMeanModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = x.view(x.size(0), -1).mean(dim=1) + self.bias
        zeros = torch.zeros_like(pooled)
        return torch.stack([pooled, zeros], dim=1)


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


def _fake_checkpoint_suite(call_counts: dict[str, int]):
    def clean(images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        del labels
        call_counts["clean"] += 1
        return images

    def hard(images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        del labels
        call_counts["hard"] += 1
        return images - 1.0

    return {"clean": clean, "hard": hard}


def test_checkpoint_base_updates_q_only_at_period_boundaries(monkeypatch: pytest.MonkeyPatch) -> None:
    call_counts = {"clean": 0, "hard": 0}
    monkeypatch.setattr(
        custom_protocol_module,
        "build_train_suite",
        lambda cfg, model: _fake_checkpoint_suite(call_counts),
    )

    wrapped = CheckpointCacheDataset(TinyTrainDataset())
    loader = DataLoader(wrapped, batch_size=2, shuffle=False)
    objective = CustomProtocol(_checkpoint_base_cfg(), TwoClassMeanModel())
    objective.on_train_start({"train": loader, "val": loader})

    assert wrapped.has_active_cache()
    assert objective.consume_train_time_seconds() >= 0.0
    q_before = objective.q.clone()
    calls_before_loss = dict(call_counts)

    loss, metrics = objective.compute_loss(objective.model, next(iter(loader)))

    assert torch.isfinite(loss)
    assert dict(call_counts) == calls_before_loss
    assert torch.allclose(objective.q, q_before)
    assert "loss_adv" in metrics
    assert "acc_adv" in metrics

    objective.on_epoch_end(1, {"train": loader, "val": loader})
    assert torch.allclose(objective.q, q_before)

    objective.on_epoch_end(2, {"train": loader, "val": loader})
    assert objective.current_period_idx == 1
    assert not torch.allclose(objective.q, q_before)
    assert objective.latest_cache_generate_time_sec >= 0.0
    assert objective.latest_refresh_score_time_sec >= 0.0


def test_checkpoint_base_online_warmup_runs_before_first_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    call_counts = {"clean": 0, "hard": 0}
    monkeypatch.setattr(
        custom_protocol_module,
        "build_train_suite",
        lambda cfg, model: _fake_checkpoint_suite(call_counts),
    )

    wrapped = CheckpointCacheDataset(TinyTrainDataset())
    loader = DataLoader(wrapped, batch_size=2, shuffle=False)
    cfg = _checkpoint_base_cfg()
    cfg["custom_protocol"]["warmup_epochs"] = 2
    objective = CustomProtocol(cfg, TwoClassMeanModel())
    monkeypatch.setattr(
        objective,
        "_sample_online_attack_ids",
        lambda batch_size, device: torch.ones(batch_size, dtype=torch.long, device=device),
    )

    objective.on_train_start({"train": loader, "val": loader})

    assert not wrapped.has_active_cache()
    loss, metrics = objective.compute_loss(objective.model, next(iter(loader)))

    assert torch.isfinite(loss)
    assert metrics["warmup_online"] == pytest.approx(1.0)
    assert call_counts["hard"] > 0

    objective.on_epoch_end(1, {"train": loader, "val": loader})
    assert not wrapped.has_active_cache()

    objective.on_epoch_end(2, {"train": loader, "val": loader})
    assert wrapped.has_active_cache()
    assert objective.current_period_idx == 0


def test_checkpoint_base_state_roundtrip() -> None:
    objective = CustomProtocol(_checkpoint_base_cfg(), TwoClassMeanModel())
    objective.q = torch.tensor([0.8, 0.2], dtype=torch.float32)
    objective.current_period_idx = 3
    objective.current_period_start_epoch = 7
    objective.current_period_end_epoch = 8
    objective.refresh_subset_indices = [1, 4, 5]
    objective.refresh_subset_count = 3
    objective.current_cache_manifest = {"path": "/tmp/cache.pt", "period_idx": 3}
    objective.latest_scores = {"clean": 0.5, "hard": 1.5}
    objective.latest_cache_counts = {"clean": 2, "hard": 4}

    restored = CustomProtocol(_checkpoint_base_cfg(), TwoClassMeanModel())
    restored.load_state_dict(objective.state_dict())

    assert torch.allclose(restored.q, objective.q)
    assert restored.current_period_idx == 3
    assert restored.current_period_start_epoch == 7
    assert restored.current_period_end_epoch == 8
    assert restored.refresh_subset_indices == [1, 4, 5]
    assert restored.current_cache_manifest["path"] == "/tmp/cache.pt"
    assert restored.latest_scores["hard"] == pytest.approx(1.5)


def test_checkpoint_base_version_defaults_to_v1() -> None:
    objective = CustomProtocol(_checkpoint_base_cfg(), TwoClassMeanModel())
    assert objective.protocol_version == "v1"


def test_checkpoint_base_v2_version_resolution() -> None:
    objective = CustomProtocol(_checkpoint_base_v2_cfg(), TwoClassMeanModel())
    assert objective.protocol_version == "v2"
    assert objective.update_mode == "online"


def test_checkpoint_base_v2_online_warmup_keeps_uniform_q(monkeypatch: pytest.MonkeyPatch) -> None:
    call_counts = {"clean": 0, "hard": 0}
    monkeypatch.setattr(
        custom_protocol_module,
        "build_train_suite",
        lambda cfg, model: _fake_checkpoint_suite(call_counts),
    )

    base_dataset = TinyTrainDataset()
    loader = DataLoader(
        RepeatedGroupDataset(base_dataset, num_groups=2),
        batch_sampler=[[0, 1]],
    )
    objective = CustomProtocol(_checkpoint_base_v2_cfg(update_mode="online"), TwoClassMeanModel())
    objective.on_train_start({"train": loader, "val": loader})
    initial_q = objective.q.clone()

    loss, metrics = objective.compute_loss(objective.model, next(iter(loader)))

    assert torch.isfinite(loss)
    assert torch.allclose(objective.q, initial_q)
    assert metrics["warmup_active"] == pytest.approx(1.0)
    assert metrics["domain_name"] == "warmup_online"
    assert metrics["q/clean"] == pytest.approx(0.5)
    assert metrics["q/hard"] == pytest.approx(0.5)


def test_checkpoint_base_v2_batch_mode_keeps_q_fixed_within_period(monkeypatch: pytest.MonkeyPatch) -> None:
    call_counts = {"clean": 0, "hard": 0}
    monkeypatch.setattr(
        custom_protocol_module,
        "build_train_suite",
        lambda cfg, model: _fake_checkpoint_suite(call_counts),
    )

    objective = CustomProtocol(_checkpoint_base_v2_cfg(update_mode="batch"), TwoClassMeanModel())
    objective.current_epoch = 3
    batch = {
        "x": torch.zeros(4, 1, 2, 2),
        "y": torch.zeros(4, dtype=torch.long),
        "group_id": torch.tensor([0, 0, 1, 1], dtype=torch.long),
    }
    q_before = objective.q.clone()

    loss, metrics = objective.compute_loss(objective.model, batch)

    assert torch.isfinite(loss)
    assert torch.allclose(objective.q, q_before)
    assert metrics["domain_name"] == "mixed"
    assert metrics["domain_batch_counts"] == {"clean": 2, "hard": 2}


def test_checkpoint_base_v2_updates_q_only_at_boundaries(monkeypatch: pytest.MonkeyPatch) -> None:
    call_counts = {"clean": 0, "hard": 0}
    monkeypatch.setattr(
        custom_protocol_module,
        "build_train_suite",
        lambda cfg, model: _fake_checkpoint_suite(call_counts),
    )

    base_dataset = TinyTrainDataset()
    loader = DataLoader(
        RepeatedGroupDataset(base_dataset, num_groups=2),
        batch_sampler=[[0, 1]],
    )
    cfg = _checkpoint_base_v2_cfg(update_mode="online")
    cfg["custom_protocol"]["warmup_enabled"] = False
    cfg["custom_protocol"]["warmup_epochs"] = 0
    objective = CustomProtocol(cfg, TwoClassMeanModel())
    objective.on_train_start({"train": loader, "val": loader})
    q_before = objective.q.clone()

    objective.compute_loss(objective.model, next(iter(loader)))
    assert torch.allclose(objective.q, q_before)

    objective.on_epoch_end(1, {"train": loader, "val": loader})
    assert torch.allclose(objective.q, q_before)

    objective.on_epoch_end(2, {"train": loader, "val": loader})
    assert not torch.allclose(objective.q, q_before)
    assert objective.current_period_idx == 1


def test_checkpoint_base_v2_refresh_subset_is_deterministic(monkeypatch: pytest.MonkeyPatch) -> None:
    call_counts = {"clean": 0, "hard": 0}
    monkeypatch.setattr(
        custom_protocol_module,
        "build_train_suite",
        lambda cfg, model: _fake_checkpoint_suite(call_counts),
    )

    dataset = RepeatedGroupDataset(TinyTrainDataset(), num_groups=2)
    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    first = CustomProtocol(_checkpoint_base_v2_cfg(), TwoClassMeanModel())
    second = CustomProtocol(_checkpoint_base_v2_cfg(), TwoClassMeanModel())

    first.on_train_start({"train": loader, "val": loader})
    second.on_train_start({"train": loader, "val": loader})

    assert first.refresh_subset_indices == second.refresh_subset_indices


def test_checkpoint_base_v2_state_roundtrip() -> None:
    objective = CustomProtocol(_checkpoint_base_v2_cfg(), TwoClassMeanModel())
    objective.q = torch.tensor([0.3, 0.7], dtype=torch.float32)
    objective.current_period_idx = 2
    objective.current_period_start_epoch = 6
    objective.current_period_end_epoch = 7
    objective.refresh_subset_indices = [0, 3, 5]
    objective.refresh_subset_count = 3
    objective.latest_scores = {"clean": 0.2, "hard": 1.2}

    restored = CustomProtocol(_checkpoint_base_v2_cfg(), TwoClassMeanModel())
    restored.load_state_dict(objective.state_dict())

    assert torch.allclose(restored.q, objective.q)
    assert restored.current_period_idx == 2
    assert restored.refresh_subset_indices == [0, 3, 5]
    assert restored.latest_scores["hard"] == pytest.approx(1.2)


def test_checkpoint_base_v2_loaders_do_not_require_cache_dataset(monkeypatch: pytest.MonkeyPatch) -> None:
    class TinyVisionDataset(Dataset):
        def __len__(self) -> int:
            return 6

        def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
            x = torch.full((1, 2, 2), float(idx), dtype=torch.float32)
            y = int(idx % 2)
            return x, y

    monkeypatch.setattr(datasets_module, "ensure_split", lambda cfg: "unused")
    monkeypatch.setattr(
        datasets_module,
        "load_splits",
        lambda path: {"splits": {"train": [0, 1, 2, 3], "val": [4], "test": [5]}},
    )
    monkeypatch.setattr(datasets_module, "build_transforms", lambda cfg, split: None)
    monkeypatch.setattr(datasets_module, "load_dataset", lambda name, root, train, transform: TinyVisionDataset())

    online_cfg = _checkpoint_base_v2_cfg(update_mode="online")
    online_cfg["dataset"].update({"num_workers": 0, "max_test_samples": 1})
    train_loader, _, _ = datasets_module.get_dataloaders(online_cfg)
    online_batch = next(iter(train_loader))

    assert isinstance(train_loader.dataset, RepeatedGroupDataset)
    assert not isinstance(train_loader.dataset.base, CheckpointCacheDataset)
    assert int(torch.unique(online_batch["group_id"]).numel()) == 1

    batch_cfg = _checkpoint_base_v2_cfg(update_mode="batch")
    batch_cfg["dataset"].update({"num_workers": 0, "max_test_samples": 1})
    train_loader_batch, _, _ = datasets_module.get_dataloaders(batch_cfg)
    batch = next(iter(train_loader_batch))

    assert isinstance(train_loader_batch.dataset, RepeatedGroupDataset)
    assert not isinstance(train_loader_batch.dataset.base, CheckpointCacheDataset)
    assert "group_id" in batch
