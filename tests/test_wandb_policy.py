from __future__ import annotations

from pathlib import Path

import pytest

from ardg.experiments.common import load_runtime_config
from ardg.utils.logging import init_wandb
from ardg.utils.paths import get_run_dir
from ardg.utils.run_metadata import derive_run_name


def test_init_wandb_fails_when_disabled_but_required() -> None:
    cfg = {
        "logging": {
            "run_name": "resnet18_local_baseline_erm_seed42",
            "wandb": {
                "enabled": False,
            },
        }
    }

    with pytest.raises(ValueError, match="W&B logging is mandatory"):
        init_wandb(cfg, require_enabled=True)


def test_derive_run_name_uses_backbone_profile_and_experiment_name() -> None:
    cfg = {
        "experiment": {"name": "baseline_erm", "seed": 42, "runtime_profile": "local_gpu"},
        "runtime": {"profile": "local_gpu"},
        "model": {"name": "resnet18_cifar"},
        "train": {"mode": "erm"},
        "logging": {"run_name": ""},
    }

    run_name = derive_run_name(cfg, suffix="eval")

    assert run_name == "resnet18_local_baseline_erm_seed42_eval"


def test_load_runtime_config_derives_training_run_dir_for_eval_config() -> None:
    cfg = load_runtime_config(
        "configs/experiments/cifar10/resnet18/eval/baseline_erm_all_attacks.yaml",
        profile_path="configs/profiles/local_gpu.yaml",
    )

    assert cfg["logging"]["run_name"] == "resnet18_local_baseline_erm_seed42"
    assert Path(get_run_dir(cfg)).as_posix().endswith(
        "/outputs/thesis/local/resnet18_local_baseline_erm_seed42"
    )
