from __future__ import annotations

from pathlib import Path

from ardg.config import load_config
from ardg.experiments.common import load_runtime_config


def test_load_config_resolves_base_chain_and_overlay(tmp_path: Path) -> None:
    base_path = tmp_path / "base.yaml"
    child_path = tmp_path / "child.yaml"
    overlay_path = tmp_path / "overlay.yaml"

    base_path.write_text(
        "\n".join(
            [
                "experiment:",
                "  seed: 1",
                "dataset:",
                "  name: cifar10",
                "  data_dir: data",
                "logging:",
                "  wandb:",
                "    enabled: true",
            ]
        ),
        encoding="utf-8",
    )
    child_path.write_text(
        "\n".join(
            [
                "_base_: base.yaml",
                "experiment:",
                "  seed: 42",
                "train:",
                "  mode: erm",
            ]
        ),
        encoding="utf-8",
    )
    overlay_path.write_text(
        "\n".join(
            [
                "runtime:",
                "  profile: dev_fast",
                "logging:",
                "  wandb:",
                "    mode: offline",
            ]
        ),
        encoding="utf-8",
    )

    cfg = load_config(str(child_path), extra_paths=[str(overlay_path)])

    assert cfg["experiment"]["seed"] == 42
    assert cfg["dataset"]["name"] == "cifar10"
    assert cfg["train"]["mode"] == "erm"
    assert cfg["runtime"]["profile"] == "dev_fast"
    assert cfg["logging"]["wandb"]["mode"] == "offline"
    assert cfg["_meta"]["config_sources"] == [
        str(base_path.resolve()),
        str(child_path.resolve()),
        str(overlay_path.resolve()),
    ]


def test_default_runtime_config_points_to_resnet18_local_profile() -> None:
    cfg = load_runtime_config("configs/default.yaml")

    assert cfg["model"]["name"] == "resnet18_cifar"
    assert cfg["experiment"]["name"] == "baseline_erm"
    assert cfg["experiment"]["runtime_profile"] == "local_gpu"
    assert cfg["logging"]["wandb"]["enabled"] is True
