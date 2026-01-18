from pathlib import Path

from ardg.data.splits import ensure_split, load_splits
from ardg.utils.paths import ensure_dir, get_run_dir


def test_smoke_split_creation(tmp_path) -> None:
    """Ensure split artifact is created and loadable."""
    data_dir = tmp_path / "data"
    cfg = {
        "dataset": {"name": "cifar10", "val_ratio": 0.02, "data_dir": str(data_dir)},
        "experiment": {"seed": 42},
    }

    split_path = ensure_split(cfg)
    assert Path(split_path).exists()

    splits = load_splits(split_path)
    assert splits["dataset"] == "cifar10"
    assert splits["seed"] == 42
    assert splits["val_ratio"] == 0.02
    assert "splits" in splits


def test_smoke_paths_run_dir(tmp_path) -> None:
    """Ensure run directory path is computed and can be created."""
    cfg = {"logging": {"output_dir": str(tmp_path / "outputs"), "run_name": "test_run"}}
    run_dir = get_run_dir(cfg)

    ensure_dir(run_dir)
    assert Path(run_dir).is_dir()