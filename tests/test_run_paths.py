from __future__ import annotations

from pathlib import Path

from ardg.utils.paths import (
    find_checkpoint,
    get_checkpoint_path,
    get_eval_summary_path,
    get_run_root_from_artifact,
    initialize_run_layout,
)


def test_find_checkpoint_prefers_standardized_checkpoints_dir(tmp_path: Path) -> None:
    run_dir = tmp_path / "resnet18_local_baseline_erm_seed42"
    initialize_run_layout(str(run_dir))
    standardized = get_checkpoint_path(str(run_dir), "best")
    legacy = run_dir / "best.pt"
    standardized.write_text("standardized", encoding="utf-8")
    legacy.write_text("legacy", encoding="utf-8")

    assert find_checkpoint(str(run_dir), names=("best",)) == str(standardized)


def test_get_run_root_from_nested_artifacts() -> None:
    run_dir = Path("/tmp/example_run")

    assert get_run_root_from_artifact(str(run_dir / "checkpoints" / "best.pt")) == run_dir
    assert get_run_root_from_artifact(str(run_dir / "eval" / "summary.json")) == run_dir
    assert get_run_root_from_artifact(str(run_dir / "wandb" / "run_id.txt")) == run_dir


def test_standard_eval_summary_path_lives_under_eval_subdir(tmp_path: Path) -> None:
    run_dir = tmp_path / "vit_b16_h100_baseline_erm_seed42_eval"
    initialize_run_layout(str(run_dir))

    assert get_eval_summary_path(str(run_dir)) == run_dir / "eval" / "summary.json"
