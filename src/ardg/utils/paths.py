"""Path utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable

from ardg.utils.platform import resolve_platform

DEFAULT_LOCAL_OUTPUT_DIR = "outputs"
DEFAULT_COLAB_OUTPUT_DIR = "/content/drive/MyDrive/ardg/HK252-CO4337-DATN/outputs"
DEFAULT_LOCAL_CONTENT_ROOT = "/content"
DEFAULT_COLAB_CONTENT_ROOT = "/content/drive/MyDrive"


def default_output_dir(cfg: Dict[str, Any]) -> str:
    """Resolve default output root by runtime platform."""
    platform = resolve_platform(cfg.get("experiment", {}).get("platform"))
    if str(platform).lower() == "colab":
        return DEFAULT_COLAB_OUTPUT_DIR
    return DEFAULT_LOCAL_OUTPUT_DIR


def platform_workspace_root(platform: str) -> Path:
    """Return the expected repository root for a runtime platform override."""
    repo_name = project_root().name
    if str(platform).lower() == "colab":
        return Path(DEFAULT_COLAB_CONTENT_ROOT) / repo_name
    if str(platform).lower() == "local":
        return Path(DEFAULT_LOCAL_CONTENT_ROOT) / repo_name
    return project_root()


def project_root() -> Path:
    """Return the repository root path."""
    return Path(__file__).resolve().parents[3]


def get_run_dir(cfg: Dict[str, Any]) -> str:
    """Compute a run directory based on config values.

    Args:
        cfg: Configuration dictionary with logging settings.

    Returns:
        Absolute path to the run directory.
    """
    output_dir = cfg.get("logging", {}).get("output_dir")
    if output_dir is None or str(output_dir).strip() == "":
        output_dir = default_output_dir(cfg)
    run_name = str(cfg.get("logging", {}).get("run_name", "") or "").strip() or "run"
    output_root = Path(str(output_dir)).expanduser()
    if not output_root.is_absolute():
        output_root = project_root() / output_root
    return str(output_root / run_name)


def get_logs_dir(run_dir: str) -> Path:
    return Path(run_dir) / "logs"


def get_checkpoints_dir(run_dir: str) -> Path:
    return Path(run_dir) / "checkpoints"


def get_checkpoint_path(run_dir: str, name: str) -> Path:
    return get_checkpoints_dir(run_dir) / f"{name}.pt"


def get_legacy_checkpoint_path(run_dir: str, name: str) -> Path:
    return Path(run_dir) / f"{name}.pt"


def get_train_dir(run_dir: str) -> Path:
    return Path(run_dir) / "train"


def get_train_summary_path(run_dir: str) -> Path:
    return get_train_dir(run_dir) / "summary.json"


def get_legacy_train_summary_path(run_dir: str) -> Path:
    return Path(run_dir) / "train_summary.json"


def get_eval_dir(run_dir: str) -> Path:
    return Path(run_dir) / "eval"


def get_eval_summary_path(run_dir: str) -> Path:
    return get_eval_dir(run_dir) / "summary.json"


def get_legacy_eval_summary_path(run_dir: str) -> Path:
    return Path(run_dir) / "eval_test_summary.json"


def get_wandb_dir(run_dir: str) -> Path:
    return Path(run_dir) / "wandb"


def get_wandb_run_id_path(run_dir: str) -> Path:
    return get_wandb_dir(run_dir) / "run_id.txt"


def get_legacy_wandb_run_id_path(run_dir: str) -> Path:
    return Path(run_dir) / "wandb_run_id.txt"


def get_wandb_run_url_path(run_dir: str) -> Path:
    return get_wandb_dir(run_dir) / "run_url.txt"


def get_legacy_wandb_run_url_path(run_dir: str) -> Path:
    return Path(run_dir) / "wandb_run_url.txt"


def initialize_run_layout(run_dir: str) -> Dict[str, str]:
    root = Path(run_dir)
    layout = {
        "run_dir": str(root),
        "resolved_config": str(root / "resolved_config.yaml"),
        "run_manifest": str(root / "run_manifest.json"),
        "logs_dir": str(get_logs_dir(run_dir)),
        "train_log": str(get_logs_dir(run_dir) / "train.log"),
        "eval_log": str(get_logs_dir(run_dir) / "eval.log"),
        "preflight_log": str(get_logs_dir(run_dir) / "preflight.log"),
        "smoke_log": str(get_logs_dir(run_dir) / "smoke.log"),
        "checkpoints_dir": str(get_checkpoints_dir(run_dir)),
        "train_summary": str(get_train_summary_path(run_dir)),
        "eval_summary": str(get_eval_summary_path(run_dir)),
        "wandb_dir": str(get_wandb_dir(run_dir)),
        "wandb_run_id": str(get_wandb_run_id_path(run_dir)),
        "wandb_run_url": str(get_wandb_run_url_path(run_dir)),
    }
    for key in ("logs_dir", "checkpoints_dir", "wandb_dir"):
        Path(layout[key]).mkdir(parents=True, exist_ok=True)
    get_train_dir(run_dir).mkdir(parents=True, exist_ok=True)
    get_eval_dir(run_dir).mkdir(parents=True, exist_ok=True)
    root.mkdir(parents=True, exist_ok=True)
    return layout


def find_checkpoint(run_dir: str, names: Iterable[str] = ("best", "last")) -> str | None:
    for name in names:
        new_path = get_checkpoint_path(run_dir, name)
        if new_path.exists():
            return str(new_path)
        legacy_path = get_legacy_checkpoint_path(run_dir, name)
        if legacy_path.exists():
            return str(legacy_path)
    return None


def get_run_root_from_artifact(path: str) -> Path:
    artifact_path = Path(path)
    if artifact_path.parent.name in {"checkpoints", "wandb", "eval", "train", "logs"}:
        return artifact_path.parent.parent
    return artifact_path.parent


def ensure_dir(path: str) -> None:
    """Ensure a directory exists.

    Args:
        path: Directory path to create.

    Side effects:
        Creates the directory path if it does not exist.
    """
    Path(path).mkdir(parents=True, exist_ok=True)
