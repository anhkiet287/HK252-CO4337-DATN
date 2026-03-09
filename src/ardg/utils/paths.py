"""Path utilities."""

from pathlib import Path
from typing import Any, Dict

from ardg.utils.platform import resolve_platform

DEFAULT_LOCAL_OUTPUT_DIR = "outputs"
DEFAULT_COLAB_OUTPUT_DIR = "/content/drive/MyDrive/ardg/HK252-CO4337-DATN/outputs"


def default_output_dir(cfg: Dict[str, Any]) -> str:
    """Resolve default output root by runtime platform."""
    platform = resolve_platform(cfg.get("experiment", {}).get("platform"))
    if str(platform).lower() == "colab":
        return DEFAULT_COLAB_OUTPUT_DIR
    return DEFAULT_LOCAL_OUTPUT_DIR


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
    run_name = cfg.get("logging", {}).get("run_name", "run")
    output_root = Path(str(output_dir)).expanduser()
    if not output_root.is_absolute():
        output_root = project_root() / output_root
    return str(output_root / run_name)


def ensure_dir(path: str) -> None:
    """Ensure a directory exists.

    Args:
        path: Directory path to create.

    Side effects:
        Creates the directory path if it does not exist.
    """
    Path(path).mkdir(parents=True, exist_ok=True)
