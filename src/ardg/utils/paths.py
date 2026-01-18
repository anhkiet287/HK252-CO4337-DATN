"""Path utilities."""

from pathlib import Path
from typing import Any, Dict


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
    output_dir = cfg.get("logging", {}).get("output_dir", "outputs")
    run_name = cfg.get("logging", {}).get("run_name", "run")
    return str(project_root() / output_dir / run_name)


def ensure_dir(path: str) -> None:
    """Ensure a directory exists.

    Args:
        path: Directory path to create.

    Side effects:
        Creates the directory path if it does not exist.
    """
    Path(path).mkdir(parents=True, exist_ok=True)
