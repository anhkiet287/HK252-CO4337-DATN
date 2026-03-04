"""Platform tag helpers."""

import os
from typing import Optional


def _is_colab_runtime() -> bool:
    """Best-effort Google Colab detection."""
    if os.getenv("COLAB_RELEASE_TAG"):
        return True
    if os.getenv("COLAB_GPU") is not None:
        return True
    try:
        import google.colab  # type: ignore  # noqa: F401
    except Exception:
        return False
    return True


def resolve_platform(config_value: Optional[str]) -> str:
    """Resolve platform tag for logging and metadata.

    Args:
        config_value: Platform value from config, if set.

    Returns:
        Platform tag string.
    """
    if config_value:
        return str(config_value)
    env_value = os.getenv("ARDG_PLATFORM")
    if env_value:
        return env_value
    if _is_colab_runtime():
        return "colab"
    return "local"
