"""Platform tag helpers."""

import os
from typing import Optional


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
    return "local"
