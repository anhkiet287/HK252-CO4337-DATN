"""Seeding utilities."""

import os
import random

try:
    import numpy as np  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    np = None

try:
    import torch  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    torch = None


def set_seed(seed: int) -> None:
    """Set RNG seeds for reproducibility.

    Args:
        seed: Random seed value.

    Side effects:
        Sets PYTHONHASHSEED, seeds random/numpy/torch RNGs, and configures
        torch.backends flags for deterministic behavior when available.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    if np is not None:
        np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_seed(env_var: str = "SEED", default: int = 42) -> int:
    """Read a seed value from the environment.

    Args:
        env_var: Environment variable name.
        default: Default seed if env var is missing.

    Returns:
        Seed value as an integer.
    """
    value = os.getenv(env_var)
    return int(value) if value is not None else default
