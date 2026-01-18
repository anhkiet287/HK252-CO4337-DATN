"""Seeding utilities."""

import os
import random

try:
    import numpy as np  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    np = None


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    if np is not None:
        np.random.seed(seed)


def get_seed(env_var: str = "SEED", default: int = 42) -> int:
    value = os.getenv(env_var)
    return int(value) if value is not None else default
