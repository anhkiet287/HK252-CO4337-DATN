"""Prepare deterministic CIFAR split artifacts.

Loads a YAML config, ensures the train/val split file exists, and prints
basic metadata for downstream stages.
"""

import argparse

from ardg.config import DEFAULT_CONFIG_PATH, load_config
from ardg.data.splits import ensure_split, load_splits


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Prepare CIFAR split artifacts.")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="Path to YAML config.")
    return parser.parse_args()


def main() -> None:
    """Create or reuse a deterministic split artifact for CIFAR.

    Args:
        None.

    Returns:
        None.

    Side effects:
        Creates the split JSON file when missing and prints summary info to stdout.
    """
    args = parse_args()
    cfg = load_config(args.config)
    split_path = ensure_split(cfg)
    splits = load_splits(split_path)

    split_groups = 0
    if isinstance(splits, dict):
        split_groups = len(splits.get("splits", {}))

    print(f"Split file: {split_path}")
    print(f"Split groups: {split_groups}")


if __name__ == "__main__":
    main()
