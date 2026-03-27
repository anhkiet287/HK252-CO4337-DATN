"""Run smoke-test training and evaluation across configs."""

import argparse
from typing import List

from ardg.experiments.smoke import run_smoke


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments.

    Raises:
        SystemExit: If argument parsing fails.
    """
    parser = argparse.ArgumentParser(description="Run smoke tests for multiple configs.")
    parser.add_argument(
        "--configs",
        nargs="+",
        default=[
            "configs/experiments/cifar10/resnet18/baselines/erm.yaml",
            "configs/experiments/cifar10/resnet50/baselines/erm.yaml",
            "configs/experiments/cifar10/vit_b16/baselines/erm.yaml",
        ],
        help="List of experiment config files to run.",
    )
    parser.add_argument(
        "--profile",
        default="configs/profiles/dev_fast.yaml",
        help="Runtime profile overlay used for smoke runs.",
    )
    return parser.parse_args()


def main() -> None:
    """Run smoke tests for the provided configs.

    Returns:
        None.

    Raises:
        RuntimeError: If a smoke test fails.
    """
    args = parse_args()
    cfg_paths: List[str] = list(args.configs)
    run_smoke(cfg_paths, profile_path=args.profile)


if __name__ == "__main__":
    main()
