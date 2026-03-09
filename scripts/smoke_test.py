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
            "configs/smoke_quick_resnet18_erm.yaml",
            "configs/smoke_quick_resnet18_pgd_at.yaml",
        ],
        help="List of config files to run.",
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
    run_smoke(cfg_paths)


if __name__ == "__main__":
    main()
