"""Train a model with the ARDG pipeline.

Loads config, sets seeds, builds data/model/trainer components, and writes
checkpoints and run summaries to outputs/.
"""

import argparse

from ardg.config import DEFAULT_CONFIG_PATH
from ardg.experiments.common import build_loaders, setup_run
from ardg.models.factory import build_model
from ardg.training.trainer1 import Trainer1


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train a model.")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="Path to YAML config.")
    return parser.parse_args()


def main() -> None:
    """Run end-to-end training.

    Args:
        None.

    Returns:
        None.

    Side effects:
        Writes outputs/checkpoints and may initialize wandb logging.
    """
    # Parsing arguments and loading config.
    args = parse_args()
    cfg, logger, _, device = setup_run(args.config)

    # Building data, model, and trainer components.
    train_loader, val_loader, _ = build_loaders(cfg)
    model = build_model(cfg)
    trainer = Trainer1(
        cfg,
        model,
        train_loader,
        val_loader,
        device=device,
        logger=logger,
    )
    trainer.train()


if __name__ == "__main__":
    main()
