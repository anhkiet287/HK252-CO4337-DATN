"""Train a CIFAR model with the ARDG pipeline.

Loads config, sets seeds, builds data/model/trainer components, and writes
checkpoints and run summaries to outputs/.
"""

import argparse

from ardg.config import DEFAULT_CONFIG_PATH, load_config
from ardg.data.cifar import get_dataloaders
from ardg.models.resnet import resnet18_cifar
from ardg.training.trainer import Trainer
from ardg.utils.logging import init_wandb, setup_logging
from ardg.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train a CIFAR model.")
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
    args = parse_args()
    cfg = load_config(args.config)

    set_seed(cfg["experiment"]["seed"])
    logger = setup_logging()
    init_wandb(cfg)

    train_loader, val_loader, _ = get_dataloaders(cfg)
    model = resnet18_cifar(cfg["model"]["num_classes"])
    trainer = Trainer(
        cfg,
        model,
        train_loader,
        val_loader,
        device=cfg["experiment"].get("device", "cpu"),
        logger=logger,
    )
    trainer.train()


if __name__ == "__main__":
    main()
