from __future__ import annotations

import io
import logging

from ardg.utils.logging import log_metrics


def test_log_metrics_formats_numeric_values_with_fixed_decimals() -> None:
    stream = io.StringIO()
    logger = logging.getLogger("test_log_metrics_formats_numeric_values_with_fixed_decimals")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.propagate = False

    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)

    log_metrics(
        logger,
        {
            "acc": 1.0,
            "loss": 1.2934,
            "batch_size": 64,
            "worst_group": "pgd_ce_linf",
            "enabled": True,
        },
        step=1,
        split="train",
    )

    message = stream.getvalue().strip()
    assert "step=1" in message
    assert "train/acc=1.000" in message
    assert "train/loss=1.293" in message
    assert "train/batch_size=64.000" in message
    assert "train/worst_group=pgd_ce_linf" in message
    assert "train/enabled=True" in message
