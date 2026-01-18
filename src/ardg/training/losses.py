"""Loss functions and helpers."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def cross_entropy_loss(logits: "torch.Tensor", targets: "torch.Tensor") -> "torch.Tensor":
    """Compute cross-entropy loss.

    Args:
        logits: Logits tensor of shape (B, C), float32.
        targets: Target labels of shape (B,), dtype torch.long.

    Returns:
        Scalar loss tensor of shape ().
    """
    try:
        import torch.nn.functional as functional  # type: ignore
    except ImportError as exc:
        raise ImportError("PyTorch is required to compute loss.") from exc

    return functional.cross_entropy(logits, targets)


def compute_loss(logits: "torch.Tensor", targets: "torch.Tensor") -> "torch.Tensor":
    """Alias for cross_entropy_loss.

    Args:
        logits: Logits tensor of shape (B, C).
        targets: Target labels of shape (B,).

    Returns:
        Scalar loss tensor.
    """
    return cross_entropy_loss(logits, targets)
