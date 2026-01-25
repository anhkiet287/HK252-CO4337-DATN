"""Shared data helpers."""

def normalize_dataset_name(name: str) -> str:
    """Normalize dataset names for internal lookup.

    Args:
        name: Dataset name from config or CLI.

    Returns:
        Normalized dataset key (lowercase, underscores to hyphens).
    """
    return name.lower().replace("_", "-")
