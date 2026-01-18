"""Dataset split helpers."""

import json
from pathlib import Path
from typing import Any, Dict


def ensure_split(cfg: Dict[str, Any]) -> str:
    """Ensure a deterministic train/val split artifact exists.

    Args:
        cfg: Configuration dictionary containing dataset.name, dataset.val_ratio,
            dataset.data_dir, and experiment.seed.

    Returns:
        Path to the split JSON file.

    Side effects:
        Creates the split directory and writes a JSON file when missing.
    """
    dataset = cfg["dataset"]["name"]
    seed = int(cfg["experiment"]["seed"])
    val_ratio = float(cfg["dataset"]["val_ratio"])
    data_dir = Path(cfg["dataset"].get("data_dir", "data"))

    token = f"{val_ratio:.2f}".replace(".", "p")
    filename = f"{dataset}_seed{seed}_val{token}.json"
    split_dir = data_dir / "processed" / "splits"
    split_path = split_dir / filename

    if not split_path.exists():
        split_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "dataset": dataset,
            "seed": seed,
            "val_ratio": val_ratio,
            "splits": {},
        }
        with split_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    return str(split_path)


def load_splits(path: str) -> Dict[str, Any]:
    """Load a split artifact from disk.

    Args:
        path: Path to a JSON split file.

    Returns:
        Dictionary containing split metadata and indices.
    """
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)
