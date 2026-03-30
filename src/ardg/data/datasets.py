"""Dataset helpers."""

from typing import Any, Tuple

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from ardg.data.grouping import GroupHomogeneousBatchSampler, RepeatedGroupDataset
from ardg.utils.data import normalize_dataset_name
from ardg.data.splits import ensure_split, load_splits
from ardg.data.transforms import build_transforms

_COLOR_MAP = torch.tensor(
    [
        [0.90, 0.20, 0.20],
        [0.20, 0.70, 0.20],
        [0.20, 0.40, 0.90],
        [0.85, 0.60, 0.15],
        [0.60, 0.25, 0.85],
        [0.10, 0.75, 0.75],
        [0.75, 0.35, 0.10],
        [0.50, 0.50, 0.50],
        [0.95, 0.35, 0.65],
        [0.30, 0.85, 0.50],
    ],
    dtype=torch.float32,
)


class ColorizedMNIST(Dataset):
    """MNIST wrapper that colorizes digits based on the label."""

    def __init__(
        self,
        root: str,
        train: bool,
        download: bool,
        transform: Any = None,
        target_transform: Any = None,
    ) -> None:
        self.base = datasets.MNIST(root=root, train=train, download=download, transform=None)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        image, target = self.base[index]
        tensor = transforms.ToTensor()(image)
        color = _COLOR_MAP[int(target)].view(3, 1, 1)
        colored = tensor.repeat(3, 1, 1) * color
        if self.transform is not None:
            colored = self.transform(colored)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return colored, target


def load_dataset(name: str, root: str, train: bool, transform: Any) -> Any:
    """Load a supported dataset."""
    key = normalize_dataset_name(name)
    if key == "cifar10":
        return datasets.CIFAR10(root=root, train=train, download=True, transform=transform)
    if key == "cifar100":
        return datasets.CIFAR100(root=root, train=train, download=True, transform=transform)
    if key == "mnist":
        return datasets.MNIST(root=root, train=train, download=True, transform=transform)
    if key == "fashion-mnist":
        return datasets.FashionMNIST(root=root, train=train, download=True, transform=transform)
    if key == "color-mnist":
        return ColorizedMNIST(root=root, train=train, download=True, transform=transform)
    raise ValueError(f"Unsupported dataset: {name}")


def get_dataloaders(cfg: dict) -> Tuple[Any, Any, Any]:
    """Build train/val/test dataloaders for supported datasets."""
    name = cfg["dataset"]["name"]
    data_dir = cfg["dataset"].get("data_dir", "data")
    train_cfg = cfg.get("train", {})
    eval_cfg = cfg.get("eval", {})
    dataset_cfg = cfg.get("dataset", {})
    batch_size = int(
        train_cfg.get(
            "batch_size",
            eval_cfg.get("batch_size", dataset_cfg.get("batch_size", 128)),
        )
    )
    if batch_size <= 0:
        raise ValueError(f"Invalid batch size: {batch_size}. Must be >= 1.")
    num_workers = cfg["dataset"].get("num_workers", 0)

    split_path = ensure_split(cfg)
    splits = load_splits(split_path)["splits"]

    train_transform = build_transforms(cfg, "train")
    eval_transform = build_transforms(cfg, "val")

    train_ds_aug = load_dataset(name, data_dir, train=True, transform=train_transform)
    train_ds_eval = load_dataset(name, data_dir, train=True, transform=eval_transform)
    test_ds = load_dataset(name, data_dir, train=False, transform=eval_transform)

    train_idx = splits["train"]
    val_idx = splits["val"]

    max_train = cfg["dataset"].get("max_train_samples")
    max_val = cfg["dataset"].get("max_val_samples")
    max_test = cfg["dataset"].get("max_test_samples")

    if max_train:
        train_idx = train_idx[: int(max_train)]
    if max_val:
        val_idx = val_idx[: int(max_val)]

    train_subset = Subset(train_ds_aug, train_idx)
    val_subset = Subset(train_ds_eval, val_idx)
    if max_test:
        test_indices = list(range(min(int(max_test), len(test_ds))))
        test_ds = Subset(test_ds, test_indices)

    pin = torch.cuda.is_available()

    if _use_groupdro_native_loader(cfg) and _has_groupdro_train_domains(cfg):
        num_groups = _resolve_groupdro_num_groups(cfg)
        grouped_train = RepeatedGroupDataset(train_subset, num_groups=num_groups)
        batch_sampler = GroupHomogeneousBatchSampler(
            base_size=len(train_subset),
            num_groups=num_groups,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
        )
        train_loader = DataLoader(
            grouped_train,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            pin_memory=pin,
        )
    else:
        train_loader = DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin,
        )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
    )
    return train_loader, val_loader, test_loader


def _use_groupdro_native_loader(cfg: dict) -> bool:
    mode = str(cfg.get("train", {}).get("mode", "")).lower()
    return mode in {"groupdro", "group_dro"}


def _has_groupdro_train_domains(cfg: dict) -> bool:
    train_domains = cfg.get("attack", {}).get("train_domains")
    return isinstance(train_domains, list) and bool(train_domains)


def _resolve_groupdro_num_groups(cfg: dict) -> int:
    train_domains = cfg.get("attack", {}).get("train_domains")
    if not isinstance(train_domains, list) or not train_domains:
        raise ValueError(
            "GroupDRO native mode requires attack.train_domains to define the fixed groups."
        )
    return len(train_domains)
