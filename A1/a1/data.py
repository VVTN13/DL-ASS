"""Load Fashion-MNIST and prepare the train, validation, and test views."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

SEED = 42
DATA_DIR = "data"
VAL_RATIO = 0.10
NUM_CLASSES = 10


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_raw_datasets(data_dir: str = DATA_DIR):
    to_tensor = transforms.ToTensor()
    train = datasets.FashionMNIST(data_dir, train=True, download=True, transform=to_tensor)
    test = datasets.FashionMNIST(data_dir, train=False, download=True, transform=to_tensor)
    return train, test


def stratified_train_val_split(labels, val_ratio: float = VAL_RATIO, seed: int = SEED):
    indices = np.arange(len(labels))
    train_idx, val_idx = train_test_split(
        indices, test_size=val_ratio, random_state=seed, stratify=np.asarray(labels)
    )
    return train_idx.astype(np.int64), val_idx.astype(np.int64)


def _source_hash(dataset) -> str:
    digest = hashlib.sha256()
    digest.update(dataset.data.numpy().tobytes())
    digest.update(dataset.targets.numpy().tobytes())
    return digest.hexdigest()


def _split_hash(train_idx, val_idx) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(train_idx, dtype=np.int64).tobytes())
    digest.update(np.asarray(val_idx, dtype=np.int64).tobytes())
    return digest.hexdigest()


def compute_normalization_stats(dataset: Dataset, batch_size: int = 1024):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    total, total_sq, count = 0.0, 0.0, 0
    for images, _ in loader:
        total += images.sum(dtype=torch.float64).item()
        total_sq += images.square().sum(dtype=torch.float64).item()
        count += images.numel()
    mean = total / count
    std = max(total_sq / count - mean * mean, 0.0) ** 0.5
    if not np.isfinite(mean) or not np.isfinite(std) or std <= 0:
        raise ValueError("Invalid normalization statistics")
    return mean, std


def _valid_cache(cache, *, expected_train, expected_val, source_hash, seed, val_ratio):
    expected_details = {
        "source": "torchvision:FashionMNIST",
        "source_sha256": source_hash,
        "seed": seed,
        "val_ratio": val_ratio,
        "n_samples": len(expected_train) + len(expected_val),
    }
    if any(cache.get(key) != value for key, value in expected_details.items()):
        return False

    try:
        train_idx = np.asarray(cache["train_idx"], dtype=np.int64)
        val_idx = np.asarray(cache["val_idx"], dtype=np.int64)
        mean, std = float(cache["mean"]), float(cache["std"])
        return (
            np.array_equal(train_idx, expected_train)
            and np.array_equal(val_idx, expected_val)
            and cache.get("split_sha256") == _split_hash(train_idx, val_idx)
            and np.isfinite(mean)
            and np.isfinite(std)
            and std > 0
        )
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


def prepare_split(train_full, *, val_ratio=VAL_RATIO, seed=SEED, split_cache_path="split_cache.json"):
    """Reuse a matching split cache, or calculate the split and statistics again."""
    labels = train_full.targets.numpy()
    train_idx, val_idx = stratified_train_val_split(labels, val_ratio, seed)
    if len(train_idx) + len(val_idx) != len(labels) or np.intersect1d(train_idx, val_idx).size:
        raise AssertionError("Train and validation indices must partition official training data")
    source_hash = _source_hash(train_full)
    cache_path = Path(split_cache_path) if split_cache_path else None
    cache = None
    if cache_path and cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text())
        except (OSError, json.JSONDecodeError):
            cache = None
    if cache and _valid_cache(
        cache, expected_train=train_idx, expected_val=val_idx,
        source_hash=source_hash, seed=seed, val_ratio=val_ratio,
    ):
        mean, std = float(cache["mean"]), float(cache["std"])
    else:
        # Validation and test images must not affect the normalization values.
        mean, std = compute_normalization_stats(Subset(train_full, train_idx))
        if cache_path:
            payload = {
                "source": "torchvision:FashionMNIST",
                "source_sha256": source_hash,
                "seed": seed,
                "val_ratio": val_ratio,
                "n_samples": len(labels),
                "train_idx": train_idx.tolist(),
                "val_idx": val_idx.tolist(),
                "split_sha256": _split_hash(train_idx, val_idx),
                "mean": mean,
                "std": std,
            }
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(payload, separators=(",", ":")) + "\n")
    return train_idx, val_idx, mean, std, source_hash


def to_representation(image: torch.Tensor, mode: str = "flatten", patch_size: int = 4):
    if mode == "flatten":
        return image.reshape(-1)
    if mode == "image":
        return image
    if mode == "row_seq":
        return image.squeeze(0)
    if mode == "col_seq":
        return image.squeeze(0).t().contiguous()
    if mode == "patch_seq":
        _, height, width = image.shape
        if patch_size <= 0 or height % patch_size or width % patch_size:
            raise ValueError("patch_size must divide both image dimensions")
        patches = image.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
        return patches.contiguous().view(-1, patch_size * patch_size)
    raise ValueError(f"Unknown representation mode: {mode!r}")


class FashionMNISTView(Dataset):
    def __init__(self, base_dataset, indices, mean, std, mode="flatten", patch_size=4, augment=False):
        self.base = base_dataset
        self.indices = np.asarray(indices, dtype=np.int64)
        self.mean = float(mean)
        self.std = float(std)
        self.mode = mode
        self.patch_size = patch_size
        self.augment = None
        if augment:
            self.augment = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(10),
            ])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        image, label = self.base[int(self.indices[index])]
        if self.augment is not None:
            image = self.augment(image)
        image = (image - self.mean) / self.std
        return to_representation(image, self.mode, self.patch_size), int(label)


def build_dataloaders(
    mode="flatten", batch_size=128, val_ratio=VAL_RATIO, seed=SEED,
    patch_size=4, augment_train=False, data_dir=DATA_DIR, num_workers=0,
    split_cache_path="split_cache.json",
):
    set_seed(seed)
    train_full, test_set = load_raw_datasets(data_dir)
    train_idx, val_idx, mean, std, source_hash = prepare_split(
        train_full, val_ratio=val_ratio, seed=seed, split_cache_path=split_cache_path
    )
    train_ds = FashionMNISTView(train_full, train_idx, mean, std, mode, patch_size, augment_train)
    val_ds = FashionMNISTView(train_full, val_idx, mean, std, mode, patch_size)
    test_ds = FashionMNISTView(test_set, np.arange(len(test_set)), mean, std, mode, patch_size)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        generator=generator, num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
    )
    split_hash = _split_hash(train_idx, val_idx)
    for loader in (train_loader, val_loader, test_loader):
        loader.source_sha256 = source_hash
        loader.split_sha256 = split_hash
    return train_loader, val_loader, test_loader
