#!/usr/bin/env python3
"""
Kaggle Dataset Handler for Handwritten Math Symbols
Handles loading and preprocessing of the Kaggle dataset
"""

import os
import random
import logging
from collections import defaultdict
from pathlib import Path
from typing import Tuple, Dict

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms
from PIL import Image

logger = logging.getLogger(__name__)


# ---------- Safe ImageFolder ----------

class SafeImageFolder(datasets.ImageFolder):
    """ImageFolder that skips classes with no valid images."""

    def find_classes(self, directory):
        classes, class_to_idx = super().find_classes(directory)

        valid_classes = []
        valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

        for c in classes:
            folder = Path(directory) / c
            has_image = any(
                f.suffix.lower() in valid_exts and f.is_file()
                for f in folder.rglob("*")
            )
            if has_image:
                valid_classes.append(c)
            else:
                print(f"⚠️ Skipping empty/broken class: {c}")

        valid_class_to_idx = {c: i for i, c in enumerate(valid_classes)}
        return valid_classes, valid_class_to_idx


# ---------- helpers ----------

def _build_label_to_samples(img_folder: datasets.ImageFolder) -> Dict[int, list]:
    """Map original label idx -> list[(path, label)] that actually exist."""
    label_to_samples = defaultdict(list)
    for path, lbl in img_folder.samples:
        label_to_samples[lbl].append((path, lbl))
    return label_to_samples


def _select_balanced_subset(
    img_folder: datasets.ImageFolder,
    max_classes: int = 20,
    k_per_class: int = 3,
    total_cap: int = 50,
    seed: int = 42,
):
    rng = random.Random(seed)
    label_to_samples = _build_label_to_samples(img_folder)

    # keep only classes that actually have files
    non_empty = [(lbl, img_folder.classes[lbl]) for lbl in range(len(img_folder.classes))
                 if len(label_to_samples[lbl]) > 0]

    if not non_empty:
        raise RuntimeError(
            f"No valid images found in {img_folder.root}. "
            f"Ensure subfolders contain .png/.jpg files."
        )

    non_empty.sort(key=lambda x: x[1])
    chosen = non_empty[:max_classes]

    old2new = {old: new for new, (old, _) in enumerate(chosen)}
    classes = [name for _, name in chosen]

    picked = []
    for old_lbl, _name in chosen:
        candidates = list(label_to_samples[old_lbl])
        rng.shuffle(candidates)
        picked.extend(candidates[:k_per_class])

    rng.shuffle(picked)
    picked = picked[:total_cap]

    picked = [(p, old2new[l]) for (p, l) in picked]
    return picked, classes, old2new


# ---------- dataset ----------

class KaggleMathDataset(Dataset):
    """Custom dataset for Kaggle handwritten math symbols"""

    def __init__(self, root_dir: str, transform=None, split='train', max_classes=20):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.split = split

        self.dataset = SafeImageFolder(
            root=str(self.root_dir),
            transform=None
        )

        valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
        valid_samples = [
            (path, label)
            for path, label in self.dataset.samples
            if Path(path).suffix.lower() in valid_exts and Path(path).is_file()
        ]

        self.dataset.samples = valid_samples
        self.dataset.imgs = valid_samples

        self.classes = self.dataset.classes[:max_classes]
        self.class_to_idx = {
            k: v for k, v in self.dataset.class_to_idx.items() if v < max_classes
        }

        samples = []
        for label, class_name in enumerate(self.dataset.classes[:max_classes]):
            class_samples = [
                (path, l) for path, l in self.dataset.samples if l == label
            ]
            if not class_samples:
                logger.warning(f"Skipping empty class: {class_name}")
                continue
            class_samples = class_samples[:3]
            samples.extend(class_samples)

        self.samples = samples[:50]

        logger.info(
            f"Loaded Kaggle dataset with {len(self.samples)} samples and {len(self.classes)} classes"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


# ---------- transforms ----------

def get_default_transforms(image_size=(64, 64), augment=True):
    if augment:
        train_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize(image_size),
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    val_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    return train_transform, val_transform


# ---------- loaders ----------

def load_kaggle_dataset(
    data_path: str = "data/kaggle_math",
    batch_size: int = 2,
    image_size: Tuple[int, int] = (64, 64),
    train_split: float = 0.8,
    num_workers: int = 0,
    shuffle: bool = True,
    augment: bool = True,
) -> Tuple[DataLoader, DataLoader, int, Dict]:
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Kaggle dataset not found at {data_path}")

    logger.info(f"Loading Kaggle dataset from {data_path}")

    train_transform, val_transform = get_default_transforms(image_size, augment)

    base_train = KaggleMathDataset(
        root_dir=str(data_path),
        transform=train_transform,
        split="train",
        max_classes=20,
    )
    base_val = KaggleMathDataset(
        root_dir=str(data_path),
        transform=val_transform,
        split="val",
        max_classes=20,
    )

    total_samples = len(base_train)
    if total_samples < 2:
        raise RuntimeError(
            f"Not enough samples after filtering in {data_path}. "
            f"Found {total_samples}. Ensure folders contain images."
        )

    train_size = int(train_split * total_samples)
    val_size = total_samples - train_size
    train_indices, val_indices = random_split(
        range(total_samples),
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_dataset = torch.utils.data.Subset(base_train, train_indices)
    val_dataset = torch.utils.data.Subset(base_val, val_indices)

    class_info = {
        "classes": base_train.classes,
        "class_to_idx": base_train.class_to_idx,
        "num_classes": len(base_train.classes),
        "total_samples": total_samples,
        "train_samples": train_size,
        "val_samples": val_size,
    }

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    return train_loader, val_loader, len(base_train.classes), class_info


def create_kaggle_data_loaders(config: Dict) -> Tuple[DataLoader, DataLoader, int]:
    data_config = config.get("data", {})
    training_config = config.get("training", {})

    data_path = data_config.get("kaggle_data_dir", "data/kaggle_math")
    batch_size = training_config.get("batch_size", 2)
    image_size = tuple(data_config.get("image_size", [64, 64]))
    num_workers = 0
    train_split = data_config.get("train_split", 0.8)

    train_loader, val_loader, num_classes, _ = load_kaggle_dataset(
        data_path=data_path,
        batch_size=batch_size,
        image_size=image_size,
        train_split=train_split,
        num_workers=num_workers,
    )

    return train_loader, val_loader, num_classes


def get_dataset_info(data_path: str = "data/kaggle_math", max_classes: int = 20) -> Dict:
    data_path = Path(data_path)
    if not data_path.exists():
        return {"error": f"Dataset not found at {data_path}"}

    img_folder = SafeImageFolder(root=str(data_path))
    samples, classes, _ = _select_balanced_subset(
        img_folder, max_classes=max_classes, k_per_class=3, total_cap=50, seed=42
    )

    class_counts = defaultdict(int)
    for _, lbl in samples:
        class_counts[classes[lbl]] += 1

    info = {
        "data_path": str(data_path),
        "total_samples": len(samples),
        "num_classes": len(classes),
        "classes": classes,
        "class_counts": dict(sorted(class_counts.items())),
    }
    return info


# ---------- quick test ----------

if __name__ == "__main__":
    print("Testing Kaggle dataset loading...")
    try:
        info = get_dataset_info()
        print(f"Dataset info: {info}")

        train_loader, val_loader, num_classes, class_info = load_kaggle_dataset(
            batch_size=2,
            num_workers=0
        )

        print("Successfully loaded dataset!")
        print(f"Train batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")
        print(f"Classes: {num_classes}")
        print(f"Class info: {class_info}")

        for batch_idx, (images, labels) in enumerate(train_loader):
            print(f"Batch shape: {images.shape} | Labels shape: {labels.shape}")
            break

    except Exception as e:
        print(f"Error: {e}")
