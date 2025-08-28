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
    
    def __init__(self, data_dir: str, transform=None, max_classes: int = 20, 
                 k_per_class: int = 3, total_cap: int = 50, seed: int = 42):
        self.data_dir = Path(data_dir)
        self.transform = transform or self._get_default_transform()
        
        # Load dataset using SafeImageFolder
        self.img_folder = SafeImageFolder(
            root=str(self.data_dir),
            transform=self.transform
        )
        
        # Select balanced subset
        self.samples, self.classes, self.old2new = _select_balanced_subset(
            self.img_folder, max_classes, k_per_class, total_cap, seed
        )
        
        logger.info(f"Loaded {len(self.samples)} samples from {len(self.classes)} classes")
        logger.info(f"Classes: {self.classes}")
    
    def _get_default_transform(self):
        """Default transform for preprocessing"""
        return transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label, dtype=torch.long)


# ---------- data loaders ----------

def create_kaggle_data_loaders(
    data_dir: str,
    batch_size: int = 32,
    train_split: float = 0.8,
    val_split: float = 0.1,
    test_split: float = 0.1,
    max_classes: int = 20,
    k_per_class: int = 3,
    total_cap: int = 50,
    seed: int = 42,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test data loaders for Kaggle dataset.
    
    Args:
        data_dir: Path to Kaggle dataset directory
        batch_size: Batch size for data loaders
        train_split: Fraction of data for training
        val_split: Fraction of data for validation
        test_split: Fraction of data for testing
        max_classes: Maximum number of classes to use
        k_per_class: Number of samples per class
        total_cap: Total number of samples to use
        seed: Random seed for reproducibility
        num_workers: Number of workers for data loading
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    
    # Create dataset
    dataset = KaggleMathDataset(
        data_dir=data_dir,
        max_classes=max_classes,
        k_per_class=k_per_class,
        total_cap=total_cap,
        seed=seed
    )
    
    # Calculate split sizes
    total_size = len(dataset)
    train_size = int(train_split * total_size)
    val_size = int(val_split * total_size)
    test_size = total_size - train_size - val_size
    
    # Split dataset
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed)
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    logger.info(f"Created data loaders:")
    logger.info(f"  Train: {len(train_loader)} batches ({len(train_dataset)} samples)")
    logger.info(f"  Val: {len(val_loader)} batches ({len(val_dataset)} samples)")
    logger.info(f"  Test: {len(test_loader)} batches ({len(test_dataset)} samples)")
    
    return train_loader, val_loader, test_loader


def create_dummy_data_loaders(
    batch_size: int = 32,
    num_samples: int = 100,
    num_classes: int = 20,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create dummy data loaders for testing purposes.
    
    Args:
        batch_size: Batch size for data loaders
        num_samples: Number of dummy samples to create
        num_classes: Number of classes
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    
    class DummyDataset(Dataset):
        def __init__(self, num_samples, num_classes, seed):
            self.num_samples = num_samples
            self.num_classes = num_classes
            torch.manual_seed(seed)
            
        def __len__(self):
            return self.num_samples
        
        def __getitem__(self, idx):
            # Create dummy image (1 channel, 128x128)
            image = torch.randn(1, 128, 128)
            # Create dummy label
            label = torch.randint(0, self.num_classes, (1,))
            return image, label
    
    # Create datasets
    train_size = int(0.8 * num_samples)
    val_size = int(0.1 * num_samples)
    test_size = num_samples - train_size - val_size
    
    train_dataset = DummyDataset(train_size, num_classes, seed)
    val_dataset = DummyDataset(val_size, num_classes, seed)
    test_dataset = DummyDataset(test_size, num_classes, seed)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    logger.info(f"Created dummy data loaders:")
    logger.info(f"  Train: {len(train_loader)} batches ({len(train_dataset)} samples)")
    logger.info(f"  Val: {len(val_loader)} batches ({len(val_dataset)} samples)")
    logger.info(f"  Test: {len(test_loader)} batches ({len(test_dataset)} samples)")
    
    return train_loader, val_loader, test_loader
