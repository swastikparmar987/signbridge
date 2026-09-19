from pathlib import Path
from typing import List, Dict, Optional, Callable, Tuple
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from signbridge.preprocessing.normalize import normalize_sequence
from signbridge.training.config import (
    SPLITS_DIR,
    CACHE_DIR,
    load_demo_glosses,
    load_full_glosses,
    AugmentationConfig,
)
from signbridge.training.augmentations import LandmarkAugmentor


class ASLCitizen50Dataset(Dataset):
    """
    SignBridge 50-Class Isolated ASL Dataset.
    
    Reads pre-extracted landmark caches from landmark_cache_full,
    strictly filters samples to the 50 target demo glosses,
    maps labels to 0..49 (alphabetically ordered),
    and enforces the verified normalize_sequence() pipeline.
    """

    def __init__(
        self,
        split: str,
        splits_dir: Optional[Path] = None,
        cache_dir: Optional[Path] = None,
        class_names: Optional[List[str]] = None,
        augmentor: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        is_training: bool = False,
    ):
        self.split = split.lower()
        self.splits_dir = Path(splits_dir or SPLITS_DIR)
        self.cache_dir = Path(cache_dir or CACHE_DIR)
        self.is_training = is_training
        self.augmentor = augmentor

        # Resolve split CSV path
        csv_file = self.splits_dir / f"{self.split}.csv"
        if not csv_file.exists():
            raise FileNotFoundError(f"Split CSV not found: {csv_file}")

        # Resolve cache subdirectory (handle both 'val' and 'validation')
        split_cache_name = "validation" if self.split in ["val", "validation"] else self.split
        self.split_cache_dir = self.cache_dir / split_cache_name
        if not self.split_cache_dir.exists():
            # Fallback to direct name
            self.split_cache_dir = self.cache_dir / self.split
            if not self.split_cache_dir.exists():
                raise FileNotFoundError(f"Cache dir not found: {self.split_cache_dir}")

        # Class vocabulary & mapping (sorted alphabetically 0..49)
        self.class_names = class_names or load_demo_glosses()
        self.gloss_to_idx: Dict[str, int] = {g: i for i, g in enumerate(self.class_names)}
        self.idx_to_gloss: Dict[int, str] = {i: g for i, g in enumerate(self.class_names)}

        # Load and filter dataframe
        raw_df = pd.read_csv(csv_file)
        target_set = set(self.class_names)
        
        # Keep track of original CSV row indices for cache lookup
        filtered_indices = [
            i for i, gloss in enumerate(raw_df["Gloss"])
            if gloss in target_set and (self.split_cache_dir / f"{i}.npy").exists()
        ]
        
        self.dataframe = raw_df.iloc[filtered_indices].reset_index(drop=True)
        self.cache_indices = filtered_indices

    def __len__(self) -> int:
        return len(self.cache_indices)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row_idx = self.cache_indices[index]
        cache_path = self.split_cache_dir / f"{row_idx}.npy"
        
        # 1. Load raw landmarks: shape (32, 42, 3) or (32, 126)
        raw_landmarks = np.load(cache_path).astype(np.float32)
        if raw_landmarks.ndim == 2 and raw_landmarks.shape[1] == 126:
            raw_landmarks = raw_landmarks.reshape(32, 42, 3)

        # 2. Augmentation (strictly on RAW landmarks during training only)
        if self.is_training and self.augmentor is not None:
            raw_landmarks = self.augmentor(raw_landmarks)

        # 3. Verified Normalization (single source of truth)
        normalized = normalize_sequence(raw_landmarks) # shape (32, 42, 3)

        # 4. Reshape to GRU model input (32, 126)
        features = normalized.reshape(32, 126).astype(np.float32)

        # 5. Label mapping to 0..49
        gloss = self.dataframe.iloc[index]["Gloss"]
        label = self.gloss_to_idx[gloss]

        return (
            torch.from_numpy(features),
            torch.tensor(label, dtype=torch.long),
        )


def seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)


def create_dataloaders(
    batch_size: int = 32,
    augmentation_config: Optional[AugmentationConfig] = None,
    num_workers: int = 0,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """
    Factory function to create train, val, and test DataLoaders for the 50-class task.
    """
    class_names = load_demo_glosses()

    augmentor = None
    if augmentation_config and augmentation_config.enabled:
        augmentor = LandmarkAugmentor(augmentation_config)

    train_dataset = ASLCitizen50Dataset(
        split="train",
        class_names=class_names,
        augmentor=augmentor,
        is_training=True,
    )

    val_dataset = ASLCitizen50Dataset(
        split="val",
        class_names=class_names,
        augmentor=None,
        is_training=False,
    )

    test_dataset = ASLCitizen50Dataset(
        split="test",
        class_names=class_names,
        augmentor=None,
        is_training=False,
    )

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=g,
        pin_memory=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    return train_loader, val_loader, test_loader, class_names


class ASLCitizenFullDataset(Dataset):
    """
    SignBridge Full 2,731-Class ASL Dataset.
    
    Reads pre-extracted landmark caches from landmark_cache_full,
    uses the complete 2,731 gloss vocabulary sorted alphabetically,
    and applies the verified normalize_sequence() pipeline.
    """

    def __init__(
        self,
        split: str,
        splits_dir: Optional[Path] = None,
        cache_dir: Optional[Path] = None,
        class_names: Optional[List[str]] = None,
        augmentor: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        is_training: bool = False,
    ):
        self.split = split.lower()
        self.splits_dir = Path(splits_dir or SPLITS_DIR)
        self.cache_dir = Path(cache_dir or CACHE_DIR)
        self.is_training = is_training
        self.augmentor = augmentor

        csv_file = self.splits_dir / f"{self.split}.csv"
        if not csv_file.exists():
            raise FileNotFoundError(f"Split CSV not found: {csv_file}")

        split_cache_name = "validation" if self.split in ["val", "validation"] else self.split
        self.split_cache_dir = self.cache_dir / split_cache_name
        if not self.split_cache_dir.exists():
            self.split_cache_dir = self.cache_dir / self.split
            if not self.split_cache_dir.exists():
                raise FileNotFoundError(f"Cache dir not found: {self.split_cache_dir}")

        self.class_names = class_names or load_full_glosses(self.splits_dir)
        self.gloss_to_idx: Dict[str, int] = {g: i for i, g in enumerate(self.class_names)}
        self.dataframe = pd.read_csv(csv_file)

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        cache_path = self.split_cache_dir / f"{index}.npy"
        raw_landmarks = np.load(cache_path).astype(np.float32)
        if raw_landmarks.ndim == 2 and raw_landmarks.shape[1] == 126:
            raw_landmarks = raw_landmarks.reshape(32, 42, 3)

        if self.is_training and self.augmentor is not None:
            raw_landmarks = self.augmentor(raw_landmarks)

        normalized = normalize_sequence(raw_landmarks)
        features = normalized.reshape(32, 126).astype(np.float32)

        gloss = self.dataframe.iloc[index]["Gloss"]
        label = self.gloss_to_idx[gloss]

        return (
            torch.from_numpy(features),
            torch.tensor(label, dtype=torch.long),
        )


class ASLCitizenFullDatasetVelocity(Dataset):
    """
    SignBridge Full 2,731-Class ASL Dataset with Position + Velocity features.
    
    Reads pre-extracted landmark caches from landmark_cache_full,
    uses the complete 2,731 gloss vocabulary sorted alphabetically,
    computes velocity features (frame-to-frame differences),
    and applies the verified normalize_sequence() pipeline.
    """

    def __init__(
        self,
        split: str,
        splits_dir: Optional[Path] = None,
        cache_dir: Optional[Path] = None,
        class_names: Optional[List[str]] = None,
        augmentor: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        is_training: bool = False,
    ):
        from signbridge.training.config import SPLITS_DIR, CACHE_DIR, load_full_glosses
        self.split = split.lower()
        self.splits_dir = Path(splits_dir or SPLITS_DIR)
        self.cache_dir = Path(cache_dir or CACHE_DIR)
        self.is_training = is_training
        self.augmentor = augmentor

        csv_file = self.splits_dir / f"{self.split}.csv"
        if not csv_file.exists():
            raise FileNotFoundError(f"Split CSV not found: {csv_file}")

        split_cache_name = "validation" if self.split in ["val", "validation"] else self.split
        self.split_cache_dir = self.cache_dir / split_cache_name
        if not self.split_cache_dir.exists():
            self.split_cache_dir = self.cache_dir / self.split
            if not self.split_cache_dir.exists():
                raise FileNotFoundError(f"Cache dir not found: {self.split_cache_dir}")

        self.class_names = class_names or load_full_glosses(self.splits_dir)
        self.gloss_to_idx: Dict[str, int] = {g: i for i, g in enumerate(self.class_names)}
        self.dataframe = pd.read_csv(csv_file)

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        cache_path = self.split_cache_dir / f"{index}.npy"
        raw_landmarks = np.load(cache_path).astype(np.float32)
        if raw_landmarks.ndim == 2 and raw_landmarks.shape[1] == 126:
            raw_landmarks = raw_landmarks.reshape(32, 42, 3)

        if self.is_training and self.augmentor is not None:
            raw_landmarks = self.augmentor(raw_landmarks)

        normalized = normalize_sequence(raw_landmarks)  # [32, 42, 3]
        normalized_flat = normalized.reshape(32, 126)  # [32, 126]
        
        # Compute velocity: frame-to-frame difference
        velocity = np.diff(normalized_flat, axis=0, prepend=normalized_flat[:1])  # [32, 126]
        
        # Concatenate position + velocity
        features = np.concatenate([normalized_flat, velocity], axis=1).astype(np.float32)  # [32, 252]

        gloss = self.dataframe.iloc[index]["Gloss"]
        label = self.gloss_to_idx[gloss]

        return (
            torch.from_numpy(features),
            torch.tensor(label, dtype=torch.long),
        )


def create_velocity_dataloaders(
    batch_size: int = 128,
    augmentation_config: Optional[AugmentationConfig] = None,
    num_workers: int = 0,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """
    Factory function to create train, val, and test DataLoaders for the full 2,731-class task with velocity features.
    """
    class_names = load_full_glosses()

    augmentor = None
    if augmentation_config and augmentation_config.enabled:
        augmentor = LandmarkAugmentor(augmentation_config)

    train_dataset = ASLCitizenFullDatasetVelocity(
        split="train",
        class_names=class_names,
        augmentor=augmentor,
        is_training=True,
    )

    val_dataset = ASLCitizenFullDatasetVelocity(
        split="val",
        class_names=class_names,
        augmentor=None,
        is_training=False,
    )

    test_dataset = ASLCitizenFullDatasetVelocity(
        split="test",
        class_names=class_names,
        augmentor=None,
        is_training=False,
    )

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=g,
        pin_memory=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    return train_loader, val_loader, test_loader, class_names


def create_full_dataloaders(
    batch_size: int = 32,
    augmentation_config: Optional[AugmentationConfig] = None,
    num_workers: int = 0,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """
    Factory function to create train, val, and test DataLoaders for the full 2,731-class task.
    """
    class_names = load_full_glosses()

    augmentor = None
    if augmentation_config and augmentation_config.enabled:
        augmentor = LandmarkAugmentor(augmentation_config)

    train_dataset = ASLCitizenFullDataset(
        split="train",
        class_names=class_names,
        augmentor=augmentor,
        is_training=True,
    )

    val_dataset = ASLCitizenFullDataset(
        split="val",
        class_names=class_names,
        augmentor=None,
        is_training=False,
    )

    test_dataset = ASLCitizenFullDataset(
        split="test",
        class_names=class_names,
        augmentor=None,
        is_training=False,
    )

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=g,
        pin_memory=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    return train_loader, val_loader, test_loader, class_names