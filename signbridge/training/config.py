from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import random
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SPLITS_DIR = PROJECT_ROOT / "dataset" / "ASL_Citizen" / "splits"
CACHE_DIR = PROJECT_ROOT / "landmark_cache_full"
PRETRAINED_CHECKPOINT = PROJECT_ROOT / "trained_models" / "best_gru_normalized.pth"
DEMO_GLOSSES_CSV = PROJECT_ROOT / "signbridge_final_demo_glosses.csv"
OUTPUT_DIR = PROJECT_ROOT / "trained_models" / "demo_50"
FULL_OUTPUT_DIR = PROJECT_ROOT / "trained_models" / "full_2731"


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility across random, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def get_device() -> torch.device:
    """Select the best available accelerator device (MPS, CUDA, or CPU)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def load_demo_glosses(csv_path: Optional[Path] = None) -> List[str]:
    """Load and return the alphabetically sorted list of 50 demo glosses."""
    csv_path = csv_path or DEMO_GLOSSES_CSV
    if not csv_path.exists():
        raise FileNotFoundError(f"Demo glosses CSV not found at: {csv_path}")
    df = pd.read_csv(csv_path)
    glosses = sorted(df["Gloss"].unique().tolist())
    if len(glosses) != 50:
        raise ValueError(f"Expected 50 demo glosses, found {len(glosses)}")
    return glosses


def load_full_glosses(splits_dir: Optional[Path] = None) -> List[str]:
    """Load and return the alphabetically sorted list of all 2,731 glosses."""
    splits_dir = Path(splits_dir or SPLITS_DIR)
    test_csv = splits_dir / "test.csv"
    if not test_csv.exists():
        raise FileNotFoundError(f"test.csv not found at: {test_csv}")
    df = pd.read_csv(test_csv)
    glosses = sorted(df["Gloss"].unique().tolist())
    if len(glosses) != 2731:
        raise ValueError(f"Expected 2731 glosses, found {len(glosses)}")
    return glosses


@dataclass
class ModelConfig:
    input_size: int = 126
    hidden_size: int = 256
    num_layers: int = 2
    num_classes: int = 50


@dataclass
class AugmentationConfig:
    enabled: bool = False
    noise_sigma: float = 0.003
    spatial_scale_range: Tuple[float, float] = (0.95, 1.05)
    temporal_speed_range: Tuple[float, float] = (0.95, 1.05)
    temporal_jitter_frames: int = 1


@dataclass
class ExperimentConfig:
    name: str
    description: str
    output_dir: Path
    pretrained_checkpoint: Optional[Path] = None
    batch_size: int = 32
    seed: int = 42
    # Phase 1 (Head-only if pretrained, or standard if scratch)
    phase1_epochs: int = 5
    phase1_lr: float = 1e-3
    phase1_freeze_backbone: bool = True
    # Phase 2 (Full fine-tune)
    phase2_epochs: int = 45
    phase2_lr: float = 1e-4
    phase2_freeze_backbone: bool = False
    # Regularization & Early stopping
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    patience: int = 15
    min_delta: float = 1e-4
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)


def get_experiment_configs() -> Dict[str, ExperimentConfig]:
    """Return configurations for all 3 benchmark experiments."""
    demo_dir = OUTPUT_DIR
    demo_dir.mkdir(parents=True, exist_ok=True)

    exp1 = ExperimentConfig(
        name="exp1_scratch",
        description="Scratch baseline: Fresh GRUClassifier (50 classes), no pretrained weights, no augmentation",
        output_dir=demo_dir / "exp1_scratch",
        pretrained_checkpoint=None,
        batch_size=32,
        seed=42,
        phase1_epochs=0,  # No freeze phase for scratch
        phase1_lr=1e-3,
        phase1_freeze_backbone=False,
        phase2_epochs=50,  # Full training from scratch
        phase2_lr=1e-3,
        phase2_freeze_backbone=False,
        weight_decay=1e-4,
        patience=15,
        augmentation=AugmentationConfig(enabled=False),
    )

    exp2 = ExperimentConfig(
        name="exp2_finetune",
        description="Transfer Learning: Pretrained 2731-class GRU weights, new 50-class head. Phase 1 freeze (5 epochs), Phase 2 fine-tune (lr=1e-4)",
        output_dir=demo_dir / "exp2_finetune",
        pretrained_checkpoint=PRETRAINED_CHECKPOINT,
        batch_size=32,
        seed=42,
        phase1_epochs=5,
        phase1_lr=1e-3,
        phase1_freeze_backbone=True,
        phase2_epochs=45,
        phase2_lr=1e-4,
        phase2_freeze_backbone=False,
        weight_decay=1e-4,
        patience=15,
        augmentation=AugmentationConfig(enabled=False),
    )

    exp3 = ExperimentConfig(
        name="exp3_augmented",
        description="Fine-tuning + Safe Augmentation: Pretrained GRU, safe raw landmark noise, mild speed & spatial scaling",
        output_dir=demo_dir / "exp3_augmented",
        pretrained_checkpoint=PRETRAINED_CHECKPOINT,
        batch_size=32,
        seed=42,
        phase1_epochs=5,
        phase1_lr=1e-3,
        phase1_freeze_backbone=True,
        phase2_epochs=45,
        phase2_lr=1e-4,
        phase2_freeze_backbone=False,
        weight_decay=1e-4,
        patience=15,
        augmentation=AugmentationConfig(
            enabled=True,
            noise_sigma=0.003,
            spatial_scale_range=(0.95, 1.05),
            temporal_speed_range=(0.95, 1.05),
            temporal_jitter_frames=1,
        ),
    )

    return {
        "exp1_scratch": exp1,
        "exp2_finetune": exp2,
        "exp3_augmented": exp3,
    }


def get_full_experiment_config(name: str = "exp1_augmented_finetune") -> ExperimentConfig:
    """Return configuration for full 2,731-class experiment."""
    full_dir = FULL_OUTPUT_DIR / name
    full_dir.mkdir(parents=True, exist_ok=True)

    return ExperimentConfig(
        name=name,
        description="Full 2731-Class Fine-Tuning with Safe Landmark Augmentation: Continued training from best_gru_normalized.pth",
        output_dir=full_dir,
        pretrained_checkpoint=PRETRAINED_CHECKPOINT,
        batch_size=128,
        seed=42,
        phase1_epochs=0,
        phase1_lr=1e-4,
        phase1_freeze_backbone=False,
        phase2_epochs=25,
        phase2_lr=1e-4,
        phase2_freeze_backbone=False,
        weight_decay=1e-4,
        grad_clip=1.0,
        patience=8,
        min_delta=1e-4,
        augmentation=AugmentationConfig(
            enabled=True,
            noise_sigma=0.003,
            spatial_scale_range=(0.95, 1.05),
            temporal_speed_range=(0.95, 1.05),
            temporal_jitter_frames=1,
        ),
    )

