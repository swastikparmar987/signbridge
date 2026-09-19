"""
SignBridge Training Package
Modular pipeline for training, fine-tuning, evaluating, and deploying focused ASL recognition models.
"""

from signbridge.training.config import (
    ExperimentConfig,
    ModelConfig,
    AugmentationConfig,
    get_experiment_configs,
    load_demo_glosses,
)
from signbridge.training.dataset import ASLCitizen50Dataset, create_dataloaders
from signbridge.training.augmentations import LandmarkAugmentor
from signbridge.training.model import (
    create_scratch_model,
    create_pretrained_model,
    freeze_gru,
    unfreeze_gru,
    save_checkpoint,
)
from signbridge.training.train import run_training
from signbridge.training.evaluate import evaluate_50_model, evaluate_baseline_2731_on_demo_test

__all__ = [
    "ExperimentConfig",
    "ModelConfig",
    "AugmentationConfig",
    "get_experiment_configs",
    "load_demo_glosses",
    "ASLCitizen50Dataset",
    "create_dataloaders",
    "LandmarkAugmentor",
    "create_scratch_model",
    "create_pretrained_model",
    "freeze_gru",
    "unfreeze_gru",
    "save_checkpoint",
    "run_training",
    "evaluate_50_model",
    "evaluate_baseline_2731_on_demo_test",
]
