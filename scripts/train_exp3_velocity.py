#!/usr/bin/env python3
"""
Experiment 3: Position + Velocity Features
Adds explicit motion information to the model.
Input: Position [126] + Velocity [126] = 252 features
Architecture: BiGRU + Attention + Mean + Max Pooling (same as Exp 2)
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, List
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from signbridge.inference.model import BiGRUAttentionPoolingClassifier, get_default_device
from signbridge.training.config import (
    AugmentationConfig,
    FULL_OUTPUT_DIR,
    load_full_glosses,
    set_seed,
    get_device,
)
from signbridge.training.dataset import create_full_dataloaders
from signbridge.training.model import save_checkpoint
from signbridge.training.metrics import compute_topk_accuracies, compute_comprehensive_metrics
from signbridge.training.evaluate import evaluate_full_model
from signbridge.preprocessing.normalize import normalize_sequence


class ASLCitizenFullDatasetVelocity(torch.utils.data.Dataset):
    """
    Full 2,731-Class ASL Dataset with Position + Velocity features.
    """
    
    def __init__(
        self,
        split: str,
        splits_dir: Path = None,
        cache_dir: Path = None,
        class_names: list = None,
        augmentor=None,
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
        self.gloss_to_idx: dict = {g: i for i, g in enumerate(self.class_names)}
        self.dataframe = pd.read_csv(csv_file)

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int):
        import pandas as pd
        cache_path = self.split_cache_dir / f"{index}.npy"
        raw_landmarks = np.load(cache_path).astype(np.float32)
        if raw_landmarks.ndim == 2 and raw_landmarks.shape[1] == 126:
            raw_landmarks = raw_landmarks.reshape(32, 42, 3)

        if self.is_training and self.augmentor is not None:
            raw_landmarks = self.augmentor(raw_landmarks)

        normalized = normalize_sequence(raw_landmarks)  # [32, 42, 3]
        normalized_flat = normalized.reshape(32, 126)  # [32, 126]
        
        # Compute velocity: frame-to-frame difference on flattened features
        velocity = np.diff(normalized_flat, axis=0, prepend=normalized_flat[:1])  # [32, 126]
        
        # Concatenate position + velocity
        features = np.concatenate([normalized_flat, velocity], axis=1).astype(np.float32)  # [32, 252]

        gloss = self.dataframe.iloc[index]["Gloss"]
        label = self.gloss_to_idx[gloss]

        return torch.from_numpy(features), torch.tensor(label, dtype=torch.long)


import pandas as pd
import numpy as np


def create_velocity_dataloaders(
    batch_size: int = 128,
    augmentation_config: AugmentationConfig = None,
    num_workers: int = 0,
    seed: int = 42,
) -> tuple:
    class_names = load_full_glosses()

    augmentor = None
    if augmentation_config and augmentation_config.enabled:
        from signbridge.training.augmentations import LandmarkAugmentor
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
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, worker_init_fn=lambda w: np.random.seed(torch.initial_seed() % 2**32),
        generator=g, pin_memory=False,
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False)

    return train_loader, val_loader, test_loader, class_names


def train_one_epoch(model, dataloader, criterion, optimizer, device, grad_clip=1.0):
    model.train()
    total_loss = 0.0
    correct = 0
    total_samples = 0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(dim=-1) == y).sum().item()
        total_samples += x.size(0)
    return {"train_loss": total_loss / total_samples, "train_accuracy": 100.0 * correct / total_samples}


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_logits = []
    all_targets = []
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0)
            all_logits.append(logits.cpu())
            all_targets.append(y.cpu())
    total_samples = len(dataloader.dataset)
    all_logits = torch.cat(all_logits, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    topk_res = compute_topk_accuracies(all_logits, all_targets, topk=(1, 3, 5))
    return {"val_loss": total_loss / total_samples, "val_top1": topk_res["top1"], "val_top3": topk_res["top3"], "val_top5": topk_res["top5"]}


def main():
    exp_name = "exp3_velocity"
    output_dir = FULL_OUTPUT_DIR / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print(f"EXPERIMENT 3: POSITION + VELOCITY FEATURES")
    print("=" * 75)

    batch_size = 128
    learning_rate = 1e-4
    weight_decay = 1e-4
    grad_clip = 1.0
    max_epochs = 35
    patience = 10
    min_delta = 1e-4
    seed = 42

    set_seed(seed)
    device = get_device()
    print(f"Device: {device}")

    aug_config = AugmentationConfig(
        enabled=True, noise_sigma=0.003,
        spatial_scale_range=(0.95, 1.05),
        temporal_speed_range=(0.95, 1.05),
        temporal_jitter_frames=1,
    )

    train_loader, val_loader, test_loader, class_names = create_velocity_dataloaders(
        batch_size=batch_size, augmentation_config=aug_config, seed=seed
    )
    print(f"Dataset: Train={len(train_loader.dataset)}, Val={len(val_loader.dataset)}, Test={len(test_loader.dataset)}")
    print(f"Classes: {len(class_names)}")

    # Build model with input_size=252 (126 position + 126 velocity)
    model = BiGRUAttentionPoolingClassifier(
        input_size=252, hidden_size=256, num_layers=2,
        num_classes=len(class_names), attention_hidden_dim=128, dropout_p=0.3
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: BiGRUAttentionPoolingClassifier, Params: {total_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    history = []
    best_val_top1 = -1.0
    best_epoch = -1
    epochs_no_improve = 0
    best_checkpoint_path = output_dir / "best_model.pth"

    print("\n--- TRAINING ---")
    for epoch in range(1, max_epochs + 1):
        t0 = time.time()
        current_lr = optimizer.param_groups[0]["lr"]

        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device, grad_clip)
        val_metrics = validate(model, val_loader, criterion, device)
        scheduler.step(val_metrics["val_top1"])
        elapsed = time.time() - t0

        epoch_record = {"epoch": epoch, "lr": current_lr, **train_metrics, **val_metrics, "elapsed_sec": round(elapsed, 2)}
        history.append(epoch_record)

        is_best = val_metrics["val_top1"] > best_val_top1 + min_delta
        if is_best:
            best_val_top1 = val_metrics["val_top1"]
            best_epoch = epoch
            epochs_no_improve = 0
            save_checkpoint(
                model, class_names, best_checkpoint_path,
                metadata={
                    "architecture": "BiGRUAttentionPoolingClassifier",
                    "experiment": exp_name, "epoch": epoch,
                    "val_top1": best_val_top1, "val_top5": val_metrics["val_top5"],
                    "total_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
                    "input_size": 252,
                }
            )
        else:
            epochs_no_improve += 1

        print(f"  Ep {epoch:2d}/{max_epochs} | "
              f"Train Loss: {train_metrics['train_loss']:.4f} Acc: {train_metrics['train_accuracy']:.2f}% | "
              f"Val Loss: {val_metrics['val_loss']:.4f} Top-1: {val_metrics['val_top1']:.2f}% Top-5: {val_metrics['val_top5']:.2f}% | "
              f"lr: {current_lr:.1e} ({elapsed:.1f}s){' 🌟 [BEST]' if is_best else ''}", flush=True)

        if epochs_no_improve >= patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break

    total_train_time = sum(h["elapsed_sec"] for h in history)
    print(f"\nTraining complete in {total_train_time/60:.1f} min. Best Val Top-1: {best_val_top1:.2f}% at epoch {best_epoch}")

    # Save history
    with open(output_dir / "training_history.json", "w") as f:
        json.dump({"experiment": exp_name, "best_epoch": best_epoch, "best_val_top1": best_val_top1, "history": history}, f, indent=2)

    # Evaluate on test set
    print("\n--- TEST EVALUATION ---")
    from signbridge.training.evaluate import evaluate_full_model
    eval_result = evaluate_full_model(best_checkpoint_path, batch_size=256)

    eval_summary = {
        "experiment": exp_name, "architecture": "BiGRUAttentionPoolingClassifier",
        "input_size": 252, "total_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "checkpoint_path": str(best_checkpoint_path), "best_epoch": best_epoch,
        "best_val_top1": best_val_top1,
        "test_top1_accuracy": eval_result["top1_accuracy"],
        "test_top3_accuracy": eval_result["top3_accuracy"],
        "test_top5_accuracy": eval_result["top5_accuracy"],
        "macro_f1": eval_result["macro_f1"], "weighted_f1": eval_result["weighted_f1"],
        "total_test_samples": eval_result["total_samples"],
        "top_confusions": eval_result["top_confusions"],
        "worst_classes": eval_result["worst_classes"], "best_classes": eval_result["best_classes"],
    }

    with open(output_dir / "eval_results.json", "w") as f:
        json.dump(eval_summary, f, indent=2)

    with open(output_dir / "config.json", "w") as f:
        json.dump({"name": exp_name, "input_size": 252, "architecture": "BiGRUAttentionPoolingClassifier",
                   "batch_size": batch_size, "lr": learning_rate, "weight_decay": weight_decay,
                   "max_epochs": max_epochs, "patience": patience, "seed": seed}, f, indent=2)

    print("\n" + "=" * 75)
    print("EXP 3 (VELOCITY) RESULTS:")
    print(f"  Test Top-1: {eval_result['top1_accuracy']:.2f}%")
    print(f"  Test Top-3: {eval_result['top3_accuracy']:.2f}%")
    print(f"  Test Top-5: {eval_result['top5_accuracy']:.2f}%")
    print(f"  Macro F1:   {eval_result['macro_f1']:.2f}%")
    print(f"  Weighted F1: {eval_result['weighted_f1']:.2f}%")
    print("=" * 75)

    return eval_summary


if __name__ == "__main__":
    main()