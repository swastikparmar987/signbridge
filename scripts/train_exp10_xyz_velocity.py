#!/usr/bin/env python3
"""Exp10: XYZ position plus normalized temporal velocity."""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from signbridge.inference.model import BiGRUArcFaceClassifier
from signbridge.preprocessing.features import build_xyz_velocity_features
from signbridge.preprocessing.normalize import normalize_sequence
from signbridge.training.config import (
    AugmentationConfig,
    CACHE_DIR,
    FULL_OUTPUT_DIR,
    get_device,
    load_full_glosses,
    set_seed,
)
from signbridge.training.dataset import create_velocity_dataloaders
from signbridge.training.evaluate import evaluate_full_model
from signbridge.training.metrics import compute_topk_accuracies
from signbridge.training.model import save_checkpoint

OUTPUT_DIR = FULL_OUTPUT_DIR / "exp10_xyz_velocity"
CHECKPOINT_PATH = OUTPUT_DIR / "best_model.pth"


def run_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = total_correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x, label=y)
        loss = criterion(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        total_correct += (logits.argmax(-1) == y).sum().item()
        total += x.size(0)
    return {"train_loss": total_loss / total, "train_accuracy": 100.0 * total_correct / total}


def validate(model, loader, criterion, device):
    model.eval()
    losses, logits_all, targets_all = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            losses.append(criterion(logits, y).item() * x.size(0))
            logits_all.append(logits.cpu())
            targets_all.append(y.cpu())
    logits = torch.cat(logits_all)
    targets = torch.cat(targets_all)
    metrics = compute_topk_accuracies(logits, targets, topk=(1, 3, 5))
    metrics["val_loss"] = sum(losses) / len(loader.dataset)
    return metrics


def validate_representation(device):
    """Fail before training if cached inputs or the model contract is wrong."""
    cache_dir = CACHE_DIR / "test"
    samples = []
    for index in range(5):
        raw = np.load(cache_dir / f"{index}.npy").astype(np.float32)
        if raw.shape != (32, 42, 3):
            raise ValueError(f"Sample {index} has shape {raw.shape}, expected (32, 42, 3)")
        normalized = normalize_sequence(raw)
        features = build_xyz_velocity_features(normalized)
        if features.shape != (32, 252):
            raise ValueError(f"Sample {index} has feature shape {features.shape}, expected (32, 252)")
        if features.dtype != np.float32 or not np.isfinite(features).all():
            raise ValueError(f"Sample {index} has invalid dtype or non-finite values")
        if not np.array_equal(features[0, 126:], np.zeros(126, dtype=np.float32)):
            raise ValueError(f"Sample {index} does not have zero first-frame velocity")
        samples.append(features)

    model = BiGRUArcFaceClassifier(
        input_size=252,
        hidden_size=256,
        num_layers=2,
        num_classes=2731,
        attention_hidden_dim=128,
        dropout_p=0.3,
        s=30.0,
        m=0.25,
    ).to(device)
    batch = torch.from_numpy(np.stack(samples)).to(device)
    with torch.no_grad():
        output = model(batch)
    if tuple(batch.shape) != (5, 32, 252) or tuple(output.shape) != (5, 2731):
        raise ValueError(f"Forward contract failed: input {tuple(batch.shape)}, output {tuple(output.shape)}")
    print("Preflight passed:")
    print(f"  raw shape: (32, 42, 3), feature shape: {tuple(samples[0].shape)}, dtype: {samples[0].dtype}")
    print(f"  first position: {samples[0][0, :6].tolist()}")
    print(f"  first velocity: {samples[0][0, 126:132].tolist()}")
    print(f"  forward: input {tuple(batch.shape)}, output {tuple(output.shape)}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    set_seed(42)
    device = get_device()
    validate_representation(device)
    class_names = load_full_glosses()
    aug = AugmentationConfig(
        enabled=True,
        noise_sigma=0.003,
        spatial_scale_range=(0.95, 1.05),
        temporal_speed_range=(0.95, 1.05),
        temporal_jitter_frames=1,
    )
    train_loader, val_loader, _, _ = create_velocity_dataloaders(
        batch_size=128, augmentation_config=aug, seed=42
    )
    model = BiGRUArcFaceClassifier(
        input_size=252,
        hidden_size=256,
        num_layers=2,
        num_classes=len(class_names),
        attention_hidden_dim=128,
        dropout_p=0.3,
        s=30.0,
        m=0.25,
    ).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=12, eta_min=2e-6)

    best_val = -1.0
    best_epoch = 0
    history = []
    started = time.time()
    for epoch in range(1, 13):
        train = run_epoch(model, train_loader, criterion, optimizer, device)
        val = validate(model, val_loader, criterion, device)
        scheduler.step()
        history.append({"epoch": epoch, **train, **val, "lr": optimizer.param_groups[0]["lr"]})
        print(
            f"Ep {epoch:02d}/12 | train {train['train_accuracy']:.2f}% | "
            f"val top1 {val['top1']:.2f}% top5 {val['top5']:.2f}%",
            flush=True,
        )
        if val["top1"] > best_val:
            best_val = val["top1"]
            best_epoch = epoch
            save_checkpoint(
                model,
                class_names,
                CHECKPOINT_PATH,
                metadata={
                    "experiment": "exp10_xyz_velocity",
                    "input_representation": "normalized XYZ plus post-normalization temporal velocity",
                    "input_shape": [32, 252],
                    "epoch": epoch,
                    "val_top1": val["top1"],
                    "val_top3": val["top3"],
                    "val_top5": val["top5"],
                    "angular_margin": 0.25,
                    "augmentation": aug.__dict__,
                },
            )

    evaluation = evaluate_full_model(CHECKPOINT_PATH, device=device, batch_size=256)
    summary = {
        "experiment": "exp10_xyz_velocity",
        "best_epoch": best_epoch,
        "best_val_top1": best_val,
        "test_top1_accuracy": evaluation["top1_accuracy"],
        "test_top3_accuracy": evaluation["top3_accuracy"],
        "test_top5_accuracy": evaluation["top5_accuracy"],
        "macro_f1": evaluation["macro_f1"],
        "weighted_f1": evaluation["weighted_f1"],
        "training_duration_seconds": round(time.time() - started, 2),
        "input_size": 252,
        "history": history,
    }
    with open(OUTPUT_DIR / "eval_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "history"}, indent=2))


if __name__ == "__main__":
    main()