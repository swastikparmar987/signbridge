#!/usr/bin/env python3
"""Exp8: fine-tune Exp7 with bounded webcam-domain landmark augmentation."""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from signbridge.inference.model import BiGRUArcFaceClassifier
from signbridge.training.config import (
    AugmentationConfig,
    FULL_OUTPUT_DIR,
    SPLITS_DIR,
    get_device,
    load_full_glosses,
    set_seed,
)
from signbridge.training.dataset import create_full_dataloaders
from signbridge.training.evaluate import evaluate_full_model
from signbridge.training.metrics import compute_topk_accuracies
from signbridge.training.model import save_checkpoint


BASELINE = PROJECT_ROOT / "trained_models" / "production" / "best_model.pth"
OUTPUT_DIR = FULL_OUTPUT_DIR / "exp8_domain_augmented"


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x, label=y)
        loss = criterion(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(-1) == y).sum().item()
        total += x.size(0)
    return {"train_loss": total_loss / total, "train_accuracy": 100.0 * correct / total}


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


def participant_metrics(checkpoint_path: Path, device: torch.device):
    from signbridge.inference.model import load_signbridge_model
    from signbridge.training.dataset import ASLCitizenFullDataset
    from torch.utils.data import DataLoader

    model, class_names, _ = load_signbridge_model(checkpoint_path, device=device)
    model.eval()
    dataset = ASLCitizenFullDataset("test", class_names=class_names)
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    df = dataset.dataframe
    correct = {p: [0, 0] for p in df["Participant ID"].unique()}
    offset = 0
    with torch.no_grad():
        for x, y in loader:
            predictions = model(x.to(device)).cpu().topk(5, dim=1).indices
            for row, target, top5 in zip(range(len(y)), y, predictions):
                participant = df.iloc[offset + row]["Participant ID"]
                correct[participant][0] += int(top5[0].item() == target.item())
                correct[participant][1] += int(target.item() in top5.tolist())
            offset += len(y)
    return {
        participant: {
            "samples": int(sum(1 for p in df["Participant ID"] if p == participant)),
            "top1": 100.0 * values[0] / sum(1 for p in df["Participant ID"] if p == participant),
            "top5": 100.0 * values[1] / sum(1 for p in df["Participant ID"] if p == participant),
        }
        for participant, values in correct.items()
    }


def main():
    if not BASELINE.exists():
        raise FileNotFoundError(f"Baseline checkpoint not found: {BASELINE}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    set_seed(42)
    device = get_device()
    class_names = load_full_glosses()
    aug = AugmentationConfig(
        enabled=True,
        noise_sigma=0.006,
        spatial_scale_range=(0.90, 1.10),
        temporal_speed_range=(0.75, 1.35),
        temporal_jitter_frames=2,
        rotation_degrees=8.0,
        frame_dropout_probability=0.04,
        landmark_dropout_probability=0.015,
    )
    train_loader, val_loader, _, _ = create_full_dataloaders(
        batch_size=128, augmentation_config=aug, seed=42
    )
    checkpoint = torch.load(BASELINE, map_location=device)
    model = BiGRUArcFaceClassifier(
        input_size=checkpoint["input_size"],
        hidden_size=checkpoint["hidden_size"],
        num_layers=checkpoint["num_layers"],
        num_classes=len(class_names),
        attention_hidden_dim=checkpoint.get("attention_hidden_dim", 128),
        dropout_p=checkpoint.get("dropout_p", 0.3),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=12, eta_min=2e-6)

    best_val = -1.0
    best_epoch = 0
    history = []
    start = time.time()
    for epoch in range(1, 13):
        train = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val = validate(model, val_loader, criterion, device)
        scheduler.step()
        record = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"], **train, **val}
        history.append(record)
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
                OUTPUT_DIR / "best_model.pth",
                metadata={
                    "architecture": "BiGRUArcFaceClassifier",
                    "experiment": "exp8_domain_augmented",
                    "epoch": epoch,
                    "val_top1": val["top1"],
                    "val_top3": val["top3"],
                    "val_top5": val["top5"],
                    "base_checkpoint": str(BASELINE),
                    "augmentation": aug.__dict__,
                },
            )

    best_path = OUTPUT_DIR / "best_model.pth"
    evaluation = evaluate_full_model(best_path, device=device, batch_size=256)
    summary = {
        "experiment": "exp8_domain_augmented",
        "base_checkpoint": str(BASELINE),
        "best_epoch": best_epoch,
        "best_val_top1": best_val,
        "test_top1_accuracy": evaluation["top1_accuracy"],
        "test_top3_accuracy": evaluation["top3_accuracy"],
        "test_top5_accuracy": evaluation["top5_accuracy"],
        "macro_f1": evaluation["macro_f1"],
        "weighted_f1": evaluation["weighted_f1"],
        "training_duration_seconds": round(time.time() - start, 2),
        "augmentation": aug.__dict__,
        "participant_metrics": participant_metrics(best_path, device),
        "history": history,
    }
    with open(OUTPUT_DIR / "eval_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: summary[k] for k in summary if k not in {"history", "participant_metrics"}}, indent=2))


if __name__ == "__main__":
    main()
