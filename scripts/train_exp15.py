import os
import sys
import time
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

PROJECT_ROOT = Path("/Users/pro/Desktop/projects/signbridge")
sys.path.insert(0, str(PROJECT_ROOT))

from signbridge.inference.model import Exp11AdapterClassifier, BiGRUArcFaceClassifier
from signbridge.training.dataset import create_exp15_dataloaders
from signbridge.training.config import load_full_glosses

EXP14C_CKPT_PATH = PROJECT_ROOT / "trained_models" / "full_2731" / "exp14c_combined" / "best_model.pth"
EXP15_OUTPUT_DIR = PROJECT_ROOT / "trained_models" / "full_2731" / "exp15_body_aware"
EXP15_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def verify_and_init_exp15_model(device: torch.device) -> Exp11AdapterClassifier:
    print("\n=== STAGE 0: ZERO-DRIFT INITIALIZATION & VERIFICATION ===", flush=True)
    if not EXP14C_CKPT_PATH.exists():
        raise FileNotFoundError(f"EXP14-C checkpoint missing at {EXP14C_CKPT_PATH}")

    print(f"Loading EXP14-C weights from {EXP14C_CKPT_PATH}...", flush=True)
    exp14c_ckpt = torch.load(EXP14C_CKPT_PATH, map_location=device, weights_only=False)

    # 1. Instantiate baseline EXP14-C model (261D)
    backbone_14c = BiGRUArcFaceClassifier(
        input_size=126,
        hidden_size=256,
        num_layers=2,
        num_classes=2731,
        dropout_p=0.3
    ).to(device)
    model_14c = Exp11AdapterClassifier(backbone=backbone_14c, input_size=261, adapter_output_size=126).to(device)
    model_14c.load_state_dict(exp14c_ckpt["model_state_dict"])
    model_14c.eval()

    # 2. Instantiate EXP15 model (292D)
    backbone_15 = BiGRUArcFaceClassifier(
        input_size=126,
        hidden_size=256,
        num_layers=2,
        num_classes=2731,
        dropout_p=0.3
    ).to(device)
    model_15 = Exp11AdapterClassifier(backbone=backbone_15, input_size=292, adapter_output_size=126).to(device)

    # Copy backbone weights directly
    backbone_weights = {k.replace("backbone.", ""): v for k, v in exp14c_ckpt["model_state_dict"].items() if k.startswith("backbone.")}
    model_15.backbone.load_state_dict(backbone_weights)

    # Initialize adapter weights: W[:, :261] = EXP14-C adapter, W[:, 261:] = 0, bias = EXP14-C bias
    old_w = exp14c_ckpt["model_state_dict"]["adapter.weight"]
    old_b = exp14c_ckpt["model_state_dict"]["adapter.bias"]

    with torch.no_grad():
        model_15.adapter.weight.zero_()
        model_15.adapter.weight[:, :261].copy_(old_w)
        model_15.adapter.bias.copy_(old_b)

    model_15.eval()

    # 3. Perform Numerical Zero-Drift Assertion
    torch.manual_seed(42)
    dummy_261 = torch.randn(16, 32, 261, device=device)
    dummy_body_zero = torch.zeros(16, 32, 31, device=device)
    dummy_292_zero = torch.cat([dummy_261, dummy_body_zero], dim=-1)

    with torch.no_grad():
        logits_14c = model_14c(dummy_261)
        logits_15_zero = model_15(dummy_292_zero)

    diff = torch.max(torch.abs(logits_14c - logits_15_zero)).item()
    print(f"Zero-Drift Logit Difference: {diff:.10e}")
    assert diff < 1e-5, f"Zero-drift check failed! Max diff: {diff}"
    print("Zero-Drift Assertion PASSED (diff < 1e-5).\n", flush=True)

    return model_15

def evaluate_model(model: nn.Module, data_loader, device: torch.device):
    model.eval()
    correct_top1 = 0
    correct_top5 = 0
    total_samples = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for features, labels in data_loader:
            features = features.to(device)
            labels = labels.to(device)

            logits = model(features)
            
            # Top-1
            _, pred_top1 = torch.max(logits, dim=1)
            correct_top1 += (pred_top1 == labels).sum().item()

            # Top-5
            _, pred_top5 = torch.topk(logits, k=5, dim=1)
            correct_top5 += torch.sum(pred_top5 == labels.unsqueeze(1)).item()

            total_samples += labels.size(0)
            all_preds.extend(pred_top1.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    top1_acc = (correct_top1 / total_samples) * 100.0
    top5_acc = (correct_top5 / total_samples) * 100.0

    return top1_acc, top5_acc, np.array(all_preds), np.array(all_labels)

def train_exp15():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    # 1. Zero-drift Initialization
    model = verify_and_init_exp15_model(device)

    # 2. Data Loaders
    print("Loading EXP15 DataLoaders (batch_size=128)...", flush=True)
    train_loader, val_loader, test_loader, class_names = create_exp15_dataloaders(
        batch_size=128,
        num_workers=4,
        seed=42,
    )
    print(f"Train samples: {len(train_loader.dataset)} | Val samples: {len(val_loader.dataset)} | Test samples: {len(test_loader.dataset)}", flush=True)

    criterion = nn.CrossEntropyLoss()

    # -------------------------------------------------------------
    # STAGE 1: Adapter Warmup (5 Epochs, Freeze Backbone)
    # -------------------------------------------------------------
    print("\n=== STAGE 1: ADAPTER WARMUP (5 Epochs) ===", flush=True)
    for p in model.backbone.parameters():
        p.requires_grad = False
    for p in model.adapter.parameters():
        p.requires_grad = True

    optimizer_s1 = AdamW(model.adapter.parameters(), lr=1e-3, weight_decay=1e-4)

    best_val_top1 = 0.0

    for epoch in range(1, 6):
        model.train()
        model.backbone.eval()  # keeps dropout disabled in backbone
        running_loss = 0.0

        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)

            optimizer_s1.zero_grad()
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer_s1.step()

            running_loss += loss.item() * features.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        val_top1, val_top5, _, _ = evaluate_model(model, val_loader, device)

        print(f"[Stage 1 - Epoch {epoch}/5] Loss: {epoch_loss:.4f} | Val Top-1: {val_top1:.2f}% | Val Top-5: {val_top5:.2f}%", flush=True)

        if val_top1 > best_val_top1:
            best_val_top1 = val_top1
            torch.save({"model_state_dict": model.state_dict()}, EXP15_OUTPUT_DIR / "best_model.pth")

    # -------------------------------------------------------------
    # STAGE 2: Differential LR Fine-Tuning (25 Epochs)
    # -------------------------------------------------------------
    print("\n=== STAGE 2: DIFFERENTIAL LR FINE-TUNING (25 Epochs) ===", flush=True)
    for p in model.parameters():
        p.requires_grad = True

    param_groups = [
        {"params": model.adapter.parameters(), "lr": 5e-4},
        {"params": model.backbone.gru.parameters(), "lr": 1e-5},
        {"params": [p for n, p in model.backbone.named_parameters() if not n.startswith("gru.")], "lr": 1e-4},
    ]

    optimizer_s2 = AdamW(param_groups, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer_s2, T_max=25, eta_min=1e-6)

    for epoch in range(1, 26):
        model.train()
        running_loss = 0.0

        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)

            optimizer_s2.zero_grad()
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer_s2.step()

            running_loss += loss.item() * features.size(0)

        scheduler.step()
        epoch_loss = running_loss / len(train_loader.dataset)
        val_top1, val_top5, _, _ = evaluate_model(model, val_loader, device)

        print(f"[Stage 2 - Epoch {epoch}/25] Loss: {epoch_loss:.4f} | Val Top-1: {val_top1:.2f}% | Val Top-5: {val_top5:.2f}%", flush=True)

        if val_top1 > best_val_top1:
            best_val_top1 = val_top1
            torch.save({"model_state_dict": model.state_dict()}, EXP15_OUTPUT_DIR / "best_model.pth")
            print(f"  --> Saved new best checkpoint with Val Top-1: {val_top1:.2f}%")

    print(f"\nTraining Complete! Best Val Top-1: {best_val_top1:.2f}%", flush=True)

    # -------------------------------------------------------------
    # FINAL EVALUATION ON UNTOUCHED TEST SET (32,941 samples)
    # -------------------------------------------------------------
    print("\n=== FINAL TEST EVALUATION ON 32,941 SAMPLES ===", flush=True)
    best_ckpt = torch.load(EXP15_OUTPUT_DIR / "best_model.pth", map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])

    test_top1, test_top5, test_preds, test_labels = evaluate_model(model, test_loader, device)
    print(f"EXP15 TEST TOP-1: {test_top1:.2f}%")
    print(f"EXP15 TEST TOP-5: {test_top5:.2f}%")

    # Save predictions
    np.save(EXP15_OUTPUT_DIR / "test_preds.npy", test_preds)
    np.save(EXP15_OUTPUT_DIR / "test_labels.npy", test_labels)

if __name__ == "__main__":
    train_exp15()
