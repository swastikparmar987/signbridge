import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Tuple, List, Dict, Any
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from signbridge.inference.model import load_signbridge_model, Exp11AdapterClassifier, BiGRUArcFaceClassifier
from signbridge.training.config import (
    SPLITS_DIR,
    CACHE_DIR,
    FULL_OUTPUT_DIR,
    load_full_glosses,
    set_seed,
    get_device,
    AugmentationConfig,
)
from signbridge.training.dataset import create_exp14_dataloaders
from sklearn.metrics import f1_score

PRETRAINED_CHECKPOINT = PROJECT_ROOT / "trained_models" / "production" / "best_model.pth"

EXP13_BASELINE_TOP1 = 51.41
EXP13_BASELINE_TOP5 = 76.33
EXP13_BASELINE_MACRO_F1 = 50.26


def zero_drift_sanity_check(exp13_model: nn.Module, exp14_model: nn.Module, device: torch.device, input_size: int) -> float:
    """
    Verify zero-drift initialization: max absolute difference between EXP13 output
    and EXP14 output (with new features zeroed out) must be < 1e-6.
    """
    exp13_model.eval()
    exp14_model.eval()

    # Create dummy batch of 16 sequences
    dummy_252 = torch.randn(16, 32, 252, device=device)
    dummy_exp14 = torch.zeros(16, 32, input_size, device=device)
    dummy_exp14[:, :, :252] = dummy_252

    with torch.no_grad():
        out13 = exp13_model(dummy_252)
        out14 = exp14_model(dummy_exp14)

    max_diff = torch.max(torch.abs(out13 - out14)).item()
    print(f"Zero-Drift Sanity Check Max Abs Difference: {max_diff:.10e}", flush=True)
    return max_diff


def train_epoch_stage1(model: nn.Module, dataloader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device) -> float:
    model.train()
    # Ensure backbone stays in eval mode during freeze phase
    model.backbone.eval()
    total_loss = 0.0
    num_batches = 0

    for features, labels in dataloader:
        features = features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(features, label=labels)
        loss = nn.functional.cross_entropy(logits, labels)

        loss.backward()
        nn.utils.clip_grad_norm_(model.adapter.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(1, num_batches)


def train_epoch_stage2(model: nn.Module, dataloader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device) -> float:
    model.train()
    total_loss = 0.0
    num_batches = 0

    for features, labels in dataloader:
        features = features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(features, label=labels)
        loss = nn.functional.cross_entropy(logits, labels)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(1, num_batches)


def evaluate_model(model: nn.Module, dataloader: DataLoader, device: torch.device) -> Tuple[float, float, float, float]:
    model.eval()
    all_preds = []
    all_targets = []
    all_probabilities = []

    with torch.no_grad():
        for features, labels in dataloader:
            features = features.to(device)
            logits = model(features)
            probs = torch.softmax(logits, dim=1).cpu().numpy()

            all_probabilities.append(probs)
            all_targets.append(labels.numpy())

    all_probabilities = np.vstack(all_probabilities)
    all_targets = np.concatenate(all_targets)
    all_preds = np.argmax(all_probabilities, axis=1)

    top1 = np.mean(all_preds == all_targets) * 100.0

    # Top-3 & Top-5
    top3_count = 0
    top5_count = 0
    num_samples = len(all_targets)
    for i in range(num_samples):
        top_k = np.argsort(all_probabilities[i])[-5:]
        if all_targets[i] in top_k[-3:]:
            top3_count += 1
        if all_targets[i] in top_k:
            top5_count += 1

    top3 = (top3_count / num_samples) * 100.0
    top5 = (top5_count / num_samples) * 100.0

    macro_f1 = float(f1_score(all_targets, all_preds, average="macro", zero_division=0) * 100.0)

    return top1, top3, top5, macro_f1


def run_exp14(exp_code: str, eval_only: bool = False):
    set_seed(42)
    device = get_device()

    exp_code = exp_code.lower()
    if exp_code in ["exp14a", "a", "14a"]:
        feature_mode = "a"
        input_size = 255
        exp_name = "exp14a_interhand"
        description = "EXP14-A: 252D baseline + 3D palm-scale normalized inter-hand displacement (255D)"
    elif exp_code in ["exp14b", "b", "14b"]:
        feature_mode = "b"
        input_size = 258
        exp_name = "exp14b_palm"
        description = "EXP14-B: 252D baseline + 6D unit palm normals (258D)"
    elif exp_code in ["exp14c", "c", "14c"]:
        feature_mode = "c"
        input_size = 261
        exp_name = "exp14c_combined"
        description = "EXP14-C: 252D baseline + 3D inter-hand displacement + 6D palm normals (261D)"
    else:
        raise ValueError(f"Unknown experiment code: {exp_code}")

    output_dir = FULL_OUTPUT_DIR / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = output_dir / "best_model.pth"

    print(f"\n==================================================", flush=True)
    print(f"  RUNNING {exp_name.upper()}", flush=True)
    print(f"  Description: {description}", flush=True)
    print(f"  Input Dim: {input_size}D", flush=True)
    print(f"  Output Dir: {output_dir}", flush=True)
    print(f"==================================================\n", flush=True)

    # 1. Load Pretrained EXP13 Model
    print(f"Loading pretrained EXP13 checkpoint from: {PRETRAINED_CHECKPOINT}", flush=True)
    exp13_model, class_names, metadata = load_signbridge_model(PRETRAINED_CHECKPOINT, device=device)

    # 2. Build EXP14 Model
    backbone = exp13_model.backbone

    exp14_model = Exp11AdapterClassifier(
        backbone=backbone,
        input_size=input_size,
        adapter_output_size=126,
    ).to(device)

    # Initialize EXP14 adapter from EXP13 adapter weights
    with torch.no_grad():
        exp14_model.adapter.weight.zero_()
        exp14_model.adapter.weight[:, :252].copy_(exp13_model.adapter.weight)
        exp14_model.adapter.weight[:, 252:].zero_()
        exp14_model.adapter.bias.copy_(exp13_model.adapter.bias)

    # 3. Perform ZERO-DRIFT SANITY CHECK
    max_drift = zero_drift_sanity_check(exp13_model, exp14_model, device, input_size)
    if max_drift >= 1e-6:
        raise RuntimeError(f"CRITICAL ERROR: Zero-drift check failed with max_diff = {max_drift:.10e} >= 1e-6! Aborting training.")
    print(f"PASSED Zero-Drift Sanity Check! (max_diff = {max_drift:.10e})\n", flush=True)

    # 4. Create DataLoaders
    print("Creating EXP14 DataLoaders...", flush=True)
    aug_config = AugmentationConfig(
        enabled=True,
        noise_sigma=0.003,
        spatial_scale_range=(0.95, 1.05),
        temporal_speed_range=(0.95, 1.05),
        temporal_jitter_frames=1,
    )
    train_loader, val_loader, test_loader, class_names = create_exp14_dataloaders(
        feature_mode=feature_mode,
        batch_size=128,
        augmentation_config=aug_config,
        use_skeleton_warp=True,
        num_workers=0,
        seed=42,
    )
    print(f"DataLoaders created: Train batches={len(train_loader)}, Val batches={len(val_loader)}, Test batches={len(test_loader)}\n", flush=True)

    start_time = time.time()
    best_val_top1 = 0.0

    if eval_only and best_model_path.exists():
        print(f"Eval-only flag set. Skipping training and evaluating checkpoint: {best_model_path}", flush=True)
        training_time = 0.0
        checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
        best_val_top1 = checkpoint.get("val_top1", 0.0)
    else:
        # 5. STAGE 1: Freeze Backbone, Train Adapter Only (5 Epochs)
        print("--- STAGE 1: Training Adapter Only (5 Epochs, Backbone Frozen) ---", flush=True)
        for p in exp14_model.backbone.parameters():
            p.requires_grad = False
        for p in exp14_model.adapter.parameters():
            p.requires_grad = True

        opt_stage1 = torch.optim.AdamW(exp14_model.adapter.parameters(), lr=1e-3, weight_decay=1e-4)

        for epoch in range(1, 6):
            loss = train_epoch_stage1(exp14_model, train_loader, opt_stage1, device)
            val_top1, val_top3, val_top5, val_f1 = evaluate_model(exp14_model, val_loader, device)

            print(f"Stage 1 Epoch {epoch:2d}/5 | Loss: {loss:.4f} | Val Top-1: {val_top1:.2f}% | Val Top-5: {val_top5:.2f}% | Val F1: {val_f1:.2f}%", flush=True)

            if val_top1 > best_val_top1:
                best_val_top1 = val_top1
                torch.save({
                    "model_state_dict": exp14_model.state_dict(),
                    "input_size": input_size,
                    "hidden_size": 256,
                    "num_layers": 2,
                    "num_classes": len(class_names),
                    "class_names": class_names,
                    "architecture": "Exp11AdapterClassifier",
                    "val_top1": val_top1,
                    "val_top5": val_top5,
                    "val_macro_f1": val_f1,
                    "stage": 1,
                    "epoch": epoch,
                }, best_model_path)

        # 6. STAGE 2: Unfreeze Backbone, Differential Learning Rates (25 Epochs)
        print("\n--- STAGE 2: End-to-End Fine-Tuning (25 Epochs, Differential LR) ---", flush=True)
        for p in exp14_model.parameters():
            p.requires_grad = True

        param_groups = [
            {"params": exp14_model.adapter.parameters(), "lr": 1e-4},
            {"params": list(exp14_model.backbone.gru.parameters()) +
                       list(exp14_model.backbone.attention_net.parameters()) +
                       list(exp14_model.backbone.layer_norm.parameters()), "lr": 5e-6},
            {"params": exp14_model.backbone.classifier.parameters(), "lr": 1e-4},
        ]

        opt_stage2 = torch.optim.AdamW(param_groups, weight_decay=1e-4)

        def get_lr_scale(epoch: int) -> float:
            if epoch < 2:
                return (epoch + 1) / 2.0
            else:
                progress = (epoch - 2) / 23.0
                return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(opt_stage2, lr_lambda=get_lr_scale)

        for epoch in range(1, 26):
            loss = train_epoch_stage2(exp14_model, train_loader, opt_stage2, device)
            scheduler.step()

            val_top1, val_top3, val_top5, val_f1 = evaluate_model(exp14_model, val_loader, device)

            print(f"Stage 2 Epoch {epoch:2d}/25 | Loss: {loss:.4f} | Val Top-1: {val_top1:.2f}% | Val Top-5: {val_top5:.2f}% | Val F1: {val_f1:.2f}%", flush=True)

            if val_top1 > best_val_top1:
                best_val_top1 = val_top1
                torch.save({
                    "model_state_dict": exp14_model.state_dict(),
                    "input_size": input_size,
                    "hidden_size": 256,
                    "num_layers": 2,
                    "num_classes": len(class_names),
                    "class_names": class_names,
                    "architecture": "Exp11AdapterClassifier",
                    "val_top1": val_top1,
                    "val_top5": val_top5,
                    "val_macro_f1": val_f1,
                    "stage": 2,
                    "epoch": epoch,
                }, best_model_path)

        training_time = time.time() - start_time
        print(f"\nTraining Complete in {training_time/60.0:.2f} minutes.", flush=True)

    # 7. FINAL TEST SET EVALUATION
    print(f"\n--- FINAL TEST EVALUATION ({exp_name.upper()}) ---", flush=True)
    best_checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
    exp14_model.load_state_dict(best_checkpoint["model_state_dict"])

    test_top1, test_top3, test_top5, test_f1 = evaluate_model(exp14_model, test_loader, device)

    delta_top1 = test_top1 - EXP13_BASELINE_TOP1
    delta_top5 = test_top5 - EXP13_BASELINE_TOP5
    delta_f1 = test_f1 - EXP13_BASELINE_MACRO_F1

    if test_top1 > EXP13_BASELINE_TOP1 and delta_f1 >= -1.0:
        decision = "KEEP"
    else:
        decision = "REJECT"

    results = {
        "experiment": exp_name,
        "input_size": input_size,
        "feature_mode": feature_mode,
        "val_top1": best_val_top1,
        "test_top1": test_top1,
        "test_top3": test_top3,
        "test_top5": test_top5,
        "test_macro_f1": test_f1,
        "delta_top1": delta_top1,
        "delta_top5": delta_top5,
        "delta_macro_f1": delta_f1,
        "decision": decision,
        "training_time_sec": training_time,
    }

    with open(output_dir / "test_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n=== RESULTS FOR {exp_name.upper()} ===", flush=True)
    print(f"  Validation Top-1: {best_val_top1:.2f}%", flush=True)
    print(f"  Test Top-1:       {test_top1:.2f}% (EXP13: {EXP13_BASELINE_TOP1:.2f}% | Delta: {delta_top1:+.2f}%)", flush=True)
    print(f"  Test Top-3:       {test_top3:.2f}%", flush=True)
    print(f"  Test Top-5:       {test_top5:.2f}% (EXP13: {EXP13_BASELINE_TOP5:.2f}% | Delta: {delta_top5:+.2f}%)", flush=True)
    print(f"  Test Macro F1:    {test_f1:.2f}% (EXP13: {EXP13_BASELINE_MACRO_F1:.2f}% | Delta: {delta_f1:+.2f}%)", flush=True)
    print(f"  DECISION:         {decision}\n", flush=True)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EXP14 Implementation & Ablation Suite")
    parser.add_argument("--exp", type=str, default="exp14a", choices=["exp14a", "exp14b", "exp14c", "all"], help="Experiment code")
    parser.add_argument("--eval_only", action="store_true", help="Skip training and perform test evaluation on existing best_model.pth")
    args = parser.parse_args()

    if args.exp == "all":
        res_a = run_exp14("exp14a", eval_only=args.eval_only)
        res_b = run_exp14("exp14b", eval_only=args.eval_only)
        res_c = run_exp14("exp14c", eval_only=args.eval_only)
    else:
        run_exp14(args.exp, eval_only=args.eval_only)

