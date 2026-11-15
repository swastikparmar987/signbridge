#!/usr/bin/env python3
"""
Exp11: Identity-initialized Linear(252 -> 126) adapter feeding into pretrained EXP9 GRU + ArcFace head.
Preserves EXP9 pretrained representation while introducing frame-to-frame velocity features.
"""

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

from signbridge.inference.model import BiGRUArcFaceClassifier, Exp11AdapterClassifier
from signbridge.training.config import (
    AugmentationConfig,
    FULL_OUTPUT_DIR,
    get_device,
    load_full_glosses,
    set_seed,
)
from signbridge.training.dataset import create_velocity_dataloaders
from signbridge.training.evaluate import evaluate_full_model
from signbridge.training.metrics import compute_topk_accuracies
from signbridge.training.model import save_checkpoint

EXP9_CHECKPOINT = PROJECT_ROOT / "trained_models" / "full_2731" / "exp9_true_arcface" / "best_model.pth"
OUTPUT_DIR = FULL_OUTPUT_DIR / "exp11_adapter_velocity"
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


def sanity_check_adapter(exp9_checkpoint_path: Path, device: torch.device):
    """
    Mandatory preflight check:
    Verify EXP11 with adapter initialized as [I_126 | 0_126] produces identical outputs to EXP9
    when velocity features are zero.
    """
    print("=== RUNNING MANDATORY SANITY CHECK ===")
    if not exp9_checkpoint_path.exists():
        raise FileNotFoundError(f"EXP9 baseline checkpoint missing: {exp9_checkpoint_path}")

    checkpoint = torch.load(exp9_checkpoint_path, map_location=device)
    class_names = checkpoint["class_names"]

    # 1. Instantiate pure EXP9 model
    exp9_model = BiGRUArcFaceClassifier(
        input_size=126,
        hidden_size=256,
        num_layers=2,
        num_classes=len(class_names),
        attention_hidden_dim=128,
        dropout_p=0.3,
        s=30.0,
        m=0.25,
    ).to(device)
    exp9_model.load_state_dict(checkpoint["model_state_dict"])
    exp9_model.eval()

    # 2. Instantiate EXP11 model and load EXP9 backbone
    backbone = BiGRUArcFaceClassifier(
        input_size=126,
        hidden_size=256,
        num_layers=2,
        num_classes=len(class_names),
        attention_hidden_dim=128,
        dropout_p=0.3,
        s=30.0,
        m=0.25,
    )
    backbone.load_state_dict(checkpoint["model_state_dict"])
    exp11_model = Exp11AdapterClassifier(backbone=backbone, input_size=252, adapter_output_size=126).to(device)
    exp11_model.eval()

    # 3. Create dummy 126-dim input and zero-velocity padded 252-dim input
    torch.manual_seed(42)
    x_126 = torch.randn(8, 32, 126, device=device)
    x_252_zero_vel = torch.cat([x_126, torch.zeros_like(x_126)], dim=-1)

    with torch.no_grad():
        out_exp9 = exp9_model(x_126)
        out_exp11 = exp11_model(x_252_zero_vel)

    max_diff = (out_exp9 - out_exp11).abs().max().item()
    print(f"Max absolute logit difference between EXP9 and EXP11([XYZ, 0]): {max_diff:.8f}")

    if max_diff > 1e-4:
        raise ValueError(f"SANITY CHECK FAILED! Output difference {max_diff} > 1e-4. Stopping training.")
    print("SANITY CHECK PASSED: EXP11 identity adapter matches EXP9 output.\n")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    set_seed(42)
    device = get_device()

    # 1. Run mandatory preflight sanity check
    sanity_check_adapter(EXP9_CHECKPOINT, device)

    # 2. Load dataset and glosses
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

    # 3. Load EXP9 pretrained checkpoint into EXP11 model
    exp9_cp = torch.load(EXP9_CHECKPOINT, map_location=device)
    backbone = BiGRUArcFaceClassifier(
        input_size=126,
        hidden_size=256,
        num_layers=2,
        num_classes=len(class_names),
        attention_hidden_dim=128,
        dropout_p=0.3,
        s=30.0,
        m=0.25,
    )
    backbone.load_state_dict(exp9_cp["model_state_dict"])
    model = Exp11AdapterClassifier(backbone=backbone, input_size=252, adapter_output_size=126).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    history = []
    started = time.time()
    best_val = -1.0
    best_epoch = 0

    # =========================================================================
    # STAGE 1: Train ONLY Linear(252 -> 126) adapter (Backbone Frozen)
    # =========================================================================
    print("=== STARTING STAGE 1: Adapter-Only Training (Backbone Frozen) ===")
    for param in model.backbone.parameters():
        param.requires_grad = False
    for param in model.adapter.parameters():
        param.requires_grad = True

    stage1_optimizer = torch.optim.Adam(model.adapter.parameters(), lr=5e-4, weight_decay=1e-4)

    for epoch in range(1, 7):
        train_res = run_epoch(model, train_loader, criterion, stage1_optimizer, device)
        val_res = validate(model, val_loader, criterion, device)
        history.append({
            "stage": 1,
            "epoch": epoch,
            **train_res,
            **val_res,
            "lr_adapter": stage1_optimizer.param_groups[0]["lr"],
            "lr_gru": 0.0,
        })
        print(
            f"[Stage 1] Ep {epoch:02d}/06 | train acc {train_res['train_accuracy']:.2f}% | "
            f"val top1 {val_res['top1']:.2f}% top5 {val_res['top5']:.2f}%",
            flush=True,
        )
        if val_res["top1"] > best_val:
            best_val = val_res["top1"]
            best_epoch = epoch
            save_checkpoint(
                model,
                class_names,
                CHECKPOINT_PATH,
                metadata={
                    "architecture": "Exp11AdapterClassifier",
                    "experiment": "exp11_adapter_velocity",
                    "stage": 1,
                    "epoch": epoch,
                    "val_top1": val_res["top1"],
                    "val_top3": val_res["top3"],
                    "val_top5": val_res["top5"],
                    "base_checkpoint": str(EXP9_CHECKPOINT),
                    "input_size": 252,
                    "adapter_output_size": 126,
                },
            )

    # =========================================================================
    # STAGE 2: Staged Fine-Tuning (Unfreeze GRU with Differential Learning Rate)
    # =========================================================================
    print("\n=== STARTING STAGE 2: Unfreezing GRU Backbone with Differential LR ===")
    for param in model.backbone.gru.parameters():
        param.requires_grad = True
    for param in model.backbone.attention_net.parameters():
        param.requires_grad = True
    for param in model.backbone.layer_norm.parameters():
        param.requires_grad = True
    # Keep ArcFace head protected
    for param in model.backbone.classifier.parameters():
        param.requires_grad = False

    stage2_optimizer = torch.optim.Adam([
        {"params": model.adapter.parameters(), "lr": 2e-4},
        {"params": [p for p in model.backbone.parameters() if p.requires_grad], "lr": 5e-6},
    ], weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(stage2_optimizer, T_max=10, eta_min=1e-6)

    for epoch in range(7, 17):
        train_res = run_epoch(model, train_loader, criterion, stage2_optimizer, device)
        val_res = validate(model, val_loader, criterion, device)
        scheduler.step()

        history.append({
            "stage": 2,
            "epoch": epoch,
            **train_res,
            **val_res,
            "lr_adapter": stage2_optimizer.param_groups[0]["lr"],
            "lr_gru": stage2_optimizer.param_groups[1]["lr"],
        })
        print(
            f"[Stage 2] Ep {epoch:02d}/16 | train acc {train_res['train_accuracy']:.2f}% | "
            f"val top1 {val_res['top1']:.2f}% top5 {val_res['top5']:.2f}%",
            flush=True,
        )
        if val_res["top1"] > best_val:
            best_val = val_res["top1"]
            best_epoch = epoch
            save_checkpoint(
                model,
                class_names,
                CHECKPOINT_PATH,
                metadata={
                    "architecture": "Exp11AdapterClassifier",
                    "experiment": "exp11_adapter_velocity",
                    "stage": 2,
                    "epoch": epoch,
                    "val_top1": val_res["top1"],
                    "val_top3": val_res["top3"],
                    "val_top5": val_res["top5"],
                    "base_checkpoint": str(EXP9_CHECKPOINT),
                    "input_size": 252,
                    "adapter_output_size": 126,
                },
            )

    # 4. Final Test Set Evaluation
    print("\n=== EVALUATING FINAL BEST CHECKPOINT ON UNTOUCHED TEST SET ===")
    evaluation = evaluate_full_model(CHECKPOINT_PATH, device=device, batch_size=256)

    exp9_top1 = 51.08527367110896
    exp9_top5 = 75.96915697762667
    exp9_macro_f1 = 50.413835203301986

    diff_top1 = evaluation["top1_accuracy"] - exp9_top1
    diff_top5 = evaluation["top5_accuracy"] - exp9_top5
    diff_macro_f1 = evaluation["macro_f1"] - exp9_macro_f1

    accepted = (evaluation["top1_accuracy"] > exp9_top1) and (diff_top5 >= -0.5)

    summary = {
        "experiment": "exp11_adapter_velocity",
        "base_checkpoint": str(EXP9_CHECKPOINT),
        "best_epoch": best_epoch,
        "best_val_top1": best_val,
        "test_top1_accuracy": evaluation["top1_accuracy"],
        "test_top3_accuracy": evaluation["top3_accuracy"],
        "test_top5_accuracy": evaluation["top5_accuracy"],
        "macro_f1": evaluation["macro_f1"],
        "weighted_f1": evaluation["weighted_f1"],
        "diff_vs_exp9": {
            "top1_delta_pct_points": diff_top1,
            "top5_delta_pct_points": diff_top5,
            "macro_f1_delta_pct_points": diff_macro_f1,
        },
        "decision": "ACCEPTED" if accepted else "REJECTED",
        "training_duration_seconds": round(time.time() - started, 2),
        "history": history,
    }

    with open(OUTPUT_DIR / "eval_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== EXP11 FINAL RESULTS ===")
    print(f"Test Top-1 Accuracy:  {evaluation['top1_accuracy']:.2f}% (EXP9: {exp9_top1:.2f}%, Diff: {diff_top1:+.2f} pp)")
    print(f"Test Top-5 Accuracy:  {evaluation['top5_accuracy']:.2f}% (EXP9: {exp9_top5:.2f}%, Diff: {diff_top5:+.2f} pp)")
    print(f"Test Macro F1 Score:  {evaluation['macro_f1']:.2f}% (EXP9: {exp9_macro_f1:.2f}%, Diff: {diff_macro_f1:+.2f} pp)")
    print(f"Decision: {summary['decision']}")


if __name__ == "__main__":
    main()
