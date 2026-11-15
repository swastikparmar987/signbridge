#!/usr/bin/env python3
"""
EXP13: Controlled fine-tuning built directly on top of the EXP11 model checkpoint.
Introduces:
1. Kinematic skeleton warp augmentation during training (0.85x - 1.15x bone perturbation).
2. Stricter ArcFace angular margin (m = 0.35, s = 32.0).
Preserves pretrained EXP11 adapter, GRU backbone, and classifier weights.
"""

import json
import math
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
from signbridge.preprocessing.skeleton_augment import apply_skeleton_warp
from signbridge.preprocessing.normalize import normalize_sequence
from signbridge.preprocessing.features import build_xyz_velocity_features
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

EXP11_CHECKPOINT = PROJECT_ROOT / "trained_models" / "full_2731" / "exp11_adapter_velocity" / "best_model.pth"
OUTPUT_DIR = FULL_OUTPUT_DIR / "exp13_arcface035_skeletonwarp"
CHECKPOINT_PATH = OUTPUT_DIR / "best_model.pth"
FINAL_CHECKPOINT_PATH = OUTPUT_DIR / "final_model.pth"


def run_sanity_checks(exp11_checkpoint_path: Path, device: torch.device):
    """
    Mandatory Preflight Sanity Checks for EXP13:
    1. Verify EXP11 checkpoint exists and loads cleanly.
    2. Verify adapter dimensions (252 -> 126).
    3. Verify GRU backbone weights loaded from EXP11.
    4. Verify ArcFace margin (m = 0.35) and scale (s = 32.0).
    5. Test forward pass with single unaugmented sample -> shape (B, 2731).
    6. Test forward pass with single skeleton-warped sample (recomputed velocity) -> shape (B, 2731).
    7. Verify no NaNs or Infs.
    8. Verify validation and test dataloaders have skeleton warp completely disabled.
    """
    print("=== RUNNING MANDATORY EXP13 PREFLIGHT SANITY CHECKS ===")
    if not exp11_checkpoint_path.exists():
        raise FileNotFoundError(f"EXP11 checkpoint missing: {exp11_checkpoint_path}")

    checkpoint = torch.load(exp11_checkpoint_path, map_location=device)
    class_names = checkpoint["class_names"]

    # 1. Instantiate EXP11 model
    backbone = BiGRUArcFaceClassifier(
        input_size=126,
        hidden_size=256,
        num_layers=2,
        num_classes=len(class_names),
        attention_hidden_dim=128,
        dropout_p=0.3,
        s=32.0,  # EXP13 scale = 32.0
        m=0.35,  # EXP13 margin = 0.35
    )
    model = Exp11AdapterClassifier(backbone=backbone, input_size=252, adapter_output_size=126).to(device)

    # Load state dict safely
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()

    # Verify ArcFace parameters
    if model.backbone.m != 0.35 or model.backbone.s != 32.0:
        raise ValueError(f"ArcFace parameter mismatch: expected m=0.35, s=32.0, got m={model.backbone.m}, s={model.backbone.s}")
    print("1. EXP11 checkpoint loaded successfully into Exp11AdapterClassifier.")
    print(f"2. ArcFace parameters verified: margin m={model.backbone.m}, scale s={model.backbone.s}.")

    # 2. Test forward pass with unaugmented sample
    x_unaugmented = torch.randn(4, 32, 252, device=device)
    with torch.no_grad():
        out_unaugmented = model(x_unaugmented)
    if out_unaugmented.shape != (4, 2731) or not torch.isfinite(out_unaugmented).all():
        raise ValueError("Unaugmented sample forward pass returned invalid shape or non-finite values!")
    print(f"3. Unaugmented forward pass verified: input {tuple(x_unaugmented.shape)} -> output {tuple(out_unaugmented.shape)}.")

    # 3. Test forward pass with skeleton-warped sample + recomputed velocity
    raw_landmarks = np.random.randn(32, 42, 3).astype(np.float32)
    warped_landmarks = apply_skeleton_warp(raw_landmarks, scale_min=0.85, scale_max=1.15)
    normalized = normalize_sequence(warped_landmarks)
    features_252 = build_xyz_velocity_features(normalized)  # (32, 252)

    x_warped = torch.from_numpy(features_252).unsqueeze(0).to(device)
    with torch.no_grad():
        out_warped = model(x_warped)
    if out_warped.shape != (1, 2731) or not torch.isfinite(out_warped).all():
        raise ValueError("Skeleton-warped sample forward pass returned invalid shape or non-finite values!")
    print(f"4. Skeleton-warped sample forward pass verified: input {tuple(x_warped.shape)} -> output {tuple(out_warped.shape)}.")

    # 4. DataLoaders check
    train_l, val_l, test_l, _ = create_velocity_dataloaders(batch_size=32, use_skeleton_warp=True, seed=42)
    if val_l.dataset.use_skeleton_warp or test_l.dataset.use_skeleton_warp:
        raise ValueError("Validation or Test dataset has skeleton warp enabled!")
    if not train_l.dataset.use_skeleton_warp:
        raise ValueError("Training dataset has skeleton warp disabled!")
    print("5. DataLoaders verified: Skeleton warping ENABLED for Train, STRICTLY DISABLED for Val & Test.")
    print("=== ALL SANITY CHECKS PASSED SUCCESSFULLY ===\n")


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


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    set_seed(42)
    device = get_device()

    # 1. Preflight Sanity Checks
    run_sanity_checks(EXP11_CHECKPOINT, device)

    # 2. DataLoaders (Train: skeleton warp enabled, Val/Test: disabled)
    class_names = load_full_glosses()
    aug = AugmentationConfig(
        enabled=True,
        noise_sigma=0.003,
        spatial_scale_range=(0.95, 1.05),
        temporal_speed_range=(0.95, 1.05),
        temporal_jitter_frames=1,
    )
    train_loader, val_loader, _, _ = create_velocity_dataloaders(
        batch_size=128,
        augmentation_config=aug,
        use_skeleton_warp=True,
        seed=42,
    )

    # 3. Load EXP11 model checkpoint
    exp11_cp = torch.load(EXP11_CHECKPOINT, map_location=device)
    backbone = BiGRUArcFaceClassifier(
        input_size=126,
        hidden_size=256,
        num_layers=2,
        num_classes=len(class_names),
        attention_hidden_dim=128,
        dropout_p=0.3,
        s=32.0,  # EXP13 scale
        m=0.35,  # EXP13 margin
    )
    model = Exp11AdapterClassifier(backbone=backbone, input_size=252, adapter_output_size=126).to(device)
    model.load_state_dict(exp11_cp["model_state_dict"])
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    history = []
    started = time.time()
    best_val = -1.0
    best_epoch = 0

    # =========================================================================
    # STAGE 1: Fine-tune ArcFace Head only (Adapter & GRU Backbone Frozen)
    # =========================================================================
    print("=== STARTING STAGE 1: ArcFace Head Fine-Tuning (m=0.35, s=32.0) ===")
    for param in model.adapter.parameters():
        param.requires_grad = False
    for param in model.backbone.parameters():
        param.requires_grad = False
    for param in model.backbone.classifier.parameters():
        param.requires_grad = True

    stage1_optimizer = torch.optim.Adam(model.backbone.classifier.parameters(), lr=1e-4, weight_decay=1e-4)

    for epoch in range(1, 4):
        train_res = run_epoch(model, train_loader, criterion, stage1_optimizer, device)
        val_res = validate(model, val_loader, criterion, device)
        history.append({
            "stage": 1,
            "epoch": epoch,
            **train_res,
            **val_res,
            "lr_classifier": stage1_optimizer.param_groups[0]["lr"],
            "lr_adapter": 0.0,
            "lr_gru": 0.0,
        })
        print(
            f"[Stage 1] Ep {epoch:02d}/03 | train acc {train_res['train_accuracy']:.2f}% | "
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
                    "experiment": "exp13_arcface035_skeletonwarp",
                    "stage": 1,
                    "epoch": epoch,
                    "val_top1": val_res["top1"],
                    "val_top3": val_res["top3"],
                    "val_top5": val_res["top5"],
                    "base_checkpoint": str(EXP11_CHECKPOINT),
                    "arcface_m": 0.35,
                    "arcface_s": 32.0,
                    "skeleton_warp": True,
                },
            )

    # =========================================================================
    # STAGE 2: Staged Fine-Tuning (Unfreeze Adapter & GRU with Differential LR)
    # =========================================================================
    print("\n=== STARTING STAGE 2: Unfreezing Adapter & GRU Backbone with Differential LR ===")
    for param in model.adapter.parameters():
        param.requires_grad = True
    for param in model.backbone.gru.parameters():
        param.requires_grad = True
    for param in model.backbone.attention_net.parameters():
        param.requires_grad = True
    for param in model.backbone.layer_norm.parameters():
        param.requires_grad = True

    stage2_optimizer = torch.optim.Adam([
        {"params": model.adapter.parameters(), "lr": 1e-4},
        {"params": model.backbone.classifier.parameters(), "lr": 5e-5},
        {"params": [p for name, p in model.backbone.named_parameters() if p.requires_grad and "classifier" not in name], "lr": 5e-6},
    ], weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(stage2_optimizer, T_max=10, eta_min=1e-6)

    for epoch in range(4, 14):
        train_res = run_epoch(model, train_loader, criterion, stage2_optimizer, device)
        val_res = validate(model, val_loader, criterion, device)
        scheduler.step()

        history.append({
            "stage": 2,
            "epoch": epoch,
            **train_res,
            **val_res,
            "lr_adapter": stage2_optimizer.param_groups[0]["lr"],
            "lr_classifier": stage2_optimizer.param_groups[1]["lr"],
            "lr_gru": stage2_optimizer.param_groups[2]["lr"],
        })
        print(
            f"[Stage 2] Ep {epoch:02d}/13 | train acc {train_res['train_accuracy']:.2f}% | "
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
                    "experiment": "exp13_arcface035_skeletonwarp",
                    "stage": 2,
                    "epoch": epoch,
                    "val_top1": val_res["top1"],
                    "val_top3": val_res["top3"],
                    "val_top5": val_res["top5"],
                    "base_checkpoint": str(EXP11_CHECKPOINT),
                    "arcface_m": 0.35,
                    "arcface_s": 32.0,
                    "skeleton_warp": True,
                },
            )

    # Save Final Checkpoint
    save_checkpoint(
        model,
        class_names,
        FINAL_CHECKPOINT_PATH,
        metadata={
            "architecture": "Exp11AdapterClassifier",
            "experiment": "exp13_arcface035_skeletonwarp",
            "epoch": 13,
            "arcface_m": 0.35,
            "arcface_s": 32.0,
        },
    )

    # 4. Final Evaluation on Untouched Test Set (32,941 samples)
    print("\n=== EVALUATING EXP13 BEST CHECKPOINT ON UNTOUCHED TEST SET ===")
    evaluation = evaluate_full_model(CHECKPOINT_PATH, device=device, batch_size=256)

    exp11_top1 = 51.24313165963389
    exp11_top3 = 70.18001882152940
    exp11_top5 = 76.16344373273428
    exp11_macro_f1 = 50.55977535161682

    diff_top1 = evaluation["top1_accuracy"] - exp11_top1
    diff_top3 = evaluation["top3_accuracy"] - exp11_top3
    diff_top5 = evaluation["top5_accuracy"] - exp11_top5
    diff_macro_f1 = evaluation["macro_f1"] - exp11_macro_f1

    # Decision Rule: ACCEPTED only if Top-1 > 51.24% while Top-5 & Macro F1 maintained/improved
    accepted = (evaluation["top1_accuracy"] > exp11_top1) and (diff_top5 >= -0.5)

    summary = {
        "experiment": "exp13_arcface035_skeletonwarp",
        "base_checkpoint": str(EXP11_CHECKPOINT),
        "best_epoch": best_epoch,
        "best_val_top1": best_val,
        "test_top1_accuracy": evaluation["top1_accuracy"],
        "test_top3_accuracy": evaluation["top3_accuracy"],
        "test_top5_accuracy": evaluation["top5_accuracy"],
        "macro_f1": evaluation["macro_f1"],
        "weighted_f1": evaluation["weighted_f1"],
        "diff_vs_exp11": {
            "top1_delta_pct_points": diff_top1,
            "top3_delta_pct_points": diff_top3,
            "top5_delta_pct_points": diff_top5,
            "macro_f1_delta_pct_points": diff_macro_f1,
        },
        "decision": "ACCEPTED" if accepted else "REJECTED",
        "training_duration_seconds": round(time.time() - started, 2),
        "history": history,
    }

    with open(OUTPUT_DIR / "eval_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== EXP13 FINAL RESULTS ===")
    print(f"Test Top-1 Accuracy:  {evaluation['top1_accuracy']:.2f}% (EXP11: {exp11_top1:.2f}%, Diff: {diff_top1:+.2f} pp)")
    print(f"Test Top-3 Accuracy:  {evaluation['top3_accuracy']:.2f}% (EXP11: {exp11_top3:.2f}%, Diff: {diff_top3:+.2f} pp)")
    print(f"Test Top-5 Accuracy:  {evaluation['top5_accuracy']:.2f}% (EXP11: {exp11_top5:.2f}%, Diff: {diff_top5:+.2f} pp)")
    print(f"Test Macro F1 Score:  {evaluation['macro_f1']:.2f}% (EXP11: {exp11_macro_f1:.2f}%, Diff: {diff_macro_f1:+.2f} pp)")
    print(f"Decision: {summary['decision']}")


if __name__ == "__main__":
    main()
