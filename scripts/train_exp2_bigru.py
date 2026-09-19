#!/usr/bin/env python3
"""
SignBridge Full 2,731-Class Experiment 2: BiGRU with Triple Temporal Attention & Pooling
Architecture:
- 2-Layer Bidirectional GRU (Hidden Size: 256 per dir = 512 total) -> [B, 32, 512]
- Triple Temporal Aggregation:
  1. Learnable Additive Attention Pooling -> [B, 512]
  2. Global Temporal Mean Pooling        -> [B, 512]
  3. Global Temporal Max Pooling         -> [B, 512]
- Concatenated Fused Vector:             -> [B, 1536]
- Classification Head: LayerNorm(1536) -> Dropout(0.3) -> Linear(1536, 2731)
"""

import json
import sys
import time
from pathlib import Path
from typing import Dict, Any, List

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


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float = 1.0,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total_samples = 0

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        batch_size = x.size(0)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()

        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        total_loss += loss.item() * batch_size
        preds = logits.argmax(dim=-1)
        correct += (preds == y).sum().item()
        total_samples += batch_size

    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    accuracy = (correct / total_samples * 100.0) if total_samples > 0 else 0.0

    return {
        "train_loss": float(avg_loss),
        "train_accuracy": float(accuracy),
    }


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    all_logits = []
    all_targets = []

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            batch_size = x.size(0)

            logits = model(x)
            loss = criterion(logits, y)

            total_loss += loss.item() * batch_size
            all_logits.append(logits.cpu())
            all_targets.append(y.cpu())

    total_samples = len(dataloader.dataset)
    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0

    all_logits = torch.cat(all_logits, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    topk_res = compute_topk_accuracies(all_logits, all_targets, topk=(1, 3, 5))

    return {
        "val_loss": float(avg_loss),
        "val_top1": topk_res["top1"],
        "val_top3": topk_res["top3"],
        "val_top5": topk_res["top5"],
    }


def main():
    exp_name = "exp2_bigru_attention_pooling"
    output_dir = FULL_OUTPUT_DIR / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"🚀 STARTING FULL 2,731-CLASS EXPERIMENT 2: {exp_name.upper()}")
    print("=" * 80)
    print(f"Output Directory:    {output_dir}")

    # Hyperparameters
    batch_size = 128
    learning_rate = 1e-3
    weight_decay = 1e-4
    grad_clip = 1.0
    max_epochs = 35
    patience = 10
    min_delta = 1e-4
    seed = 42

    set_seed(seed)
    device = get_device()
    print(f"Target Device:       {device}")

    # 1. Dataloaders with Safe Augmentation
    aug_config = AugmentationConfig(
        enabled=True,
        noise_sigma=0.003,
        spatial_scale_range=(0.95, 1.05),
        temporal_speed_range=(0.95, 1.05),
        temporal_jitter_frames=1,
    )

    train_loader, val_loader, test_loader, class_names = create_full_dataloaders(
        batch_size=batch_size,
        augmentation_config=aug_config,
        seed=seed,
    )
    print(f"Dataset Split:       Train={len(train_loader.dataset)}, Val={len(val_loader.dataset)}, Test={len(test_loader.dataset)} samples across {len(class_names)} classes")

    # 2. Build BiGRUAttentionPoolingClassifier
    model = BiGRUAttentionPoolingClassifier(
        input_size=126,
        hidden_size=256,
        num_layers=2,
        num_classes=len(class_names),
        attention_hidden_dim=128,
        dropout_p=0.3,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Architecture:  BiGRUAttentionPoolingClassifier")
    print(f"Total Parameters:    {total_params:,} (6.04M params)")
    print(f"Fused Dimension:     {model.fused_dim} (512 Attn + 512 Mean + 512 Max)")
    print(f"Regularization:      LayerNorm(1536) -> Dropout(0.3) -> Weight Decay 1e-4")
    print(f"Optimizer:           Adam (lr={learning_rate:.1e}, weight_decay={weight_decay:.1e})")
    print("=" * 80)

    # Verify tensor shapes with a dummy pass
    dummy_x = torch.randn(4, 32, 126, device=device)
    with torch.no_grad():
        dummy_out = model(dummy_x)
    assert dummy_out.shape == (4, 2731), f"Unexpected shape {dummy_out.shape}"
    print(f"✅ Verified forward tensor shape: {tuple(dummy_out.shape)}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3,
        min_lr=1e-5,
    )

    history: List[Dict[str, Any]] = []
    best_val_top1 = -1.0
    best_epoch = -1
    epochs_no_improve = 0
    best_checkpoint_path = output_dir / "best_model.pth"

    t_train_start = time.time()
    print("\n--- TRAINING PROGRESSION ---")

    for epoch in range(1, max_epochs + 1):
        t0 = time.time()
        current_lr = optimizer.param_groups[0]["lr"]

        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, grad_clip
        )
        val_metrics = validate(model, val_loader, criterion, device)
        scheduler.step(val_metrics["val_top1"])
        elapsed = time.time() - t0

        epoch_record = {
            "epoch": epoch,
            "lr": current_lr,
            **train_metrics,
            **val_metrics,
            "elapsed_sec": round(elapsed, 2),
        }
        history.append(epoch_record)

        is_best = val_metrics["val_top1"] > best_val_top1 + min_delta
        if is_best:
            best_val_top1 = val_metrics["val_top1"]
            best_epoch = epoch
            epochs_no_improve = 0
            save_checkpoint(
                model,
                class_names,
                best_checkpoint_path,
                metadata={
                    "architecture": "BiGRUAttentionPoolingClassifier",
                    "experiment": exp_name,
                    "epoch": epoch,
                    "val_top1": best_val_top1,
                    "val_top3": val_metrics["val_top3"],
                    "val_top5": val_metrics["val_top5"],
                    "total_params": total_params,
                    "attention_hidden_dim": 128,
                    "dropout_p": 0.3,
                },
            )
        else:
            epochs_no_improve += 1

        best_marker = " 🌟 [BEST]" if is_best else ""
        print(
            f"   Ep {epoch:2d}/{max_epochs} | "
            f"Train Loss: {train_metrics['train_loss']:.4f} Acc: {train_metrics['train_accuracy']:.2f}% | "
            f"Val Loss: {val_metrics['val_loss']:.4f} Top-1: {val_metrics['val_top1']:.2f}% Top-5: {val_metrics['val_top5']:.2f}% | "
            f"lr: {current_lr:.1e} ({elapsed:.1f}s){best_marker}",
            flush=True,
        )

        if epochs_no_improve >= patience:
            print(f"\n   ⏹️ Early stopping triggered after {epochs_no_improve} epochs with no validation improvement.")
            break

    total_train_duration = time.time() - t_train_start
    print(f"\n✅ Training complete in {total_train_duration / 60:.1f} minutes ({total_train_duration:.1f}s).")
    print(f"   Best Val Top-1: {best_val_top1:.2f}% at epoch {best_epoch}")
    print(f"   Best checkpoint: {best_checkpoint_path}")

    # Save training history and configuration
    history_file = output_dir / "training_history.json"
    with open(history_file, "w") as f:
        json.dump(
            {
                "experiment": exp_name,
                "architecture": "BiGRUAttentionPoolingClassifier",
                "total_params": total_params,
                "best_epoch": best_epoch,
                "best_val_top1": best_val_top1,
                "training_duration_seconds": round(total_train_duration, 2),
                "history": history,
            },
            f,
            indent=2,
        )

    config_file = output_dir / "config.json"
    with open(config_file, "w") as f:
        json.dump(
            {
                "name": exp_name,
                "architecture": "BiGRUAttentionPoolingClassifier",
                "total_params": total_params,
                "input_size": 126,
                "hidden_size": 256,
                "num_layers": 2,
                "bidirectional": True,
                "attention_hidden_dim": 128,
                "fused_dim": 1536,
                "dropout_p": 0.3,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "grad_clip": grad_clip,
                "max_epochs": max_epochs,
                "patience": patience,
                "seed": seed,
                "augmentation": {
                    "enabled": aug_config.enabled,
                    "noise_sigma": aug_config.noise_sigma,
                    "spatial_scale_range": aug_config.spatial_scale_range,
                    "temporal_speed_range": aug_config.temporal_speed_range,
                    "temporal_jitter_frames": aug_config.temporal_jitter_frames,
                },
            },
            f,
            indent=2,
        )

    # 3. Final Evaluation on Untouched Test Split (32,941 samples)
    print("\n" + "=" * 80)
    print(f"🧪 EVALUATING BEST CHECKPOINT ON UNTOUCHED TEST SET (32,941 SAMPLES)...")
    print(f"   Checkpoint: {best_checkpoint_path}")
    print("=" * 80)

    t_eval_start = time.time()
    eval_result = evaluate_full_model(best_checkpoint_path, batch_size=256)
    eval_duration = time.time() - t_eval_start

    eval_summary = {
        "experiment": exp_name,
        "architecture": "BiGRUAttentionPoolingClassifier",
        "total_params": total_params,
        "checkpoint_path": str(best_checkpoint_path),
        "best_epoch": best_epoch,
        "best_val_top1": best_val_top1,
        "test_top1_accuracy": eval_result["top1_accuracy"],
        "test_top3_accuracy": eval_result["top3_accuracy"],
        "test_top5_accuracy": eval_result["top5_accuracy"],
        "macro_f1": eval_result["macro_f1"],
        "weighted_f1": eval_result["weighted_f1"],
        "total_test_samples": eval_result["total_samples"],
        "training_duration_seconds": round(total_train_duration, 2),
        "evaluation_duration_seconds": round(eval_duration, 2),
        "top_confusions": eval_result["top_confusions"],
        "worst_classes": eval_result["worst_classes"],
        "best_classes": eval_result["best_classes"],
    }

    eval_file = output_dir / "eval_results.json"
    with open(eval_file, "w") as f:
        json.dump(eval_summary, f, indent=2)

    # 4. Final Comparison vs Baseline & Exp 1
    base_top1 = 34.84
    base_top5 = 61.54
    exp1_top1 = 36.56
    exp1_top3 = 55.69
    exp1_top5 = 63.21

    new_top1 = eval_result["top1_accuracy"]
    new_top3 = eval_result["top3_accuracy"]
    new_top5 = eval_result["top5_accuracy"]

    d_base_top1 = new_top1 - base_top1
    d_base_top5 = new_top5 - base_top5
    d_exp1_top1 = new_top1 - exp1_top1
    d_exp1_top5 = new_top5 - exp1_top5

    print("\n" + "=" * 80)
    print("📊 FINAL BENCHMARK COMPARISON REPORT")
    print("=" * 80)
    print(f"{'Metric':<25} | {'Baseline':<12} | {'Exp 1 (Fine-Tune)':<18} | {'Exp 2 (BiGRU-Attn)':<18} | {'vs Exp 1 Delta':<15}")
    print("-" * 95)
    print(f"{'Architecture':<25} | {'Uni-GRU (1.39M)':<12} | {'Uni-GRU (1.39M)':<18} | {'BiGRU-Attn (6.04M)':<18} | {'+4.65M params'}")
    print(f"{'Test Top-1 Accuracy':<25} | {base_top1:6.2f}%      | {exp1_top1:6.2f}%           | {new_top1:6.2f}%           | {'+' if d_exp1_top1 >= 0 else ''}{d_exp1_top1:+.2f}%")
    print(f"{'Test Top-3 Accuracy':<25} | {'—':<12} | {exp1_top3:6.2f}%           | {new_top3:6.2f}%           | {'+' if new_top3 - exp1_top3 >= 0 else ''}{new_top3 - exp1_top3:+.2f}%")
    print(f"{'Test Top-5 Accuracy':<25} | {base_top5:6.2f}%      | {exp1_top5:6.2f}%           | {new_top5:6.2f}%           | {'+' if d_exp1_top5 >= 0 else ''}{d_exp1_top5:+.2f}%")
    print(f"{'Test Macro F1':<25} | {'—':<12} | {36.36:6.2f}%           | {eval_result['macro_f1']:6.2f}%           | {'+' if eval_result['macro_f1'] - 36.36 >= 0 else ''}{eval_result['macro_f1'] - 36.36:+.2f}%")
    print(f"{'Test Weighted F1':<25} | {'—':<12} | {36.44:6.2f}%           | {eval_result['weighted_f1']:6.2f}%           | {'+' if eval_result['weighted_f1'] - 36.44 >= 0 else ''}{eval_result['weighted_f1'] - 36.44:+.2f}%")
    print(f"{'Best Val Top-1':<25} | {45.38:6.2f}%      | {47.16:6.2f}%           | {best_val_top1:6.2f}%           | {'+' if best_val_top1 - 47.16 >= 0 else ''}{best_val_top1 - 47.16:+.2f}%")
    print(f"{'Training Duration':<25} | {'—':<12} | {'40.6 min':<18} | {f'{total_train_duration/60:.1f} min':<18} | {'—'}")
    print("=" * 80)


if __name__ == "__main__":
    main()
