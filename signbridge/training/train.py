import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from signbridge.inference.model import GRUClassifier, get_default_device
from signbridge.training.config import (
    ExperimentConfig,
    set_seed,
    get_device,
)
from signbridge.training.dataset import create_dataloaders, create_full_dataloaders
from signbridge.training.model import (
    create_scratch_model,
    create_pretrained_model,
    load_full_pretrained_model,
    freeze_gru,
    unfreeze_gru,
    save_checkpoint,
)
from signbridge.training.metrics import compute_topk_accuracies


def train_one_epoch(
    model: GRUClassifier,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float = 1.0,
) -> Dict[str, float]:
    """Train model for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total_samples = 0

    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)
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
    model: GRUClassifier,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """Validate model and compute Top-1, Top-3, and Top-5 accuracy."""
    model.eval()
    total_loss = 0.0
    all_logits = []
    all_targets = []

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
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


def run_training(config: ExperimentConfig) -> Dict[str, Any]:
    """
    Execute full training procedure for an experiment configuration.
    Supports two-phase training (frozen head warmup -> full unfreeze).
    """
    print("=" * 75)
    print(f"🚀 STARTING TRAINING: {config.name}")
    print(f"   Description: {config.description}")
    print("=" * 75)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(config.seed)
    device = get_device()
    print(f"   Target Device: {device}")

    # Create DataLoaders
    train_loader, val_loader, test_loader, class_names = create_dataloaders(
        batch_size=config.batch_size,
        augmentation_config=config.augmentation,
        seed=config.seed,
    )
    print(f"   Dataset: Train={len(train_loader.dataset)}, Val={len(val_loader.dataset)}, Test={len(test_loader.dataset)} samples")

    # Initialize Model
    if config.pretrained_checkpoint:
        print(f"   Loading pretrained weights from: {config.pretrained_checkpoint}")
        model, meta = create_pretrained_model(
            config.pretrained_checkpoint,
            num_classes=len(class_names),
            device=device,
        )
    else:
        print("   Initializing fresh model from scratch (no pretraining)...")
        model = create_scratch_model(num_classes=len(class_names))

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    history: List[Dict[str, Any]] = []
    best_val_top1 = -1.0
    best_epoch = -1
    epochs_no_improve = 0
    total_epoch_count = 0
    best_checkpoint_path = config.output_dir / "best_model.pth"

    # =========================================================================
    # PHASE 1: Classifier Head Warmup (Optional, for transfer learning)
    # =========================================================================
    if config.phase1_epochs > 0:
        print(f"\n--- PHASE 1: Head Warmup ({config.phase1_epochs} epochs, lr={config.phase1_lr}, frozen GRU) ---")
        if config.phase1_freeze_backbone:
            freeze_gru(model)

        optimizer1 = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config.phase1_lr,
            weight_decay=config.weight_decay,
        )

        for epoch in range(1, config.phase1_epochs + 1):
            total_epoch_count += 1
            t0 = time.time()
            train_metrics = train_one_epoch(
                model, train_loader, criterion, optimizer1, device, config.grad_clip
            )
            val_metrics = validate(model, val_loader, criterion, device)
            elapsed = time.time() - t0

            epoch_record = {
                "phase": 1,
                "epoch": total_epoch_count,
                "lr": config.phase1_lr,
                **train_metrics,
                **val_metrics,
                "elapsed_sec": round(elapsed, 2),
            }
            history.append(epoch_record)

            print(
                f"   [Phase 1] Ep {epoch}/{config.phase1_epochs} | "
                f"Train Loss: {train_metrics['train_loss']:.4f} Acc: {train_metrics['train_accuracy']:.2f}% | "
                f"Val Loss: {val_metrics['val_loss']:.4f} Top-1: {val_metrics['val_top1']:.2f}% Top-5: {val_metrics['val_top5']:.2f}% | "
                f"({elapsed:.1f}s)"
            )

            # Check for best checkpoint
            if val_metrics["val_top1"] > best_val_top1:
                best_val_top1 = val_metrics["val_top1"]
                best_epoch = total_epoch_count
                save_checkpoint(
                    model,
                    class_names,
                    best_checkpoint_path,
                    metadata={
                        "experiment": config.name,
                        "epoch": total_epoch_count,
                        "val_top1": best_val_top1,
                        "val_top5": val_metrics["val_top5"],
                    },
                )

    # =========================================================================
    # PHASE 2: Full End-to-End Training / Fine-Tuning
    # =========================================================================
    if config.phase2_epochs > 0:
        print(f"\n--- PHASE 2: Full Training ({config.phase2_epochs} epochs max, lr={config.phase2_lr}, unfrozen GRU) ---")
        unfreeze_gru(model)

        optimizer2 = torch.optim.Adam(
            model.parameters(),
            lr=config.phase2_lr,
            weight_decay=config.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer2,
            mode="max",
            factor=0.5,
            patience=3,
        )

        for epoch in range(1, config.phase2_epochs + 1):
            total_epoch_count += 1
            t0 = time.time()
            current_lr = optimizer2.param_groups[0]["lr"]

            train_metrics = train_one_epoch(
                model, train_loader, criterion, optimizer2, device, config.grad_clip
            )
            val_metrics = validate(model, val_loader, criterion, device)
            scheduler.step(val_metrics["val_top1"])
            elapsed = time.time() - t0

            epoch_record = {
                "phase": 2,
                "epoch": total_epoch_count,
                "lr": current_lr,
                **train_metrics,
                **val_metrics,
                "elapsed_sec": round(elapsed, 2),
            }
            history.append(epoch_record)

            is_best = val_metrics["val_top1"] > best_val_top1 + config.min_delta
            if is_best:
                best_val_top1 = val_metrics["val_top1"]
                best_epoch = total_epoch_count
                epochs_no_improve = 0
                save_checkpoint(
                    model,
                    class_names,
                    best_checkpoint_path,
                    metadata={
                        "experiment": config.name,
                        "epoch": total_epoch_count,
                        "val_top1": best_val_top1,
                        "val_top5": val_metrics["val_top5"],
                    },
                )
            else:
                epochs_no_improve += 1

            best_marker = " 🌟 [BEST]" if is_best else ""
            print(
                f"   [Phase 2] Ep {epoch}/{config.phase2_epochs} (Total {total_epoch_count}) | "
                f"Train Loss: {train_metrics['train_loss']:.4f} Acc: {train_metrics['train_accuracy']:.2f}% | "
                f"Val Loss: {val_metrics['val_loss']:.4f} Top-1: {val_metrics['val_top1']:.2f}% Top-5: {val_metrics['val_top5']:.2f}% | "
                f"lr: {current_lr:.1e}{best_marker}"
            )

            # Early stopping check
            if epochs_no_improve >= config.patience:
                print(f"\n   ⏹️ Early stopping triggered after {epochs_no_improve} epochs with no validation improvement.")
                break

    # Save training history
    history_file = config.output_dir / "training_history.json"
    with open(history_file, "w") as f:
        json.dump(
            {
                "experiment": config.name,
                "description": config.description,
                "best_epoch": best_epoch,
                "best_val_top1": best_val_top1,
                "history": history,
            },
            f,
            indent=2,
        )

    print(f"\n✅ Training complete for {config.name}.")
    print(f"   Best Val Top-1: {best_val_top1:.2f}% at epoch {best_epoch}")
    print(f"   Best checkpoint: {best_checkpoint_path}")
    print(f"   History saved: {history_file}\n")

    return {
        "experiment": config.name,
        "best_epoch": best_epoch,
        "best_val_top1": best_val_top1,
        "checkpoint_path": str(best_checkpoint_path),
        "history_file": str(history_file),
    }


def run_full_training(config: ExperimentConfig) -> Dict[str, Any]:
    """
    Execute full 2,731-class training / fine-tuning procedure.
    """
    print("=" * 75)
    print(f"🚀 STARTING FULL 2731-CLASS TRAINING: {config.name}")
    print(f"   Description: {config.description}")
    print("=" * 75)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(config.seed)
    device = get_device()
    print(f"   Target Device: {device}")

    # Create Full DataLoaders
    train_loader, val_loader, test_loader, class_names = create_full_dataloaders(
        batch_size=config.batch_size,
        augmentation_config=config.augmentation,
        seed=config.seed,
    )
    print(f"   Dataset: Train={len(train_loader.dataset)}, Val={len(val_loader.dataset)}, Test={len(test_loader.dataset)} samples across {len(class_names)} classes")

    # Load Full Pretrained Model
    if config.pretrained_checkpoint:
        print(f"   Loading 2731-class pretrained weights from: {config.pretrained_checkpoint}")
        model, class_names, meta = load_full_pretrained_model(
            config.pretrained_checkpoint,
            device=device,
        )
    else:
        print("   Initializing fresh 2731-class model from scratch...")
        model = create_scratch_model(num_classes=len(class_names))

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    # Initial validation before training
    print("\n--- BASELINE VALIDATION (Epoch 0) ---")
    t0 = time.time()
    val_init = validate(model, val_loader, criterion, device)
    init_elapsed = time.time() - t0
    print(
        f"   [Baseline] Val Loss: {val_init['val_loss']:.4f} | "
        f"Top-1: {val_init['val_top1']:.2f}% | Top-3: {val_init['val_top3']:.2f}% | Top-5: {val_init['val_top5']:.2f}% | "
        f"({init_elapsed:.1f}s)"
    )

    history: List[Dict[str, Any]] = [{
        "phase": 2,
        "epoch": 0,
        "lr": config.phase2_lr,
        "train_loss": None,
        "train_accuracy": None,
        **val_init,
        "elapsed_sec": round(init_elapsed, 2),
    }]

    best_val_top1 = val_init["val_top1"]
    best_epoch = 0
    epochs_no_improve = 0
    best_checkpoint_path = config.output_dir / "best_model.pth"

    # Save initial best checkpoint
    save_checkpoint(
        model,
        class_names,
        best_checkpoint_path,
        metadata={
            "experiment": config.name,
            "epoch": 0,
            "val_top1": best_val_top1,
            "val_top5": val_init["val_top5"],
        },
    )

    # Optimizer & Scheduler
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.phase2_lr,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
    )

    print(f"\n--- FINE-TUNING ({config.phase2_epochs} epochs max, lr={config.phase2_lr}) ---")
    for epoch in range(1, config.phase2_epochs + 1):
        t0 = time.time()
        current_lr = optimizer.param_groups[0]["lr"]

        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, config.grad_clip
        )
        val_metrics = validate(model, val_loader, criterion, device)
        scheduler.step(val_metrics["val_top1"])
        elapsed = time.time() - t0

        epoch_record = {
            "phase": 2,
            "epoch": epoch,
            "lr": current_lr,
            **train_metrics,
            **val_metrics,
            "elapsed_sec": round(elapsed, 2),
        }
        history.append(epoch_record)

        is_best = val_metrics["val_top1"] > best_val_top1 + config.min_delta
        if is_best:
            best_val_top1 = val_metrics["val_top1"]
            best_epoch = epoch
            epochs_no_improve = 0
            save_checkpoint(
                model,
                class_names,
                best_checkpoint_path,
                metadata={
                    "experiment": config.name,
                    "epoch": epoch,
                    "val_top1": best_val_top1,
                    "val_top5": val_metrics["val_top5"],
                },
            )
        else:
            epochs_no_improve += 1

        best_marker = " 🌟 [BEST]" if is_best else ""
        print(
            f"   Ep {epoch:2d}/{config.phase2_epochs} | "
            f"Train Loss: {train_metrics['train_loss']:.4f} Acc: {train_metrics['train_accuracy']:.2f}% | "
            f"Val Loss: {val_metrics['val_loss']:.4f} Top-1: {val_metrics['val_top1']:.2f}% Top-5: {val_metrics['val_top5']:.2f}% | "
            f"lr: {current_lr:.1e} ({elapsed:.1f}s){best_marker}"
        )

        if epochs_no_improve >= config.patience:
            print(f"\n   ⏹️ Early stopping triggered after {epochs_no_improve} epochs with no validation improvement.")
            break

    # Save training history and configuration
    history_file = config.output_dir / "training_history.json"
    with open(history_file, "w") as f:
        json.dump(
            {
                "experiment": config.name,
                "description": config.description,
                "best_epoch": best_epoch,
                "best_val_top1": best_val_top1,
                "history": history,
            },
            f,
            indent=2,
        )

    config_file = config.output_dir / "config.json"
    with open(config_file, "w") as f:
        json.dump(
            {
                "name": config.name,
                "description": config.description,
                "pretrained_checkpoint": str(config.pretrained_checkpoint),
                "batch_size": config.batch_size,
                "seed": config.seed,
                "phase2_epochs": config.phase2_epochs,
                "phase2_lr": config.phase2_lr,
                "weight_decay": config.weight_decay,
                "grad_clip": config.grad_clip,
                "patience": config.patience,
                "augmentation": {
                    "enabled": config.augmentation.enabled,
                    "noise_sigma": config.augmentation.noise_sigma,
                    "spatial_scale_range": config.augmentation.spatial_scale_range,
                    "temporal_speed_range": config.augmentation.temporal_speed_range,
                    "temporal_jitter_frames": config.augmentation.temporal_jitter_frames,
                },
            },
            f,
            indent=2,
        )

    print(f"\n✅ Full training complete for {config.name}.")
    print(f"   Best Val Top-1: {best_val_top1:.2f}% at epoch {best_epoch}")
    print(f"   Best checkpoint: {best_checkpoint_path}")
    print(f"   History saved: {history_file}\n")

    return {
        "experiment": config.name,
        "best_epoch": best_epoch,
        "best_val_top1": best_val_top1,
        "checkpoint_path": str(best_checkpoint_path),
        "history_file": str(history_file),
    }

