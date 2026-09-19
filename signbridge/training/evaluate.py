import json
from pathlib import Path
from typing import Dict, Any, Optional
import torch
from torch.utils.data import DataLoader

from signbridge.inference.model import GRUClassifier, load_signbridge_model, get_default_device
from signbridge.training.config import (
    load_demo_glosses,
    load_full_glosses,
    PRETRAINED_CHECKPOINT,
    get_device,
)
from signbridge.training.dataset import create_dataloaders, create_full_dataloaders, create_velocity_dataloaders
from signbridge.training.metrics import compute_comprehensive_metrics
from signbridge.preprocessing.normalize import normalize_sequence


def evaluate_50_model(
    checkpoint_path: Path | str,
    device: Optional[torch.device] = None,
    batch_size: int = 32,
) -> Dict[str, Any]:
    """
    Evaluate a 50-class trained model checkpoint on the untouched test split.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = device or get_device()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    class_names = checkpoint["class_names"]

    model = GRUClassifier(
        input_size=checkpoint.get("input_size", 126),
        hidden_size=checkpoint.get("hidden_size", 256),
        num_layers=checkpoint.get("num_layers", 2),
        num_classes=len(class_names),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    # Load untouched test split
    _, _, test_loader, _ = create_dataloaders(batch_size=batch_size)

    all_logits = []
    all_targets = []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            logits = model(x)
            all_logits.append(logits.cpu())
            all_targets.append(y)

    all_logits = torch.cat(all_logits, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    metrics = compute_comprehensive_metrics(all_logits, all_targets, class_names)
    metrics["checkpoint_path"] = str(checkpoint_path)
    metrics["val_top1_at_save"] = checkpoint.get("val_top1")
    metrics["epoch_at_save"] = checkpoint.get("epoch")

    return metrics


def evaluate_full_model(
    checkpoint_path: Path | str,
    device: Optional[torch.device] = None,
    batch_size: int = 256,
    input_size: int = 126,
) -> Dict[str, Any]:
    """
    Evaluate a full 2,731-class trained model checkpoint on the complete untouched test split (32,941 samples).
    Supports both GRUClassifier and BiGRUAttentionPoolingClassifier.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = device or get_device()
    model, class_names, metadata = load_signbridge_model(checkpoint_path, device=device)
    model.eval()

    # Check if model expects velocity features (input_size > 126)
    model_input_size = metadata.get("input_size", input_size)
    
    # Load untouched test split (32,941 samples) with appropriate features
    if model_input_size > 126:
        # Model expects velocity features
        from signbridge.training.dataset import create_velocity_dataloaders
        # We need to import the velocity dataloader function
        _, _, test_loader, _ = create_velocity_dataloaders(batch_size=batch_size, num_workers=0)
    else:
        _, _, test_loader, _ = create_full_dataloaders(batch_size=batch_size, num_workers=0)

    all_logits = []
    all_targets = []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            logits = model(x)
            all_logits.append(logits.cpu())
            all_targets.append(y)

    all_logits = torch.cat(all_logits, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    metrics = compute_comprehensive_metrics(all_logits, all_targets, class_names)
    metrics["checkpoint_path"] = str(checkpoint_path)
    metrics["val_top1_at_save"] = metadata.get("val_top1")
    metrics["epoch_at_save"] = metadata.get("epoch")
    metrics["architecture"] = metadata.get("architecture")

    return metrics



def evaluate_baseline_2731_on_demo_test(
    baseline_checkpoint: Optional[Path] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Evaluate the existing 2731-class baseline model on the exact same 50-class test subset.
    Computes both full 2731-space metrics and restricted 50-space metrics for complete rigor.
    """
    baseline_checkpoint = baseline_checkpoint or PRETRAINED_CHECKPOINT
    device = device or get_device()

    model, class_names_2731, meta = load_signbridge_model(baseline_checkpoint, device=device)
    model.eval()
    class_to_idx_2731 = {c: i for i, c in enumerate(class_names_2731)}

    demo_class_names = load_demo_glosses()
    demo_indices_in_2731 = [class_to_idx_2731[g] for g in demo_class_names]
    target_idx_to_demo_idx = {idx_2731: i for i, idx_2731 in enumerate(demo_indices_in_2731)}

    # Get test dataset
    _, _, test_loader, _ = create_dataloaders(batch_size=32)
    test_dataset = test_loader.dataset

    correct_top1_2731 = 0
    correct_top3_2731 = 0
    correct_top5_2731 = 0

    correct_top1_restricted = 0
    correct_top3_restricted = 0
    correct_top5_restricted = 0

    all_demo_logits = []
    all_targets = []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            logits_2731 = model(x) # (B, 2731)

            # 2731 top-k calculation
            for b in range(logits_2731.size(0)):
                target_demo_idx = y[b].item()
                target_gloss = demo_class_names[target_demo_idx]
                target_2731_idx = class_to_idx_2731[target_gloss]

                sample_logits = logits_2731[b]
                top5_preds = torch.topk(sample_logits, 5).indices.tolist()

                if top5_preds[0] == target_2731_idx:
                    correct_top1_2731 += 1
                if target_2731_idx in top5_preds[:3]:
                    correct_top3_2731 += 1
                if target_2731_idx in top5_preds[:5]:
                    correct_top5_2731 += 1

            # Restricted 50-space logits
            restricted_logits = logits_2731[:, demo_indices_in_2731]
            all_demo_logits.append(restricted_logits.cpu())
            all_targets.append(y)

    total_samples = len(test_dataset)
    all_demo_logits = torch.cat(all_demo_logits, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    restricted_metrics = compute_comprehensive_metrics(all_demo_logits, all_targets, demo_class_names)

    return {
        "model_name": "2731_baseline_gru",
        "total_samples": total_samples,
        "space_2731": {
            "top1_accuracy": (correct_top1_2731 / total_samples) * 100.0,
            "top3_accuracy": (correct_top3_2731 / total_samples) * 100.0,
            "top5_accuracy": (correct_top5_2731 / total_samples) * 100.0,
        },
        "space_50_restricted": restricted_metrics,
    }
