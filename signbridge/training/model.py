from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import torch
import torch.nn as nn

from signbridge.inference.model import (
    GRUClassifier,
    BiGRUAttentionPoolingClassifier,
    get_default_device,
)


def create_scratch_model(
    input_size: int = 126,
    hidden_size: int = 256,
    num_layers: int = 2,
    num_classes: int = 50,
) -> GRUClassifier:
    """
    Create a fresh GRUClassifier with randomly initialized weights.
    Used for Experiment 1 (Scratch Baseline).
    """
    model = GRUClassifier(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=num_classes,
    )
    return model


def create_bigru_attention_model(
    input_size: int = 126,
    hidden_size: int = 256,
    num_layers: int = 2,
    num_classes: int = 2731,
    attention_hidden_dim: int = 128,
    dropout_p: float = 0.3,
) -> BiGRUAttentionPoolingClassifier:
    """
    Create a fresh BiGRUAttentionPoolingClassifier with properly initialized weights.
    Used for Experiment 2 (exp2_bigru_attention_pooling).
    """
    model = BiGRUAttentionPoolingClassifier(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=num_classes,
        attention_hidden_dim=attention_hidden_dim,
        dropout_p=dropout_p,
    )
    return model


def create_pretrained_model(
    checkpoint_path: Path | str,
    num_classes: int = 50,
    device: Optional[torch.device] = None,
) -> Tuple[GRUClassifier, Dict[str, Any]]:
    """
    Load pretrained 2731-class SignBridge model and transfer its GRU backbone weights.
    Replaces the 2731-class classifier head with a fresh 50-class head.
    
    Used for Experiment 2 (Transfer Learning) and Experiment 3 (Augmented Transfer Learning).
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Pretrained checkpoint not found: {checkpoint_path}")

    device = device or torch.device("cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    input_size = checkpoint.get("input_size", 126)
    hidden_size = checkpoint.get("hidden_size", 256)
    num_layers = checkpoint.get("num_layers", 2)

    # Initialize 50-class model
    model = GRUClassifier(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=num_classes,
    )

    # Extract model state dict
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    # Transfer only GRU parameters
    gru_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("gru."):
            gru_state_dict[k] = v

    missing_keys, unexpected_keys = model.load_state_dict(gru_state_dict, strict=False)
    
    # Initialize the new classifier layer with Xavier uniform
    nn.init.xavier_uniform_(model.classifier.weight)
    nn.init.zeros_(model.classifier.bias)

    meta = {
        "original_checkpoint": str(checkpoint_path),
        "transferred_gru_params": list(gru_state_dict.keys()),
        "missing_keys": missing_keys,
        "unexpected_keys": unexpected_keys,
        "input_size": input_size,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "num_classes": num_classes,
    }

    return model, meta


def load_full_pretrained_model(
    checkpoint_path: Path | str,
    device: Optional[torch.device] = None,
) -> Tuple[GRUClassifier, List[str], Dict[str, Any]]:
    """
    Load the complete 2,731-class pretrained SignBridge model for end-to-end fine-tuning.
    Preserves all backbone and classifier weights.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Pretrained checkpoint not found: {checkpoint_path}")

    device = device or torch.device("cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    input_size = checkpoint.get("input_size", 126)
    hidden_size = checkpoint.get("hidden_size", 256)
    num_layers = checkpoint.get("num_layers", 2)
    num_classes = checkpoint.get("num_classes", 2731)
    class_names = checkpoint["class_names"]

    model = GRUClassifier(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=num_classes,
    )

    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)

    meta = {
        "original_checkpoint": str(checkpoint_path),
        "epoch_at_start": checkpoint.get("epoch"),
        "val_accuracy_at_start": checkpoint.get("val_accuracy"),
        "input_size": input_size,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "num_classes": num_classes,
    }

    return model, class_names, meta



def freeze_gru(model: GRUClassifier) -> None:
    """Freeze all GRU parameters, leaving only the classifier head trainable."""
    for param in model.gru.parameters():
        param.requires_grad = False


def unfreeze_gru(model: GRUClassifier) -> None:
    """Unfreeze all GRU parameters for end-to-end fine-tuning."""
    for param in model.gru.parameters():
        param.requires_grad = True


def save_checkpoint(
    model: nn.Module,
    class_names: List[str],
    save_path: Path | str,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save model checkpoint with full metadata compatible with SignBridgePredictor.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    meta = metadata or {}
    checkpoint = {
        "architecture": model.__class__.__name__,
        "model_state_dict": model.state_dict(),
        "class_names": class_names,
        "input_size": getattr(model, "input_size", getattr(model.gru, "input_size", 126)),
        "hidden_size": getattr(model, "hidden_size", getattr(model.gru, "hidden_size", 256)),
        "num_layers": getattr(model, "num_layers", getattr(model.gru, "num_layers", 2)),
        "num_classes": len(class_names),
        **meta,
    }

    torch.save(checkpoint, save_path)

