from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional
import torch
import torch.nn as nn


class GRUClassifier(nn.Module):
    def __init__(
        self,
        input_size: int = 126,
        hidden_size: int = 256,
        num_layers: int = 2,
        num_classes: int = 2731,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, sequence_length, features)
        gru_output, _ = self.gru(x)
        last_output = gru_output[:, -1, :]
        logits = self.classifier(last_output)
        return logits


def get_default_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def load_signbridge_model(
    checkpoint_path: Path | str,
    device: Optional[torch.device] = None,
) -> Tuple[GRUClassifier, List[str], Dict[str, Any]]:
    """
    Load SignBridge GRU model from checkpoint.
    
    Args:
        checkpoint_path: Path to best_gru_normalized.pth
        device: Target torch.device (defaults to auto MPS/CUDA/CPU selection)
        
    Returns:
        Tuple of (model, class_names, checkpoint_metadata)
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    if device is None:
        device = get_default_device()

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
    model = model.to(device)
    model.eval()

    metadata = {
        "epoch": checkpoint.get("epoch"),
        "val_accuracy": checkpoint.get("val_accuracy"),
        "input_size": input_size,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "num_classes": num_classes,
        "device": device,
    }

    return model, class_names, metadata
