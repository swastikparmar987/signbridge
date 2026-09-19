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


class BiGRUAttentionPoolingClassifier(nn.Module):
    """
    2-Layer Bidirectional GRU with Triple Temporal Feature Aggregation:
    1. Learnable Additive Attention Pooling across all 32 frames [B, 512]
    2. Global Temporal Mean Pooling [B, 512]
    3. Global Temporal Max Pooling [B, 512]
    
    Fused Features: Concatenation of all 3 -> [B, 1536]
    Head: LayerNorm(1536) -> Dropout(0.3) -> Linear(1536, num_classes) -> [B, num_classes]
    """

    def __init__(
        self,
        input_size: int = 126,
        hidden_size: int = 256,
        num_layers: int = 2,
        num_classes: int = 2731,
        attention_hidden_dim: int = 128,
        dropout_p: float = 0.3,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.bidirectional = True
        self.gru_feature_dim = hidden_size * 2  # 512
        self.fused_dim = self.gru_feature_dim * 3  # 1536

        # 1. Bidirectional GRU Backbone
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )

        # 2. Learnable Additive Attention Pooling Network
        self.attention_net = nn.Sequential(
            nn.Linear(self.gru_feature_dim, attention_hidden_dim),
            nn.Tanh(),
            nn.Linear(attention_hidden_dim, 1),
        )

        # 3. Regularized Classification Head
        self.layer_norm = nn.LayerNorm(self.fused_dim)
        self.dropout = nn.Dropout(p=dropout_p)
        self.classifier = nn.Linear(self.fused_dim, num_classes)

        # Weight Initialization
        self._init_weights()

    def _init_weights(self):
        for name, param in self.gru.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

        for module in self.attention_net:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [Batch, 32, 126]
        gru_output, _ = self.gru(x)  # shape: [Batch, 32, 512]

        # 1. Attention Pooling
        attn_scores = self.attention_net(gru_output)  # shape: [Batch, 32, 1]
        attn_weights = torch.softmax(attn_scores, dim=1)  # shape: [Batch, 32, 1]
        attention_context = torch.sum(attn_weights * gru_output, dim=1)  # shape: [Batch, 512]

        # 2. Global Mean Pooling
        mean_context = torch.mean(gru_output, dim=1)  # shape: [Batch, 512]

        # 3. Global Max Pooling
        max_context = torch.max(gru_output, dim=1).values  # shape: [Batch, 512]

        # 4. Feature Fusion
        fused_features = torch.cat(
            [attention_context, mean_context, max_context], dim=1
        )  # shape: [Batch, 1536]

        # 5. Classification Head
        normed = self.layer_norm(fused_features)  # shape: [Batch, 1536]
        dropped = self.dropout(normed)  # shape: [Batch, 1536]
        logits = self.classifier(dropped)  # shape: [Batch, num_classes]

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
) -> Tuple[nn.Module, List[str], Dict[str, Any]]:
    """
    Load SignBridge model from checkpoint (supports both GRUClassifier and BiGRUAttentionPoolingClassifier).
    
    Args:
        checkpoint_path: Path to checkpoint .pth file
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
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    # Detect architecture type
    is_bigru_attention = (
        checkpoint.get("architecture") == "BiGRUAttentionPoolingClassifier"
        or ("attention_net.0.weight" in state_dict and checkpoint.get("architecture") != "BiGRUArcFaceClassifier")
    )
    is_arcface = checkpoint.get("architecture") == "BiGRUArcFaceClassifier"

    if is_arcface:
        attention_hidden_dim = checkpoint.get("attention_hidden_dim", 128)
        dropout_p = checkpoint.get("dropout_p", 0.3)
        s = checkpoint.get("s", 30.0)
        m = checkpoint.get("m", 0.50)
        model = BiGRUArcFaceClassifier(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_classes=num_classes,
            attention_hidden_dim=attention_hidden_dim,
            dropout_p=dropout_p,
            s=s,
            m=m,
        )
    elif is_bigru_attention:
        attention_hidden_dim = checkpoint.get("attention_hidden_dim", 128)
        dropout_p = checkpoint.get("dropout_p", 0.3)
        model = BiGRUAttentionPoolingClassifier(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_classes=num_classes,
            attention_hidden_dim=attention_hidden_dim,
            dropout_p=dropout_p,
        )
    else:
        model = GRUClassifier(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_classes=num_classes,
        )

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    metadata = {
        "architecture": model.__class__.__name__,
        "epoch": checkpoint.get("epoch"),
        "val_accuracy": checkpoint.get("val_accuracy"),
        "val_top1": checkpoint.get("val_top1"),
        "val_top5": checkpoint.get("val_top5"),
        "input_size": input_size,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "num_classes": num_classes,
        "device": device,
    }

    return model, class_names, metadata


class BiGRUArcFaceClassifier(nn.Module):
    """
    2-Layer Bidirectional GRU with Triple Temporal Feature Aggregation:
    Uses ArcFace (ArcMarginProduct) for the classification head.
    """
    def __init__(
        self,
        input_size: int = 126,
        hidden_size: int = 256,
        num_layers: int = 2,
        num_classes: int = 2731,
        attention_hidden_dim: int = 128,
        dropout_p: float = 0.3,
        s: float = 30.0,
        m: float = 0.50,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.bidirectional = True
        self.gru_feature_dim = hidden_size * 2
        self.fused_dim = self.gru_feature_dim * 3
        
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )
        
        self.attention_net = nn.Sequential(
            nn.Linear(self.gru_feature_dim, attention_hidden_dim),
            nn.Tanh(),
            nn.Linear(attention_hidden_dim, 1),
        )
        
        self.layer_norm = nn.LayerNorm(self.fused_dim)
        self.dropout = nn.Dropout(p=dropout_p)
        
        # In inference/model.py we might not want to import ArcMarginProduct directly 
        # to avoid circular deps or because it's in training. But since training imports 
        # from inference, we can import it here safely or just copy the linear layer for inference.
        # Actually, let's just make it a Linear layer for inference to avoid importing ArcFace here.
        # Wait, if we save the model, we need to load ArcFace weights.
        # The weight shape in ArcMarginProduct is (out_features, in_features).
        # We can just use nn.Linear(fused_dim, num_classes, bias=False) and normalize it in forward.
        self.classifier = nn.Linear(self.fused_dim, num_classes, bias=False)
        self.s = s

        self._init_weights()

    def _init_weights(self):
        for name, param in self.gru.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
        for module in self.attention_net:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.xavier_uniform_(self.classifier.weight)

    def forward(self, x: torch.Tensor, label=None) -> torch.Tensor:
        gru_output, _ = self.gru(x)
        attn_scores = self.attention_net(gru_output)
        attn_weights = torch.softmax(attn_scores, dim=1)
        attention_context = torch.sum(attn_weights * gru_output, dim=1)
        mean_context = torch.mean(gru_output, dim=1)
        max_context = torch.max(gru_output, dim=1).values
        fused_features = torch.cat([attention_context, mean_context, max_context], dim=1)
        normed = self.layer_norm(fused_features)
        dropped = self.dropout(normed)
        
        # Inference-only ArcFace forward (cosine similarity * s)
        import torch.nn.functional as F
        cosine = F.linear(F.normalize(dropped), F.normalize(self.classifier.weight))
        return cosine * self.s
