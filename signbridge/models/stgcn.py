import math
from typing import List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

# MediaPipe Hand Topology: 21 joints per hand
# 0: Wrist
# 1-4: Thumb (CMC, MCP, IP, Tip)
# 5-8: Index (MCP, PIP, DIP, Tip)
# 9-12: Middle (MCP, PIP, DIP, Tip)
# 13-16: Ring (MCP, PIP, DIP, Tip)
# 17-20: Pinky (MCP, PIP, DIP, Tip)
SINGLE_HAND_EDGES: List[Tuple[int, int]] = [
    # Wrist to finger bases
    (0, 1), (0, 5), (0, 9), (0, 13), (0, 17),
    # Thumb chain
    (1, 2), (2, 3), (3, 4),
    # Index chain
    (5, 6), (6, 7), (7, 8),
    # Middle chain
    (9, 10), (10, 11), (11, 12),
    # Ring chain
    (13, 14), (14, 15), (15, 16),
    # Pinky chain
    (17, 18), (18, 19), (19, 20),
]


def build_hand_graph_adj(num_joints: int = 42) -> torch.Tensor:
    """
    Build symmetric normalized adjacency matrix A_norm for 42 joints (2 hands x 21 joints).
    A_norm = D^{-1/2} * (A + I) * D^{-1/2}
    """
    A = torch.zeros((num_joints, num_joints), dtype=torch.float32)

    # Hand 1 edges (0..20)
    for u, v in SINGLE_HAND_EDGES:
        A[u, v] = 1.0
        A[v, u] = 1.0

    # Hand 2 edges (21..41)
    for u, v in SINGLE_HAND_EDGES:
        A[u + 21, v + 21] = 1.0
        A[v + 21, u + 21] = 1.0

    # Add self-loops (A + I)
    A_self = A + torch.eye(num_joints, dtype=torch.float32)

    # Compute D^{-1/2}
    deg = torch.sum(A_self, dim=1)
    deg_inv_sqrt = torch.pow(deg, -0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
    D_inv_sqrt = torch.diag(deg_inv_sqrt)

    # A_norm = D^{-1/2} * A_self * D^{-1/2}
    A_norm = torch.mm(torch.mm(D_inv_sqrt, A_self), D_inv_sqrt)
    return A_norm


class SpatialGraphConv(nn.Module):
    """
    Spatial Graph Convolution layer over hand joints.
    Y = (X @ W) @ A_norm
    Input: (B, C_in, T, V)
    Output: (B, C_out, T, V)
    """

    def __init__(self, in_channels: int, out_channels: int, adj_matrix: torch.Tensor):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.register_buffer("A", adj_matrix)  # (V, V)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, C_in, T, V)
        # Apply 1x1 conv across channels -> (B, C_out, T, V)
        x_conv = self.conv(x)
        # Multiply by Adjacency matrix along node dimension V:
        # x_conv shape: (B, C_out, T, V), A shape: (V, V)
        out = torch.einsum("bctv,vw->bctw", x_conv, self.A)
        return out


class STGCNBlock(nn.Module):
    """
    Spatial-Temporal Graph Convolutional Block:
    1. Spatial Graph Convolution over V joints
    2. Temporal 1D Convolution over T frames
    3. BatchNorm, ReLU, Dropout, and Residual Connection
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        adj_matrix: torch.Tensor,
        temporal_kernel_size: int = 9,
        stride: int = 1,
        dropout_p: float = 0.1,
    ):
        super().__init__()
        padding = (temporal_kernel_size - 1) // 2

        # 1. Spatial Graph Conv
        self.gcn = SpatialGraphConv(in_channels, out_channels, adj_matrix)
        self.bn1 = nn.BatchNorm2d(out_channels)

        # 2. Temporal Conv
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=(temporal_kernel_size, 1),
                stride=(stride, 1),
                padding=(padding, 0),
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(p=dropout_p),
        )

        self.relu = nn.ReLU(inplace=True)

        # Residual connection
        if in_channels != out_channels or stride != 1:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=(1, 1),
                    stride=(stride, 1),
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.residual = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, C_in, T, V)
        res = self.residual(x)
        x_gcn = self.relu(self.bn1(self.gcn(x)))
        x_tcn = self.tcn(x_gcn)
        return self.relu(x_tcn + res)


class STGCNArcFaceClassifier(nn.Module):
    """
    Spatial-Temporal Graph Convolutional Network (ST-GCN) with ArcFace Classification Head.
    
    Backbone Channels: 3 -> 64 -> 64 -> 128 -> 256
    Pooling: Global Average Pooling across Temporal (T=32) and Joint (V=42) dimensions.
    Classification Head: LayerNorm -> Dropout(0.3) -> ArcFace Head.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 2731,
        num_joints: int = 42,
        dropout_p: float = 0.3,
        s: float = 30.0,
        m: float = 0.25,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.num_joints = num_joints
        self.s = s
        self.m = m

        adj_matrix = build_hand_graph_adj(num_joints)

        # ST-GCN Backbone
        self.block1 = STGCNBlock(in_channels, 64, adj_matrix)
        self.block2 = STGCNBlock(64, 64, adj_matrix)
        self.block3 = STGCNBlock(64, 128, adj_matrix)
        self.block4 = STGCNBlock(128, 256, adj_matrix)

        # Classification Head
        self.layer_norm = nn.LayerNorm(256)
        self.dropout = nn.Dropout(p=dropout_p)
        self.classifier = nn.Linear(256, num_classes, bias=False)

        # ArcFace parameters
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        nn.init.xavier_uniform_(self.classifier.weight)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, C=3, T=32, V=42)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)  # (B, 256, T, V)

        # Global Average Pooling over T and V
        pooled = torch.mean(x, dim=(2, 3))  # (B, 256)
        normed = self.layer_norm(pooled)
        return self.dropout(normed)

    def forward(self, x: torch.Tensor, label: Optional[torch.Tensor] = None) -> torch.Tensor:
        features = self.extract_features(x)
        cosine = F.linear(F.normalize(features), F.normalize(self.classifier.weight))
        if label is not None and self.m > 0:
            sine = torch.sqrt(torch.clamp(1.0 - cosine.pow(2), min=1e-6))
            phi = cosine * self.cos_m - sine * self.sin_m
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
            one_hot = torch.zeros_like(cosine)
            one_hot.scatter_(1, label.view(-1, 1).long(), 1.0)
            cosine = one_hot * phi + (1.0 - one_hot) * cosine
        return cosine * self.s
