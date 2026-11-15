"""
Modern pose-kinematic sequence architectures for 2,731-class ASL recognition.

Designed cleanly for:
  1. Apple Silicon M2 Pro MPS throughput + pure ONNX opset-17 exportability.
  2. Strict invariance to sparse ASL Citizen active spans vs dense live webcam windows
     via explicit active-frame masked pooling.
  3. Sub-center ArcFace (K sub-centers per gloss) to natively model left/right-handed
     and signer polymorphism, plus an auxiliary binary sign-activity head for OOD gating.
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.empty(shape, dtype=x.dtype, device=x.device).bernoulli_(keep)
        return x * mask / keep


class SqueezeExcite1d(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(16, channels // reduction)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C), mask: (B, T, 1)
        denom = mask.sum(dim=1).clamp(min=1.0)
        pooled = (x * mask).sum(dim=1) / denom  # (B, C)
        scale = torch.sigmoid(self.fc2(F.silu(self.fc1(pooled)))).unsqueeze(1)
        return x * scale


class MultiScaleConvStem(nn.Module):
    """Parallel multi-receptive-field 1D temporal stem (k=1, 3, 5, 7) with BatchNorm normalization."""

    def __init__(self, in_dim: int, d_model: int, dropout: float = 0.15):
        super().__init__()
        self.in_bn = nn.BatchNorm1d(in_dim)
        branch_dim = d_model // 4
        assert branch_dim * 4 == d_model, "d_model must be divisible by 4"

        self.b1 = nn.Conv1d(in_dim, branch_dim, kernel_size=1)
        self.b3 = nn.Conv1d(in_dim, branch_dim, kernel_size=3, padding=1)
        self.b5 = nn.Conv1d(in_dim, branch_dim, kernel_size=5, padding=2)
        self.b7 = nn.Conv1d(in_dim, branch_dim, kernel_size=7, padding=3)
        self.norm = nn.LayerNorm(d_model)
        self.se = SqueezeExcite1d(d_model, reduction=8)
        self.proj = nn.Sequential(
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, x: torch.Tensor, active_mask: torch.Tensor) -> torch.Tensor:
        # x: (B, T, in_dim)
        xc = self.in_bn(x.transpose(1, 2))
        h = torch.cat([self.b1(xc), self.b3(xc), self.b5(xc), self.b7(xc)], dim=1).transpose(1, 2)
        h = self.norm(h)
        h = self.se(h, active_mask)
        return self.proj(h)


class ConformerBlock(nn.Module):
    """Macaron-style Conformer block with explicit ONNX-clean MHSA and depthwise temporal conv."""

    def __init__(
        self,
        d_model: int,
        num_heads: int = 6,
        conv_kernel: int = 7,
        ff_mult: float = 2.0,
        dropout: float = 0.15,
        drop_path: float = 0.08,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        assert self.head_dim * num_heads == d_model

        hidden_ff = int(d_model * ff_mult)

        # 1. Half-step FFN
        self.ff1_norm = nn.LayerNorm(d_model)
        self.ff1 = nn.Sequential(
            nn.Linear(d_model, hidden_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_ff, d_model),
            nn.Dropout(dropout),
        )
        self.dp1 = DropPath(drop_path)

        # 2. Multi-Head Self-Attention (explicit matmul for MPS speed & ONNX opset portability)
        self.attn_norm = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, d_model * 3, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.attn_out = nn.Sequential(nn.Linear(d_model, d_model), nn.Dropout(dropout))
        self.dp2 = DropPath(drop_path)

        # 3. Depthwise Separable Temporal Conv Module
        self.conv_norm = nn.LayerNorm(d_model)
        self.pw_in = nn.Conv1d(d_model, d_model * 2, kernel_size=1)
        self.dw = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=conv_kernel,
            padding=conv_kernel // 2,
            groups=d_model,
        )
        self.conv_bn = nn.BatchNorm1d(d_model)
        self.pw_out = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.conv_drop = nn.Dropout(dropout)
        self.se = SqueezeExcite1d(d_model, reduction=8)
        self.dp3 = DropPath(drop_path)

        # 4. Closing Half-step FFN + final norm
        self.ff2_norm = nn.LayerNorm(d_model)
        self.ff2 = nn.Sequential(
            nn.Linear(d_model, hidden_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_ff, d_model),
            nn.Dropout(dropout),
        )
        self.dp4 = DropPath(drop_path)
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, active_mask: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape

        # Half FFN 1
        x = x + 0.5 * self.dp1(self.ff1(self.ff1_norm(x)))

        # MHSA
        xn = self.attn_norm(x)
        qkv = self.qkv(xn).reshape(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, H, T, dh)
        scores = torch.matmul(q, k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = self.attn_drop(torch.softmax(scores, dim=-1))
        ctx = torch.matmul(attn, v).transpose(1, 2).reshape(B, T, D)
        x = x + self.dp2(self.attn_out(ctx))

        # Conv module
        xc = self.pw_in(self.conv_norm(x).transpose(1, 2))
        a_gate, b_gate = xc.chunk(2, dim=1)
        xc = a_gate * torch.sigmoid(b_gate)  # GLU
        xc = F.silu(self.conv_bn(self.dw(xc)))
        xc = self.conv_drop(self.pw_out(xc)).transpose(1, 2)
        xc = self.se(xc, active_mask)
        x = x + self.dp3(xc)

        # Half FFN 2
        x = x + 0.5 * self.dp4(self.ff2(self.ff2_norm(x)))
        return self.final_norm(x)


class ActiveTemporalPooling(nn.Module):
    """
    Pools variable-active-length temporal features into a fixed embedding vector:
      - Active-masked global mean
      - Active-masked global max
      - M-head active-masked additive attention pooling
    Strictly prevents zero-padding dilution between ASL Citizen clips (~9 active frames)
    and continuous/live webcam windows (24-32 active frames).
    """

    def __init__(self, d_model: int, emb_dim: int = 512, num_pool_heads: int = 4, dropout: float = 0.25):
        super().__init__()
        self.num_pool_heads = num_pool_heads
        self.attn_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, num_pool_heads),
        )
        fused_dim = d_model * (2 + num_pool_heads)
        self.head = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Dropout(dropout),
            nn.Linear(fused_dim, emb_dim),
            nn.LayerNorm(emb_dim),
        )

    def forward(self, h: torch.Tensor, active_mask: torch.Tensor) -> torch.Tensor:
        # h: (B, T, D), active_mask: (B, T, 1) with values in {0, 1}
        # Fallback if a clip is completely empty -> treat all frames as weight 1/T so no NaN
        has_any = (active_mask.sum(dim=1, keepdim=True) > 0.5).float()
        w_mask = active_mask * has_any + (1.0 - has_any)  # (B, T, 1)

        # Masked Mean
        mean_pool = (h * w_mask).sum(dim=1) / w_mask.sum(dim=1).clamp(min=1.0)

        # Masked Max
        huge_neg = (1.0 - w_mask) * (-1e4)
        max_pool = (h + huge_neg).max(dim=1).values

        # Multi-head Attentive Pool
        attn_logits = self.attn_proj(h) + huge_neg  # (B, T, M)
        attn_weights = torch.softmax(attn_logits, dim=1)  # (B, T, M)
        attn_pools = torch.bmm(attn_weights.transpose(1, 2), h).reshape(h.shape[0], -1)  # (B, M*D)

        fused = torch.cat([mean_pool, max_pool, attn_pools], dim=-1)
        return self.head(fused)


class SubCenterCosineHead(nn.Module):
    """Sub-center normalized cosine classifier supporting ArcFace / CosFace margin training."""

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        num_subcenters: int = 2,
        s: float = 32.0,
        m: float = 0.30,
    ):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.num_subcenters = num_subcenters
        self.s = float(s)
        self.m = float(m)
        self.weight = nn.Parameter(torch.empty(num_classes * num_subcenters, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m

    def forward(self, emb: torch.Tensor, label: Optional[torch.Tensor] = None) -> torch.Tensor:
        norm_emb = F.normalize(emb, p=2, dim=-1)
        norm_w = F.normalize(self.weight, p=2, dim=-1)
        cos_all = F.linear(norm_emb, norm_w)  # (B, C * K)
        if self.num_subcenters > 1:
            cos_all = cos_all.view(emb.shape[0], self.num_classes, self.num_subcenters)
            cosine = cos_all.max(dim=-1).values
        else:
            cosine = cos_all

        if label is not None and self.m > 0.0:
            cos_clamped = cosine.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
            sine = torch.sqrt(torch.clamp(1.0 - cos_clamped * cos_clamped, min=1e-6))
            phi = cos_clamped * self.cos_m - sine * self.sin_m
            phi = torch.where(cos_clamped > self.th, phi, cos_clamped - self.mm)
            one_hot = torch.zeros_like(cos_clamped)
            one_hot.scatter_(1, label.view(-1, 1).long(), 1.0)
            cosine = one_hot * phi + (1.0 - one_hot) * cos_clamped

        return cosine * self.s


class _BasePoseSignModel(nn.Module):
    def _extract_active_mask(self, x: torch.Tensor) -> torch.Tensor:
        # In 476D kinematic_v2 features, indices -8 (468) and -7 (469) are h0_active and h1_active.
        if x.shape[-1] == 476:
            return ((x[:, :, -8] + x[:, :, -7]) > 0.5).float().unsqueeze(-1)
        return (x.abs().sum(dim=-1, keepdim=True) > 1e-4).float()

    def encode_sequence(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def extract_embedding(self, x: torch.Tensor) -> torch.Tensor:
        emb, _ = self.encode_sequence(x)
        return F.normalize(emb, p=2, dim=-1)

    def predict_ood_logit(self, x: torch.Tensor) -> torch.Tensor:
        _, ood_logit = self.encode_sequence(x)
        return ood_logit

    def forward_with_aux(
        self, x: torch.Tensor, label: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        emb, ood_logit = self.encode_sequence(x)
        logits = self.classifier(emb, label=label)
        return logits, ood_logit, F.normalize(emb, p=2, dim=-1)

    def forward(self, x: torch.Tensor, label: Optional[torch.Tensor] = None) -> torch.Tensor:
        emb, _ = self.encode_sequence(x)
        return self.classifier(emb, label=label)


class SignConformerClassifier(_BasePoseSignModel):
    """Multi-Scale 1D Conv Stem + 4-Block Conformer + Active-Masked Pooling + SubCenter ArcFace."""

    def __init__(
        self,
        input_size: int = 476,
        d_model: int = 384,
        num_layers: int = 4,
        num_heads: int = 6,
        num_classes: int = 2731,
        emb_dim: int = 512,
        num_subcenters: int = 2,
        dropout_p: float = 0.2,
        drop_path: float = 0.08,
        s: float = 32.0,
        m: float = 0.30,
        max_len: int = 64,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = d_model
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.emb_dim = emb_dim
        self.num_subcenters = num_subcenters
        self.dropout_p = dropout_p
        self.s = s
        self.m = m

        self.stem = MultiScaleConvStem(input_size, d_model, dropout=dropout_p)
        self.pos_emb = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        self.blocks = nn.ModuleList(
            [
                ConformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    conv_kernel=7,
                    ff_mult=2.0,
                    dropout=dropout_p,
                    drop_path=drop_path * (i + 1) / max(1, num_layers),
                )
                for i in range(num_layers)
            ]
        )
        self.pool = ActiveTemporalPooling(d_model, emb_dim=emb_dim, num_pool_heads=4, dropout=dropout_p)
        self.ood_head = nn.Linear(emb_dim, 1)
        self.classifier = SubCenterCosineHead(
            emb_dim, num_classes, num_subcenters=num_subcenters, s=s, m=m
        )

    def encode_sequence(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        active_mask = self._extract_active_mask(x)
        T = x.shape[1]
        h = self.stem(x, active_mask) + self.pos_emb[:, :T, :]
        for blk in self.blocks:
            h = blk(h, active_mask)
        emb = self.pool(h, active_mask)
        ood_logit = self.ood_head(emb).squeeze(-1)
        return emb, ood_logit


class SignPyramidBiGRUClassifier(_BasePoseSignModel):
    """Multi-Scale Conv Stem + 3-Layer Residual BiGRU + Active-Masked Pooling + SubCenter ArcFace."""

    def __init__(
        self,
        input_size: int = 476,
        d_model: int = 384,
        hidden_size: int = 320,
        num_layers: int = 3,
        num_classes: int = 2731,
        emb_dim: int = 512,
        num_subcenters: int = 2,
        dropout_p: float = 0.25,
        s: float = 32.0,
        m: float = 0.30,
    ):
        super().__init__()
        self.input_size = input_size
        self.d_model = d_model
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.emb_dim = emb_dim
        self.num_subcenters = num_subcenters
        self.dropout_p = dropout_p
        self.s = s
        self.m = m

        self.stem = MultiScaleConvStem(input_size, d_model, dropout=dropout_p * 0.6)
        self.gru_layers = nn.ModuleList()
        self.gru_norms = nn.ModuleList()
        self.gru_drops = nn.ModuleList()
        bi_dim = hidden_size * 2

        for i in range(num_layers):
            in_d = d_model if i == 0 else bi_dim
            self.gru_layers.append(nn.GRU(in_d, hidden_size, num_layers=1, batch_first=True, bidirectional=True))
            self.gru_norms.append(nn.LayerNorm(bi_dim))
            self.gru_drops.append(nn.Dropout(dropout_p))

        self.skip_proj = nn.Linear(d_model, bi_dim) if d_model != bi_dim else nn.Identity()
        self.pool = ActiveTemporalPooling(bi_dim, emb_dim=emb_dim, num_pool_heads=4, dropout=dropout_p)
        self.ood_head = nn.Linear(emb_dim, 1)
        self.classifier = SubCenterCosineHead(
            emb_dim, num_classes, num_subcenters=num_subcenters, s=s, m=m
        )

    def encode_sequence(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        active_mask = self._extract_active_mask(x)
        h = self.stem(x, active_mask)
        res_base = self.skip_proj(h)
        for i, (gru, norm, drop) in enumerate(zip(self.gru_layers, self.gru_norms, self.gru_drops)):
            out, _ = gru(h)
            out = drop(norm(out))
            h = (out + res_base) if i == 0 else (out + h)
        emb = self.pool(h, active_mask)
        ood_logit = self.ood_head(emb).squeeze(-1)
        return emb, ood_logit
