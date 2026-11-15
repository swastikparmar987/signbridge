"""
Vectorized 476-D signer-invariant kinematic & handshape feature extractor (Torch & NumPy parity).

Feature breakdown per frame (476 dims total):
  - 126D: Palm-scale stabilized local coordinates for Hand 0 and Hand 1
  - 126D: Active-pair masked local hand velocities
  - 120D: Unit 3D bone direction vectors (20 bones x 3 x 2 hands)
  -  30D: Joint flexion cosines along each finger chain (15 x 2 hands)
  -  30D: Key normalized intra-hand distances (fingertip-wrist, thumb-tip, spread, palm closure x 2)
  -   6D: Unit 3D palm orientation normals (2 hands)
  -   6D: Inter-hand 3D wrist displacement + masked velocity (scale-normalized)
  -  24D: Clip-centered & scale-normalized global wrist trajectory (rel pos, 1f vel, 4f vel, acc x 2)
  -   8D: Active masks, wrist speeds, finger spreads, inter-hand distance
"""

from __future__ import annotations

from typing import Tuple
import numpy as np

try:
    import torch
except ImportError:
    torch = None

BONE_PARENTS = [
    0, 1, 2, 3,
    0, 5, 6, 7,
    0, 9, 10, 11,
    0, 13, 14, 15,
    0, 17, 18, 19,
]
BONE_CHILDREN = [
    1, 2, 3, 4,
    5, 6, 7, 8,
    9, 10, 11, 12,
    13, 14, 15, 16,
    17, 18, 19, 20,
]

# 15 sequential joint triplets (parent -> mid -> child) per hand
ANGLE_A = [0, 1, 2, 0, 5, 6, 0, 9, 10, 0, 13, 14, 0, 17, 18]
ANGLE_B = [1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19]
ANGLE_C = [2, 3, 4, 6, 7, 8, 10, 11, 12, 14, 15, 16, 18, 19, 20]

# 15 intra-hand distance landmark pairs
DIST_I = [4, 8, 12, 16, 20, 4, 4, 4, 4, 8, 12, 16, 8, 12, 20]
DIST_J = [0, 0, 0, 0, 0, 8, 12, 16, 20, 12, 16, 20, 9, 9, 9]

KINEMATIC_V2_DIM = 476


def _hand_features_torch(
    hand: torch.Tensor,
    active: torch.Tensor,
    eps: float = 1e-4,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute per-hand geometric and local kinematic features.
    hand: (B, T, 21, 3), active: (B, T, 1)
    Returns (local_xyz [B,T,63], local_vel [B,T,63], bone_dirs [B,T,60],
             flex_cos [B,T,15], dists [B,T,15], palm_normal [B,T,3], scale [B,T,1], wrist [B,T,3])
    """
    wrist = hand[:, :, 0:1, :]
    s_mid = torch.linalg.norm(hand[:, :, 9:10, :] - wrist, dim=-1, keepdim=True)
    s_idx = torch.linalg.norm(hand[:, :, 5:6, :] - wrist, dim=-1, keepdim=True)
    s_pnk = torch.linalg.norm(hand[:, :, 17:18, :] - wrist, dim=-1, keepdim=True)
    s_tri = torch.linalg.norm(hand[:, :, 5:6, :] - hand[:, :, 17:18, :], dim=-1, keepdim=True)
    scale = (0.4 * s_mid + 0.2 * s_idx + 0.2 * s_pnk + 0.2 * s_tri).clamp(min=eps)  # (B, T, 1, 1)

    mask4 = active.unsqueeze(-1)  # (B, T, 1, 1)
    mask3 = active  # (B, T, 1)

    loc = ((hand - wrist) / scale) * mask4
    loc_flat = loc.reshape(hand.shape[0], hand.shape[1], 63)

    vel_flat = torch.zeros_like(loc_flat)
    if hand.shape[1] > 1:
        pair_active = mask3[:, 1:, :] * mask3[:, :-1, :]
        vel_flat[:, 1:, :] = (loc_flat[:, 1:, :] - loc_flat[:, :-1, :]) * pair_active

    # Unit bone vectors (20 x 3 = 60)
    bones = loc[:, :, BONE_CHILDREN, :] - loc[:, :, BONE_PARENTS, :]
    bone_norms = torch.linalg.norm(bones, dim=-1, keepdim=True).clamp(min=eps)
    bone_dirs = (bones / bone_norms) * mask4
    bone_flat = bone_dirs.reshape(hand.shape[0], hand.shape[1], 60)

    # Joint flexion cosines (15)
    v1 = loc[:, :, ANGLE_B, :] - loc[:, :, ANGLE_A, :]
    v2 = loc[:, :, ANGLE_C, :] - loc[:, :, ANGLE_B, :]
    v1_n = v1 / torch.linalg.norm(v1, dim=-1, keepdim=True).clamp(min=eps)
    v2_n = v2 / torch.linalg.norm(v2, dim=-1, keepdim=True).clamp(min=eps)
    flex_cos = (v1_n * v2_n).sum(dim=-1) * mask3  # (B, T, 15)

    # Key intra-hand normalized distances (15)
    d_vec = loc[:, :, DIST_I, :] - loc[:, :, DIST_J, :]
    dists = torch.linalg.norm(d_vec, dim=-1) * mask3  # (B, T, 15)

    # Unit palm normal (3) via explicit elementwise cross product for fast MPS & ONNX portability
    p1 = hand[:, :, 5, :] - hand[:, :, 0, :]
    p2 = hand[:, :, 17, :] - hand[:, :, 0, :]
    p_cross = torch.stack(
        [
            p1[..., 1] * p2[..., 2] - p1[..., 2] * p2[..., 1],
            p1[..., 2] * p2[..., 0] - p1[..., 0] * p2[..., 2],
            p1[..., 0] * p2[..., 1] - p1[..., 1] * p2[..., 0],
        ],
        dim=-1,
    )
    p_norm = torch.linalg.norm(p_cross, dim=-1, keepdim=True).clamp(min=eps)
    palm_normal = (p_cross / p_norm) * mask3  # (B, T, 3)

    return loc_flat, vel_flat, bone_flat, flex_cos, dists, palm_normal, scale.squeeze(-1), wrist.squeeze(2)


def build_kinematic_v2_torch(raw_sequence: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """
    Build 476-D kinematic features purely in PyTorch for batched GPU/MPS training and CPU/ONNX parity.
    Accepts (B, T, 42, 3), (T, 42, 3), or (B, T, 126).
    Returns float32 tensor of shape (B, T, 476).
    """
    seq = torch.nan_to_num(raw_sequence.float(), nan=0.0, posinf=0.0, neginf=0.0)
    if seq.ndim == 2 and seq.shape[1] == 126:
        seq = seq.unsqueeze(0).reshape(1, seq.shape[0], 42, 3)
    elif seq.ndim == 3 and seq.shape[1:] == (42, 3):
        seq = seq.unsqueeze(0)
    elif seq.ndim == 3 and seq.shape[2] == 126:
        seq = seq.reshape(seq.shape[0], seq.shape[1], 42, 3)
    elif seq.ndim != 4 or seq.shape[2:] != (42, 3):
        raise ValueError(f"Expected shape (B, T, 42, 3), (T, 42, 3), or (B, T, 126), got {tuple(seq.shape)}")

    B, T, _, _ = seq.shape
    h0 = seq[:, :, :21, :]
    h1 = seq[:, :, 21:42, :]

    a0 = (h0.abs().sum(dim=(-2, -1), keepdim=False) > 1e-4).unsqueeze(-1).float()  # (B, T, 1)
    a1 = (h1.abs().sum(dim=(-2, -1), keepdim=False) > 1e-4).unsqueeze(-1).float()  # (B, T, 1)

    loc0, vel0, bone0, flex0, dist0, pn0, sc0, w0 = _hand_features_torch(h0, a0, eps=eps)
    loc1, vel1, bone1, flex1, dist1, pn1, sc1, w1 = _hand_features_torch(h1, a1, eps=eps)

    # Clip-wide scale & spatial anchor over active hands (translation + camera distance invariant)
    sc_num = (sc0 * a0).sum(dim=1, keepdim=True) + (sc1 * a1).sum(dim=1, keepdim=True)
    sc_den = (a0.sum(dim=1, keepdim=True) + a1.sum(dim=1, keepdim=True)).clamp(min=1.0)
    clip_scale = (sc_num / sc_den).clamp(min=0.02)  # (B, 1, 1)

    w_num = (w0 * a0).sum(dim=1, keepdim=True) + (w1 * a1).sum(dim=1, keepdim=True)
    clip_center = w_num / sc_den  # (B, 1, 3)

    # Inter-hand displacement & delta displacement (6D)
    both_active = a0 * a1  # (B, T, 1)
    mean_frame_scale = ((sc0 + sc1) * 0.5).clamp(min=eps)
    inter_disp = ((w1 - w0) / mean_frame_scale) * both_active
    inter_vel = torch.zeros_like(inter_disp)
    if T > 1:
        both_pair = both_active[:, 1:, :] * both_active[:, :-1, :]
        inter_vel[:, 1:, :] = (inter_disp[:, 1:, :] - inter_disp[:, :-1, :]) * both_pair

    # Global wrist trajectory per hand (12D each -> 24D total)
    traj_feats = []
    speeds = []
    for w, a in ((w0, a0), (w1, a1)):
        w_rel = ((w - clip_center) / clip_scale) * a
        w_vel = torch.zeros_like(w_rel)
        w_vel4 = torch.zeros_like(w_rel)
        w_acc = torch.zeros_like(w_rel)
        if T > 1:
            p1 = a[:, 1:, :] * a[:, :-1, :]
            w_vel[:, 1:, :] = (w_rel[:, 1:, :] - w_rel[:, :-1, :]) * p1
        if T > 2:
            p2 = a[:, 2:, :] * a[:, 1:-1, :] * a[:, :-2, :]
            w_acc[:, 2:, :] = (w_vel[:, 2:, :] - w_vel[:, 1:-1, :]) * p2
        if T > 4:
            p4 = a[:, 4:, :] * a[:, :-4, :]
            w_vel4[:, 4:, :] = ((w_rel[:, 4:, :] - w_rel[:, :-4, :]) * 0.25) * p4
        traj_feats.extend([w_rel, w_vel, w_vel4, w_acc])
        speeds.append(torch.linalg.norm(w_vel, dim=-1, keepdim=True))

    # Activity & spread scalars (8D)
    spread0 = dist0[:, :, :5].mean(dim=-1, keepdim=True) * a0
    spread1 = dist1[:, :, :5].mean(dim=-1, keepdim=True) * a1
    inter_dist = torch.linalg.norm(inter_disp, dim=-1, keepdim=True) * both_active
    scalars = torch.cat([a0, a1, both_active, speeds[0], speeds[1], spread0, spread1, inter_dist], dim=-1)

    features = torch.cat(
        [
            loc0, loc1,
            vel0, vel1,
            bone0, bone1,
            flex0, flex1,
            dist0, dist1,
            pn0, pn1,
            inter_disp, inter_vel,
            *traj_feats,
            scalars,
        ],
        dim=-1,
    )
    return torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)


def _hand_features_numpy(
    hand: np.ndarray,
    active: np.ndarray,
    eps: float = 1e-4,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Pure-NumPy counterpart of _hand_features_torch() matching float32 outputs for ONNX inference.
    hand: (B, T, 21, 3), active: (B, T, 1)
    """
    wrist = hand[:, :, 0:1, :]
    s_mid = np.linalg.norm(hand[:, :, 9:10, :] - wrist, axis=-1, keepdims=True)
    s_idx = np.linalg.norm(hand[:, :, 5:6, :] - wrist, axis=-1, keepdims=True)
    s_pnk = np.linalg.norm(hand[:, :, 17:18, :] - wrist, axis=-1, keepdims=True)
    s_tri = np.linalg.norm(hand[:, :, 5:6, :] - hand[:, :, 17:18, :], axis=-1, keepdims=True)
    scale = np.maximum(0.4 * s_mid + 0.2 * s_idx + 0.2 * s_pnk + 0.2 * s_tri, eps)

    mask4 = active[..., np.newaxis]
    mask3 = active

    loc = ((hand - wrist) / scale) * mask4
    loc_flat = loc.reshape(hand.shape[0], hand.shape[1], 63)

    vel_flat = np.zeros_like(loc_flat)
    if hand.shape[1] > 1:
        pair_active = mask3[:, 1:, :] * mask3[:, :-1, :]
        vel_flat[:, 1:, :] = (loc_flat[:, 1:, :] - loc_flat[:, :-1, :]) * pair_active

    bones = loc[:, :, BONE_CHILDREN, :] - loc[:, :, BONE_PARENTS, :]
    bone_norms = np.maximum(np.linalg.norm(bones, axis=-1, keepdims=True), eps)
    bone_dirs = (bones / bone_norms) * mask4
    bone_flat = bone_dirs.reshape(hand.shape[0], hand.shape[1], 60)

    v1 = loc[:, :, ANGLE_B, :] - loc[:, :, ANGLE_A, :]
    v2 = loc[:, :, ANGLE_C, :] - loc[:, :, ANGLE_B, :]
    v1_n = v1 / np.maximum(np.linalg.norm(v1, axis=-1, keepdims=True), eps)
    v2_n = v2 / np.maximum(np.linalg.norm(v2, axis=-1, keepdims=True), eps)
    flex_cos = (v1_n * v2_n).sum(axis=-1) * mask3

    d_vec = loc[:, :, DIST_I, :] - loc[:, :, DIST_J, :]
    dists = np.linalg.norm(d_vec, axis=-1) * mask3

    p1 = hand[:, :, 5, :] - hand[:, :, 0, :]
    p2 = hand[:, :, 17, :] - hand[:, :, 0, :]
    p_cross = np.stack(
        [
            p1[..., 1] * p2[..., 2] - p1[..., 2] * p2[..., 1],
            p1[..., 2] * p2[..., 0] - p1[..., 0] * p2[..., 2],
            p1[..., 0] * p2[..., 1] - p1[..., 1] * p2[..., 0],
        ],
        axis=-1,
    )
    p_norm = np.maximum(np.linalg.norm(p_cross, axis=-1, keepdims=True), eps)
    palm_normal = (p_cross / p_norm) * mask3

    return loc_flat, vel_flat, bone_flat, flex_cos, dists, palm_normal, scale.squeeze(-1), wrist.squeeze(2)


def build_kinematic_v2_numpy(raw_sequence: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    """
    Pure-NumPy 476-D kinematic feature builder matching build_kinematic_v2_torch() without importing torch.
    Accepts (B, T, 42, 3), (T, 42, 3), or (T, 126); returns float32 (B, T, 476).
    """
    seq = np.nan_to_num(np.asarray(raw_sequence, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if seq.ndim == 2 and seq.shape[1] == 126:
        seq = seq.reshape(1, seq.shape[0], 42, 3)
    elif seq.ndim == 3 and seq.shape[1:] == (42, 3):
        seq = seq[np.newaxis, ...]
    elif seq.ndim == 3 and seq.shape[2] == 126:
        seq = seq.reshape(seq.shape[0], seq.shape[1], 42, 3)
    elif seq.ndim != 4 or seq.shape[2:] != (42, 3):
        raise ValueError(f"Expected shape (B, T, 42, 3), (T, 42, 3), or (B, T, 126), got {seq.shape}")

    _, T, _, _ = seq.shape
    h0 = seq[:, :, :21, :]
    h1 = seq[:, :, 21:42, :]

    a0 = (np.abs(h0).sum(axis=(-2, -1)) > 1e-4)[..., np.newaxis].astype(np.float32)
    a1 = (np.abs(h1).sum(axis=(-2, -1)) > 1e-4)[..., np.newaxis].astype(np.float32)

    loc0, vel0, bone0, flex0, dist0, pn0, sc0, w0 = _hand_features_numpy(h0, a0, eps=eps)
    loc1, vel1, bone1, flex1, dist1, pn1, sc1, w1 = _hand_features_numpy(h1, a1, eps=eps)

    sc_num = (sc0 * a0).sum(axis=1, keepdims=True) + (sc1 * a1).sum(axis=1, keepdims=True)
    sc_den = np.maximum(a0.sum(axis=1, keepdims=True) + a1.sum(axis=1, keepdims=True), 1.0)
    clip_scale = np.maximum(sc_num / sc_den, 0.02)

    w_num = (w0 * a0).sum(axis=1, keepdims=True) + (w1 * a1).sum(axis=1, keepdims=True)
    clip_center = w_num / sc_den

    both_active = a0 * a1
    mean_frame_scale = np.maximum((sc0 + sc1) * 0.5, eps)
    inter_disp = ((w1 - w0) / mean_frame_scale) * both_active
    inter_vel = np.zeros_like(inter_disp)
    if T > 1:
        both_pair = both_active[:, 1:, :] * both_active[:, :-1, :]
        inter_vel[:, 1:, :] = (inter_disp[:, 1:, :] - inter_disp[:, :-1, :]) * both_pair

    traj_feats = []
    speeds = []
    for w, a in ((w0, a0), (w1, a1)):
        w_rel = ((w - clip_center) / clip_scale) * a
        w_vel = np.zeros_like(w_rel)
        w_vel4 = np.zeros_like(w_rel)
        w_acc = np.zeros_like(w_rel)
        if T > 1:
            p1 = a[:, 1:, :] * a[:, :-1, :]
            w_vel[:, 1:, :] = (w_rel[:, 1:, :] - w_rel[:, :-1, :]) * p1
        if T > 2:
            p2 = a[:, 2:, :] * a[:, 1:-1, :] * a[:, :-2, :]
            w_acc[:, 2:, :] = (w_vel[:, 2:, :] - w_vel[:, 1:-1, :]) * p2
        if T > 4:
            p4 = a[:, 4:, :] * a[:, :-4, :]
            w_vel4[:, 4:, :] = ((w_rel[:, 4:, :] - w_rel[:, :-4, :]) * 0.25) * p4
        traj_feats.extend([w_rel, w_vel, w_vel4, w_acc])
        speeds.append(np.linalg.norm(w_vel, axis=-1, keepdims=True))

    spread0 = dist0[:, :, :5].mean(axis=-1, keepdims=True) * a0
    spread1 = dist1[:, :, :5].mean(axis=-1, keepdims=True) * a1
    inter_dist = np.linalg.norm(inter_disp, axis=-1, keepdims=True) * both_active
    scalars = np.concatenate([a0, a1, both_active, speeds[0], speeds[1], spread0, spread1, inter_dist], axis=-1)

    features = np.concatenate(
        [
            loc0, loc1,
            vel0, vel1,
            bone0, bone1,
            flex0, flex1,
            dist0, dist1,
            pn0, pn1,
            inter_disp, inter_vel,
            *traj_feats,
            scalars,
        ],
        axis=-1,
    )
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def build_kinematic_v2_features(raw_sequence: np.ndarray) -> np.ndarray:
    """
    NumPy entrypoint matching build_kinematic_v2_torch() without requiring PyTorch.
    Accepts (T, 42, 3) or (T, 126); returns float32 (T, 476).
    """
    arr = np.asarray(raw_sequence, dtype=np.float32)
    return build_kinematic_v2_numpy(arr)[0].astype(np.float32)
