"""
Vectorized PyTorch batch augmentation and synthetic OOD negative generator on MPS/CPU.
All geometric perturbations operate directly on raw (B, 32, 42, 3) camera landmarks BEFORE
feature extraction so zero-frames and biomechanical structure remain strictly consistent.
"""

from typing import Tuple
import torch

FINGER_CHAINS = (
    (0, 1, 2, 3, 4),
    (0, 5, 6, 7, 8),
    (0, 9, 10, 11, 12),
    (0, 13, 14, 15, 16),
    (0, 17, 18, 19, 20),
)


@torch.no_grad()
def augment_landmark_batch(
    raw_batch: torch.Tensor,
    p_mirror: float = 0.28,
    p_bone_warp: float = 0.75,
    p_rotate: float = 0.75,
    p_speed_crop: float = 0.85,
    p_frame_drop: float = 0.45,
    p_hold_extend: float = 0.35,
    noise_std: float = 0.0025,
) -> torch.Tensor:
    """
    Apply vectorized domain augmentations on a batch of raw landmarks (B, 32, 42, 3).
    """
    seq = raw_batch.clone()
    B, T, _, _ = seq.shape
    device = seq.device
    dtype = seq.dtype

    # 1. Temporal Speed & Webcam Window Offset/Crop Simulation
    #    Randomly stretch/compress around active region and optionally shift within [0, 31]
    if p_speed_crop > 0:
        apply_t = torch.rand(B, device=device) < p_speed_crop
        speed = torch.where(
            apply_t,
            torch.empty(B, device=device).uniform_(0.72, 1.35),
            torch.ones(B, device=device),
        ).view(B, 1)
        shift = torch.where(
            apply_t,
            torch.empty(B, device=device).uniform_(-4.0, 4.0),
            torch.zeros(B, device=device),
        ).view(B, 1)
        t_grid = torch.arange(T, device=device, dtype=dtype).view(1, T)
        src_t = 15.5 + (t_grid - 15.5) * speed + shift
        idx = torch.round(src_t).long().clamp(0, T - 1)  # (B, T)
        idx_exp = idx.view(B, T, 1, 1).expand(-1, -1, 42, 3)
        seq = torch.gather(seq, 1, idx_exp)

    # Active point mask (strictly preserves zero frames)
    pt_active = seq.abs().sum(dim=-1, keepdim=True) > 1e-5  # (B, T, 42, 1)
    hand_active = torch.stack(
        [
            seq[:, :, :21, :].abs().sum(dim=(-2, -1)) > 1e-4,
            seq[:, :, 21:42, :].abs().sum(dim=(-2, -1)) > 1e-4,
        ],
        dim=-1,
    )  # (B, T, 2)

    # 2. Webcam-density adaptation: vectorize stretching active span outward [first_act..last_act] -> 32
    #    so ASL Citizen clips with 70% zero margins resemble dense webcam signing windows
    if p_hold_extend > 0:
        apply_hold = torch.rand(B, device=device) < p_hold_extend
        if torch.any(apply_hold):
            any_act = hand_active.any(dim=-1)  # (B, T)
            t_idx = torch.arange(T, device=device).view(1, T).expand(B, T)
            first_act = torch.where(any_act, t_idx, T).min(dim=1).values
            last_act = torch.where(any_act, t_idx, -1).max(dim=1).values
            valid = apply_hold & (last_act > first_act + 3)
            if torch.any(valid):
                u = torch.linspace(0.0, 1.0, T, device=device, dtype=dtype).view(1, T)
                src_h = torch.round(
                    first_act[:, None].float() + u * (last_act - first_act).clamp(min=1)[:, None].float()
                ).long().clamp(0, T - 1)
                seq_held = torch.gather(seq, 1, src_h.view(B, T, 1, 1).expand(-1, -1, 42, 3))
                seq = torch.where(valid.view(B, 1, 1, 1), seq_held, seq)
                pt_active = seq.abs().sum(dim=-1, keepdim=True) > 1e-5

    # 3. Horizontal Mirror Augmentation (x -> 1 - x + optional left/right slot swap)
    #    Directly fixes the -33% accuracy cliff on mirrored webcam / left-handed signers
    if p_mirror > 0:
        do_m = (torch.rand(B, device=device) < p_mirror).view(B, 1, 1, 1)
        if torch.any(do_m):
            mirrored = seq.clone()
            nz = mirrored.abs().sum(dim=-1) > 1e-5
            x_col = mirrored[..., 0]
            mirrored[..., 0] = torch.where(nz, 1.0 - x_col, x_col)
            # Keep wrist-x ordering consistent with landmark extractor (swap slots)
            mirrored = torch.cat([mirrored[:, :, 21:42, :], mirrored[:, :, :21, :]], dim=2)
            seq = torch.where(do_m, mirrored, seq)
            pt_active = seq.abs().sum(dim=-1, keepdim=True) > 1e-5

    # 4. Kinematic Bone Length Warp per hand segment (0.86x - 1.14x)
    if p_bone_warp > 0:
        do_w = torch.rand(B, device=device) < p_bone_warp
        if torch.any(do_w):
            for offset in (0, 21):
                for chain in FINGER_CHAINS:
                    for s_idx in range(len(chain) - 1):
                        p_j = offset + chain[s_idx]
                        c_j = offset + chain[s_idx + 1]
                        scale = torch.where(
                            do_w,
                            torch.empty(B, device=device).uniform_(0.86, 1.14),
                            torch.ones(B, device=device),
                        ).view(B, 1, 1)
                        bone = seq[:, :, c_j, :] - seq[:, :, p_j, :]
                        delta = bone * (scale - 1.0)
                        for down in chain[s_idx + 1 :]:
                            dj = offset + down
                            active_dj = pt_active[:, :, dj, :]
                            seq[:, :, dj, :] = torch.where(active_dj, seq[:, :, dj, :] + delta, seq[:, :, dj, :])

    # 5. Small 3D Viewpoint Rotation + Anisotropic Scale around clip hand centroid
    if p_rotate > 0:
        do_r = torch.rand(B, device=device) < p_rotate
        if torch.any(do_r):
            ang_z = torch.where(do_r, torch.empty(B, device=device).uniform_(-0.22, 0.22), torch.zeros(B, device=device))
            ang_y = torch.where(do_r, torch.empty(B, device=device).uniform_(-0.20, 0.20), torch.zeros(B, device=device))
            scale_xy = torch.where(
                do_r.unsqueeze(1),
                torch.empty(B, 2, device=device).uniform_(0.93, 1.07),
                torch.ones(B, 2, device=device),
            )

            act_f = pt_active.float()
            cnt = act_f.sum(dim=(1, 2), keepdim=True).clamp(min=1.0)
            center = (seq * act_f).sum(dim=(1, 2), keepdim=True) / cnt

            rel = seq - center
            cz, sz = torch.cos(ang_z).view(B, 1, 1), torch.sin(ang_z).view(B, 1, 1)
            cy, sy = torch.cos(ang_y).view(B, 1, 1), torch.sin(ang_y).view(B, 1, 1)

            rx = rel[..., 0] * cz - rel[..., 1] * sz
            ry = rel[..., 0] * sz + rel[..., 1] * cz
            rz = rel[..., 2]

            rx2 = rx * cy + rz * sy
            rz2 = -rx * sy + rz * cy

            rx2 = rx2 * scale_xy[:, 0].view(B, 1, 1)
            ry = ry * scale_xy[:, 1].view(B, 1, 1)

            rot = torch.stack([rx2, ry, rz2], dim=-1) + center
            seq = torch.where(pt_active, rot, seq)

    # 6. Random Frame & Single-Hand Tracking Dropout (simulates webcam MediaPipe flicker)
    if p_frame_drop > 0:
        do_d = (torch.rand(B, device=device) < p_frame_drop).view(B, 1, 1, 1)
        frame_drop_mask = (torch.rand(B, T, 1, 1, device=device) < 0.12) & do_d
        seq = seq.masked_fill(frame_drop_mask, 0.0)
        pt_active = seq.abs().sum(dim=-1, keepdim=True) > 1e-5

    # 7. Landmark Gaussian Noise (only on non-zero active points)
    if noise_std > 0:
        noise = torch.randn_like(seq) * noise_std
        seq = torch.where(pt_active, seq + noise, seq)

    return seq


@torch.no_grad()
def synthesize_ood_negative_batch(real_batch: torch.Tensor) -> torch.Tensor:
    """
    Create realistic OOD / NO_SIGN / Transition negative samples from real batches so
    the OOD head learns to reject:
      1. All-zero / empty camera frames
      2. Frozen resting hands with natural webcam jitter
      3. Slow postural drift (non-signing lowering/raising hand)
      4. Short coarticulation blend segments between two unrelated signs
    Returns neg_batch of shape (B, 32, 42, 3) where all samples are non-signs.
    """
    B, T, _, _ = real_batch.shape
    device = real_batch.device
    neg = torch.zeros_like(real_batch)

    q1 = max(1, B // 4)
    q2 = max(1, B // 2)
    q3 = max(1, (3 * B) // 4)

    # Segment 1 [0:q1]: Empty or 1-2 stray flash frames
    flash_mask = torch.rand(q1, T, 1, 1, device=device) < 0.04
    neg[:q1] = torch.where(flash_mask, real_batch[:q1], neg[:q1])

    # Segment 2 [q1:q2]: Static frozen hand (pick an active frame from real sample + tiny tremor)
    src_static = real_batch[q1:q2]
    frame_energy = src_static.abs().sum(dim=(2, 3))  # (n, T)
    mid_frame_idx = frame_energy.argmax(dim=1)  # peak frame
    n2 = q2 - q1
    picked = src_static[torch.arange(n2, device=device), mid_frame_idx]  # (n2, 42, 3)
    held = picked.unsqueeze(1).expand(-1, T, -1, -1).clone()
    act_held = held.abs().sum(dim=-1, keepdim=True) > 1e-5
    neg[q1:q2] = held + torch.where(act_held, torch.randn_like(held) * 0.0018, torch.zeros_like(held))

    # Segment 3 [q2:q3]: Very slow constant hand drift / fidget (no lexical handshape dynamics)
    n3 = q3 - q2
    src_drift = real_batch[q2:q3]
    f_idx = src_drift.abs().sum(dim=(2, 3)).argmax(dim=1)
    base_pose = src_drift[torch.arange(n3, device=device), f_idx].unsqueeze(1).expand(-1, T, -1, -1).clone()
    drift_vec = (torch.rand(n3, 1, 1, 3, device=device) - 0.5) * 0.03
    t_lin = torch.linspace(-0.5, 0.5, T, device=device).view(1, T, 1, 1)
    act_d = base_pose.abs().sum(dim=-1, keepdim=True) > 1e-5
    neg[q2:q3] = base_pose + torch.where(act_d, drift_vec * t_lin + torch.randn_like(base_pose) * 0.002, torch.zeros_like(base_pose))

    # Segment 4 [q3:B]: Pure coarticulation interpolation blend between two random end frames (non-sign move)
    n4 = B - q3
    if n4 > 0:
        a = real_batch[q3:B]
        b = real_batch[torch.randperm(B, device=device)[:n4]]
        fa = a[torch.arange(n4, device=device), a.abs().sum(dim=(2, 3)).argmax(dim=1)]
        fb = b[torch.arange(n4, device=device), b.abs().sum(dim=(2, 3)).argmax(dim=1)]
        alpha = torch.linspace(0.0, 1.0, T, device=device).view(1, T, 1, 1)
        # Active only where both start & end have that joint, otherwise slide linearly
        blend = (1.0 - alpha) * fa.unsqueeze(1) + alpha * fb.unsqueeze(1)
        neg[q3:B] = blend

    return neg
