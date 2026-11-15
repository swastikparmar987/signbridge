import numpy as np

# Kinematic finger chains for a single hand (21 joints)
FINGER_CHAINS = [
    [0, 1, 2, 3, 4],     # Thumb
    [0, 5, 6, 7, 8],     # Index
    [0, 9, 10, 11, 12],  # Middle
    [0, 13, 14, 15, 16], # Ring
    [0, 17, 18, 19, 20], # Pinky
]


def apply_skeleton_warp(sequence: np.ndarray, scale_min: float = 0.85, scale_max: float = 1.15) -> np.ndarray:
    """
    Apply signer-robust kinematic bone length perturbation to landmark sequence.
    
    Args:
        sequence: Array of shape (T, 42, 3) or (T, 2, 21, 3).
        scale_min: Minimum scale factor for bone segments (default 0.85).
        scale_max: Maximum scale factor for bone segments (default 1.15).
        
    Returns:
        Augmented array of shape (T, 42, 3).
    """
    sequence = np.array(sequence, dtype=np.float32, copy=True)
    orig_shape = sequence.shape
    if sequence.ndim == 4:
        sequence = sequence.reshape(orig_shape[0], 42, 3)

    T = sequence.shape[0]

    # Sample segment scale factors for Hand 1 and Hand 2
    # 5 fingers x 4 segments = 20 segment scale factors per hand
    scale_factors_h1 = np.random.uniform(scale_min, scale_max, size=(5, 4))
    scale_factors_h2 = np.random.uniform(scale_min, scale_max, size=(5, 4))

    for h, scale_factors in enumerate([scale_factors_h1, scale_factors_h2]):
        offset = h * 21
        for f_idx, chain in enumerate(FINGER_CHAINS):
            for s_idx in range(len(chain) - 1):
                parent_idx = offset + chain[s_idx]
                child_idx = offset + chain[s_idx + 1]
                scale = scale_factors[f_idx, s_idx]

                # Bone vectors across all frames T: shape (T, 3)
                bone_vec = sequence[:, child_idx] - sequence[:, parent_idx]
                scaled_bone_vec = bone_vec * scale
                delta = scaled_bone_vec - bone_vec  # shape (T, 3)

                # Shift child and all downstream joints in the chain
                for downstream_joint in chain[s_idx + 1:]:
                    target_idx = offset + downstream_joint
                    sequence[:, target_idx] += delta

    return sequence
