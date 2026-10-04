from typing import Tuple, Optional
import numpy as np
from signbridge.training.config import AugmentationConfig


class LandmarkAugmentor:
    """
    Safe raw landmark augmentation pipeline for ASL sequences.
    
    CRITICAL RULES:
    1. Augment RAW landmarks BEFORE normalization.
    2. Zero frames (missing hands) must NEVER be converted into fake hands.
    3. Natural zero-frame structure must be strictly preserved.
    4. No aggressive dropping or distorted lifecycle of signs.
    """

    def __init__(self, config: Optional[AugmentationConfig] = None):
        self.config = config or AugmentationConfig(enabled=True)

    def __call__(self, sequence: np.ndarray) -> np.ndarray:
        """
        Apply safe augmentations to a raw landmark sequence.
        
        Args:
            sequence: np.ndarray of shape (32, 42, 3) or (32, 126)
            
        Returns:
            np.ndarray of shape (32, 42, 3), float32
        """
        if not self.config.enabled:
            if sequence.ndim == 2 and sequence.shape[1] == 126:
                return sequence.reshape(32, 42, 3).astype(np.float32)
            return sequence.astype(np.float32)

        # Ensure (32, 42, 3)
        if sequence.ndim == 2 and sequence.shape[1] == 126:
            seq = sequence.reshape(32, 42, 3).copy().astype(np.float32)
        elif sequence.ndim == 3 and sequence.shape[1:] == (42, 3):
            seq = sequence.copy().astype(np.float32)
        else:
            raise ValueError(f"Invalid sequence shape: {sequence.shape}. Expected (32, 42, 3) or (32, 126)")

        # 1. Mild temporal speed & jitter (nearest-neighbor index resampling)
        seq = self.apply_temporal_resampling(seq)

        # 2. Small spatial scaling (per active hand, slight anisotropic scaling)
        seq = self.apply_spatial_scaling(seq)

        # 3. Small in-plane rotation around each wrist.
        seq = self.apply_rotation(seq)

        # 4. Simulate occasional tracking loss without changing zero structure
        seq = self.apply_tracking_dropout(seq)

        # 5. Small Gaussian noise (applied ONLY to non-zero landmarks)
        seq = self.add_gaussian_noise(seq)

        return seq.astype(np.float32)

    def apply_temporal_resampling(self, seq: np.ndarray) -> np.ndarray:
        """
        Resample sequence in time with mild speed variation and jitter.
        Uses nearest-neighbor indexing to strictly preserve exact zero frames.
        """
        speed = np.random.uniform(
            self.config.temporal_speed_range[0],
            self.config.temporal_speed_range[1]
        )
        jitter = np.random.randint(
            -self.config.temporal_jitter_frames,
            self.config.temporal_jitter_frames + 1
        )
        
        # Center of time axis is 15.5
        t = np.arange(32, dtype=np.float32)
        center = 15.5
        t_resampled = center + (t - center) * speed + jitter
        indices = np.clip(np.round(t_resampled).astype(np.int64), 0, 31)

        return seq[indices]

    def apply_spatial_scaling(self, seq: np.ndarray) -> np.ndarray:
        """
        Apply slight anisotropic spatial scaling to active hands.
        Does not touch zero hands.
        """
        # For each frame and each of the two hands (21 landmarks each)
        for h_start, h_end in [(0, 21), (21, 42)]:
            hand = seq[:, h_start:h_end, :] # (32, 21, 3)
            # Find frames where this hand is active
            hand_norm = np.linalg.norm(hand, axis=(1, 2)) # (32,)
            active_frames = np.where(hand_norm > 1e-4)[0]
            
            if len(active_frames) == 0:
                continue

            # Sample scale factor per coordinate axis: [sx, sy, sz]
            scale = np.random.uniform(
                self.config.spatial_scale_range[0],
                self.config.spatial_scale_range[1],
                size=(1, 1, 3)
            ).astype(np.float32)

            # Center scaling on the hand centroid of each active frame
            for f in active_frames:
                frame_hand = hand[f] # (21, 3)
                wrist = frame_hand[0:1, :] # Landmark 0
                scaled_hand = wrist + (frame_hand - wrist) * scale[0]
                seq[f, h_start:h_end, :] = scaled_hand

        return seq

    def apply_rotation(self, seq: np.ndarray) -> np.ndarray:
        """Rotate each active hand mildly around its wrist in the camera plane."""
        max_degrees = float(self.config.rotation_degrees)
        if max_degrees <= 0:
            return seq

        angle = np.deg2rad(np.random.uniform(-max_degrees, max_degrees))
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rotation = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)
        for h_start in (0, 21):
            hand = seq[:, h_start:h_start + 21]
            active = np.linalg.norm(hand, axis=(1, 2)) > 1e-4
            for frame_idx in np.where(active)[0]:
                wrist = hand[frame_idx, 0].copy()
                xy = hand[frame_idx, :, :2] - wrist[:2]
                hand[frame_idx, :, :2] = xy @ rotation.T + wrist[:2]
        return seq

    def apply_tracking_dropout(self, seq: np.ndarray) -> np.ndarray:
        """Drop complete frames or landmarks only when explicitly configured."""
        frame_probability = float(self.config.frame_dropout_probability)
        landmark_probability = float(self.config.landmark_dropout_probability)
        if frame_probability <= 0 and landmark_probability <= 0:
            return seq

        active_frames = np.where(np.linalg.norm(seq, axis=(1, 2)) > 1e-4)[0]
        for frame_idx in active_frames:
            if frame_probability > 0 and np.random.random() < frame_probability:
                neighbors = active_frames[active_frames != frame_idx]
                if len(neighbors):
                    source_idx = neighbors[np.argmin(np.abs(neighbors - frame_idx))]
                    seq[frame_idx] = seq[source_idx]
            if landmark_probability > 0:
                active_landmarks = np.linalg.norm(seq[frame_idx], axis=1) > 1e-4
                drop = (np.random.random(42) < landmark_probability) & active_landmarks
                seq[frame_idx, drop] = 0.0
        return seq

    def add_gaussian_noise(self, seq: np.ndarray) -> np.ndarray:
        """
        Add small Gaussian noise strictly to non-zero landmarks.
        Zero landmarks remain exactly 0.0.
        """
        sigma = self.config.noise_sigma
        if sigma <= 0:
            return seq

        # Landmark activity mask: shape (32, 42)
        lm_norm = np.linalg.norm(seq, axis=-1)
        active_mask = (lm_norm > 1e-4) # (32, 42)

        noise = np.random.normal(0.0, sigma, size=seq.shape).astype(np.float32)
        # Apply noise only to active landmarks
        seq = np.where(active_mask[..., None], seq + noise, 0.0)
        return seq
