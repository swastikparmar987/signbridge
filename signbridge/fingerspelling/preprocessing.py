import json
from pathlib import Path
from typing import List, Dict, Any, Union, Optional
import numpy as np

# Landmark Indices according to Kaggle ASL Fingerspelling 92-landmark subset
POSE_INDICES = [13, 15, 17, 19, 21, 14, 16, 18, 20, 22]
FACE_INDICES = [
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 146, 91, 181, 84, 17,
    314, 405, 321, 375, 78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 95, 88,
    178, 87, 14, 317, 402, 318, 324, 308
]

class FingerspellingPreprocessor:
    """
    Landmark preprocessor for ASL fingerspelling recognition.
    Aligns hand, face, and pose landmarks to the 92-landmark (276 features) format
    and normalizes coordinates for Squeezeformer inference.
    """

    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        if config_path is None:
            config_path = Path(__file__).resolve().parent / "inference_args.json"
        
        self.config_path = Path(config_path)
        if self.config_path.exists():
            with open(self.config_path, "r") as f:
                self.selected_columns = json.load(f)["selected_columns"]
        else:
            self.selected_columns = self._build_default_columns()

        self.num_landmarks = len(self.selected_columns) // 3
        self.num_features = len(self.selected_columns)

    @staticmethod
    def _build_default_columns() -> List[str]:
        cols = []
        for axis in ["x", "y", "z"]:
            for i in range(21):
                cols.append(f"{axis}_right_hand_{i}")
            for i in range(21):
                cols.append(f"{axis}_left_hand_{i}")
            for p in POSE_INDICES:
                cols.append(f"{axis}_pose_{p}")
            for f in FACE_INDICES:
                cols.append(f"{axis}_face_{f}")
        return cols

    def extract_from_frame_dict(self, frame_dict: Dict[str, Any]) -> np.ndarray:
        """
        Extract 276 feature vector from a single frame landmark dictionary.
        Supports inputs containing 'right_hand', 'left_hand', 'pose', and 'face'.
        """
        rh = frame_dict.get("right_hand") or []
        lh = frame_dict.get("left_hand") or []
        pose = frame_dict.get("pose") or []
        face = frame_dict.get("face") or []

        # Default 21 landmarks for hands
        rh_arr = np.array(rh, dtype=np.float32) if len(rh) == 21 else np.zeros((21, 3), dtype=np.float32)
        lh_arr = np.array(lh, dtype=np.float32) if len(lh) == 21 else np.zeros((21, 3), dtype=np.float32)

        # Pose (10 keypoints)
        pose_arr = np.zeros((10, 3), dtype=np.float32)
        if len(pose) >= 33:
            for idx, p_idx in enumerate(POSE_INDICES):
                pose_arr[idx] = pose[p_idx][:3]
        elif len(pose) == 10:
            pose_arr = np.array(pose, dtype=np.float32)

        # Face (40 keypoints)
        face_arr = np.zeros((40, 3), dtype=np.float32)
        if len(face) >= 468:
            for idx, f_idx in enumerate(FACE_INDICES):
                face_arr[idx] = face[f_idx][:3]
        elif len(face) == 40:
            face_arr = np.array(face, dtype=np.float32)

        # 92 landmarks: [right_hand(21), left_hand(21), pose(10), face(40)]
        landmarks_92 = np.vstack([rh_arr, lh_arr, pose_arr, face_arr])  # (92, 3)

        # Reorder into [x_0..x_91, y_0..y_91, z_0..z_91]
        x = landmarks_92[:, 0]
        y = landmarks_92[:, 1]
        z = landmarks_92[:, 2]
        return np.concatenate([x, y, z])

    def preprocess_sequence(
        self,
        sequence: Union[np.ndarray, List[Any]],
        target_len: Optional[int] = None,
        min_len: int = 16,
    ) -> np.ndarray:
        """
        Preprocess sequence of frames into normalized [T, 276] tensor.
        Handles missing values, NaNs, zero division, and sequence interpolation.
        """
        if isinstance(sequence, list):
            if len(sequence) == 0:
                return np.zeros((min_len, 276), dtype=np.float32)
            if isinstance(sequence[0], dict):
                seq_arr = np.array([self.extract_from_frame_dict(f) for f in sequence], dtype=np.float32)
            else:
                seq_arr = np.array(sequence, dtype=np.float32)
        else:
            seq_arr = np.array(sequence, dtype=np.float32)

        # Reshape to 2D (T, 276) if needed
        if seq_arr.ndim == 3:
            if seq_arr.shape[1:] == (92, 3):
                # Transpose and flatten to (T, 276): all x, then all y, then all z
                x = seq_arr[:, :, 0]
                y = seq_arr[:, :, 1]
                z = seq_arr[:, :, 2]
                seq_arr = np.concatenate([x, y, z], axis=1)
            elif seq_arr.shape[1:] == (42, 3):
                # 42 hand landmarks only from browser -> pad pose and face with zeros
                T = seq_arr.shape[0]
                rh = seq_arr[:, :21, :]
                lh = seq_arr[:, 21:, :]
                pose = np.zeros((T, 10, 3), dtype=np.float32)
                face = np.zeros((T, 40, 3), dtype=np.float32)
                full = np.concatenate([rh, lh, pose, face], axis=1)  # (T, 92, 3)
                x = full[:, :, 0]
                y = full[:, :, 1]
                z = full[:, :, 2]
                seq_arr = np.concatenate([x, y, z], axis=1)
            else:
                seq_arr = seq_arr.reshape(seq_arr.shape[0], -1)

        T = seq_arr.shape[0]

        # Ensure minimum sequence length via temporal interpolation or padding
        if T < min_len:
            pad_frames = np.repeat(seq_arr[-1:], min_len - T, axis=0) if T > 0 else np.zeros((min_len, 276), dtype=np.float32)
            seq_arr = np.vstack([seq_arr, pad_frames]) if T > 0 else pad_frames
            T = min_len

        if target_len is not None and T != target_len:
            # Linear interpolation across temporal dimension
            orig_indices = np.linspace(0, T - 1, num=T)
            target_indices = np.linspace(0, T - 1, num=target_len)
            resampled = np.zeros((target_len, seq_arr.shape[1]), dtype=np.float32)
            for c in range(seq_arr.shape[1]):
                resampled[:, c] = np.interp(target_indices, orig_indices, seq_arr[:, c])
            seq_arr = resampled

        # Normalization (per-sequence over valid non-nan / non-zero points)
        non_nan_mask = ~np.isnan(seq_arr) & (seq_arr != 0.0)
        if np.any(non_nan_mask):
            mean = np.mean(seq_arr[non_nan_mask])
            std = np.std(seq_arr[non_nan_mask])
            if std > 1e-5:
                seq_arr = (seq_arr - mean) / std

        # Impute remaining NaNs/Infs
        seq_arr = np.nan_to_num(seq_arr, nan=0.0, posinf=0.0, neginf=0.0)
        return seq_arr.astype(np.float32)

