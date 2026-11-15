from pathlib import Path
from typing import Optional
import numpy as np

NUM_FRAMES = 32
NUM_LANDMARKS = 42
NUM_COORDS = 3


def get_sampled_frame_indices(total_frames: int, num_frames: int = 32) -> Optional[np.ndarray]:
    """
    Uniformly sample num_frames indices from total_frames.
    
    Args:
        total_frames: Integer frame count of video
        num_frames: Target sequence length (default 32)
        
    Returns:
        np.ndarray of frame indices or None if total_frames <= 0.
    """
    if total_frames <= 0:
        return None
    if total_frames == 1:
        return np.zeros(num_frames, dtype=int)
    return np.linspace(0, total_frames - 1, num_frames).astype(int)


def stabilize_hand_sequence(
    sequence: np.ndarray,
    max_gap: int = 3,
    swap_ratio: float = 0.72,
) -> np.ndarray:
    """
    Stabilize a raw (T, 42, 3) landmark sequence:
    1. Replace NaN/Inf with 0.0 and detect active hand slots per frame.
    2. Resolve transient left/right slot swaps when two hands are close or one briefly disappears.
    3. Linearly interpolate short interior dropout gaps (<= max_gap frames) without fabricating long missing hands.
    """
    seq = np.nan_to_num(np.asarray(sequence, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0).copy()
    if seq.ndim == 2 and seq.shape[1] == 126:
        seq = seq.reshape(seq.shape[0], 42, 3)
    if seq.ndim != 3 or seq.shape[1:] != (42, 3) or seq.shape[0] == 0:
        return seq

    T = seq.shape[0]
    anchor0: Optional[np.ndarray] = None
    anchor1: Optional[np.ndarray] = None
    last0 = -999
    last1 = -999

    for t in range(T):
        h0 = seq[t, :21]
        h1 = seq[t, 21:42]
        a0 = bool(np.abs(h0).sum() > 1e-4)
        a1 = bool(np.abs(h1).sum() > 1e-4)

        if a0 and a1:
            if anchor0 is not None and anchor1 is not None and (t - last0) <= 5 and (t - last1) <= 5:
                d_keep = float(np.linalg.norm(h0[0, :2] - anchor0) + np.linalg.norm(h1[0, :2] - anchor1))
                d_swap = float(np.linalg.norm(h0[0, :2] - anchor1) + np.linalg.norm(h1[0, :2] - anchor0))
                if d_swap < d_keep * swap_ratio:
                    seq[t, :21], seq[t, 21:42] = h1.copy(), h0.copy()
                    h0, h1 = seq[t, :21], seq[t, 21:42]
            anchor0 = h0[0, :2].copy()
            anchor1 = h1[0, :2].copy()
            last0 = t
            last1 = t
        elif a0 and not a1:
            if anchor0 is not None and anchor1 is not None and (t - last0) <= 5 and (t - last1) <= 5:
                d0 = float(np.linalg.norm(h0[0, :2] - anchor0))
                d1 = float(np.linalg.norm(h0[0, :2] - anchor1))
                if d1 < d0 * swap_ratio:
                    seq[t, 21:42] = h0.copy()
                    seq[t, :21] = 0.0
                    anchor1 = seq[t, 21, :2].copy()
                    last1 = t
                    continue
            anchor0 = h0[0, :2].copy()
            last0 = t
        elif a1 and not a0:
            if anchor0 is not None and anchor1 is not None and (t - last0) <= 5 and (t - last1) <= 5:
                d0 = float(np.linalg.norm(h1[0, :2] - anchor0))
                d1 = float(np.linalg.norm(h1[0, :2] - anchor1))
                if d0 < d1 * swap_ratio:
                    seq[t, :21] = h1.copy()
                    seq[t, 21:42] = 0.0
                    anchor0 = seq[t, 0, :2].copy()
                    last0 = t
                    continue
            anchor1 = h1[0, :2].copy()
            last1 = t

    if max_gap > 0 and T >= 3:
        for offset in (0, 21):
            hand_slice = seq[:, offset : offset + 21, :]
            active = np.abs(hand_slice).sum(axis=(1, 2)) > 1e-4
            valid_idx = np.where(active)[0]
            if len(valid_idx) < 2:
                continue
            for idx_pos in range(len(valid_idx) - 1):
                t_start = int(valid_idx[idx_pos])
                t_end = int(valid_idx[idx_pos + 1])
                gap = t_end - t_start - 1
                if 1 <= gap <= max_gap:
                    for step in range(1, gap + 1):
                        alpha = step / float(gap + 1)
                        seq[t_start + step, offset : offset + 21] = (
                            (1.0 - alpha) * hand_slice[t_start] + alpha * hand_slice[t_end]
                        )

    return seq


def mirror_sequence_x(sequence: np.ndarray, swap_slots: bool = False) -> np.ndarray:
    """Horizontally mirror active landmarks (x -> 1.0 - x) preserving exact zero frames."""
    seq = np.asarray(sequence, dtype=np.float32).copy()
    if seq.ndim == 2 and seq.shape[1] == 126:
        seq = seq.reshape(seq.shape[0], 42, 3)
    active_pts = np.abs(seq).sum(axis=-1) > 1e-6
    seq[active_pts, 0] = 1.0 - seq[active_pts, 0]
    if swap_slots:
        seq = np.concatenate([seq[:, 21:42, :], seq[:, :21, :]], axis=1)
    return seq


def compute_sequence_activity(sequence: np.ndarray) -> dict:
    """
    Compute palm-scale-normalized motion energy, pose articulation flux, and active frame statistics.
    Used for resting/static OOD NO_SIGN gating and continuous CSLR boundary detection.
    """
    seq = np.nan_to_num(np.asarray(sequence, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if seq.ndim == 2 and seq.shape[1] == 126:
        seq = seq.reshape(seq.shape[0], 42, 3)
    T = seq.shape[0]
    if T == 0:
        return {
            "active_frame_ratio": 0.0,
            "active_frames": 0,
            "mean_wrist_speed": 0.0,
            "max_wrist_speed": 0.0,
            "mean_pose_flux": 0.0,
            "wrist_span": 0.0,
            "motion_score": 0.0,
        }

    h0 = seq[:, :21, :]
    h1 = seq[:, 21:42, :]
    a0 = np.abs(h0).sum(axis=(1, 2)) > 1e-4
    a1 = np.abs(h1).sum(axis=(1, 2)) > 1e-4
    any_active = a0 | a1
    active_frames = int(np.sum(any_active))
    active_ratio = float(active_frames / max(1, T))

    speeds = []
    fluxes = []
    spans = []

    for h, act in ((h0, a0), (h1, a1)):
        if np.sum(act) < 2:
            continue
        valid = h[act]
        wrist = valid[:, 0, :]
        palm_scales = np.linalg.norm(valid[:, 9, :] - valid[:, 0, :], axis=-1)
        med_scale = float(np.median(palm_scales[palm_scales > 1e-4])) if np.any(palm_scales > 1e-4) else 0.08
        med_scale = max(med_scale, 0.02)

        spans.append(float(np.linalg.norm(wrist[:, :2].max(axis=0) - wrist[:, :2].min(axis=0)) / med_scale))

        # Consecutive active pairs in original timeline
        pair_mask = act[1:] & act[:-1]
        if np.any(pair_mask):
            dv = np.linalg.norm(h[1:, 0, :] - h[:-1, 0, :], axis=-1)[pair_mask] / med_scale
            speeds.extend(dv.tolist())

            rel = (h - h[:, 0:1, :]) / med_scale
            dp = np.mean(np.linalg.norm(rel[1:] - rel[:-1], axis=-1), axis=-1)[pair_mask]
            fluxes.extend(dp.tolist())

    mean_wrist_speed = float(np.mean(speeds)) if speeds else 0.0
    max_wrist_speed = float(np.percentile(speeds, 90)) if speeds else 0.0
    mean_pose_flux = float(np.mean(fluxes)) if fluxes else 0.0
    wrist_span = float(max(spans)) if spans else 0.0
    motion_score = float(mean_wrist_speed * 0.55 + mean_pose_flux * 0.85 + wrist_span * 0.12)

    return {
        "active_frame_ratio": active_ratio,
        "active_frames": active_frames,
        "mean_wrist_speed": mean_wrist_speed,
        "max_wrist_speed": max_wrist_speed,
        "mean_pose_flux": mean_pose_flux,
        "wrist_span": wrist_span,
        "motion_score": motion_score,
    }


def extract_hand_sequence(
    video_path: Path | str,
    num_frames: int = 32,
    min_detection_confidence: float = 0.5,
    stabilize: bool = False,
) -> np.ndarray:
    """
    Extract raw hand landmarks (32, 42, 3) from a video using OpenCV and MediaPipe.
    
    Hands are ordered left-to-right by wrist x-coordinate.
    Missing hands or missing frames are filled with zeros.
    
    Args:
        video_path: Path to video file
        num_frames: Target frame count (default 32)
        min_detection_confidence: MediaPipe Hands detection threshold
        
    Returns:
        np.ndarray of shape (num_frames, 42, 3), float32.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    try:
        import cv2
        import mediapipe as mp
    except ImportError as exc:
        raise RuntimeError(
            "Video landmark extraction requires opencv-python and mediapipe on the server; "
            "use live webcam or client-side landmark extraction in Lite cloud mode."
        ) from exc

    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        return np.zeros((num_frames, NUM_LANDMARKS, NUM_COORDS), dtype=np.float32)

    frame_indices = get_sampled_frame_indices(total_frames, num_frames)

    mp_hands = mp.solutions.hands
    sequence = []

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=min_detection_confidence,
    ) as hands:
        for frame_index in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            success, frame = cap.read()

            if not success or frame is None:
                sequence.append(np.zeros((NUM_LANDMARKS, NUM_COORDS), dtype=np.float32))
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            frame_landmarks = np.zeros((NUM_LANDMARKS, NUM_COORDS), dtype=np.float32)

            if results.multi_hand_landmarks:
                detected_hands = []
                for hand_landmarks in results.multi_hand_landmarks:
                    landmarks = np.array(
                        [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
                        dtype=np.float32,
                    )
                    wrist_x = landmarks[0, 0]
                    detected_hands.append((wrist_x, landmarks))

                # Order hands left-to-right by wrist X
                detected_hands.sort(key=lambda x: x[0])

                for i, (_, landmarks) in enumerate(detected_hands[:2]):
                    start = i * 21
                    frame_landmarks[start : start + 21] = landmarks

            sequence.append(frame_landmarks)

    cap.release()
    arr = np.array(sequence, dtype=np.float32)
    return stabilize_hand_sequence(arr) if stabilize else arr

