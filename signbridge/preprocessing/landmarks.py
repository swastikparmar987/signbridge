from pathlib import Path
from typing import Optional
import cv2
import mediapipe as mp
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


def extract_hand_sequence(
    video_path: Path | str,
    num_frames: int = 32,
    min_detection_confidence: float = 0.5,
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
    return np.array(sequence, dtype=np.float32)

