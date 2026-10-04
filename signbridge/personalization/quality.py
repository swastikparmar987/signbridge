from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from datetime import datetime, timezone


class SampleQualityChecker:
    ZERO_FRAME_RATIO_THRESHOLD = 0.80
    MIN_ACTIVE_FRAMES = 8
    MIN_HAND_DISTANCE = 0.01

    @staticmethod
    def check(
        sequence: np.ndarray,
        num_frames: int = 32,
        num_landmarks: int = 42,
        num_coords: int = 3,
    ) -> Dict[str, Any]:
        if sequence is None:
            return {
                "valid": False,
                "score": 0.0,
                "feedback": "No sequence data received.",
            }

        seq = np.asarray(sequence, dtype=np.float32)

        if seq.size == 0:
            return {"valid": False, "score": 0.0, "feedback": "Empty sequence."}

        if np.any(np.isnan(seq)):
            return {"valid": False, "score": 0.0, "feedback": "Sequence contains NaN values."}
        if np.any(np.isinf(seq)):
            return {"valid": False, "score": 0.0, "feedback": "Sequence contains Inf values."}

        if seq.ndim == 1:
            seq = seq.reshape(num_frames, num_landmarks, num_coords)
        elif seq.ndim == 2:
            seq = seq.reshape(-1, num_landmarks, num_coords)

        if seq.ndim != 3 or seq.shape[1:] != (num_landmarks, num_coords):
            if seq.shape[0] == num_frames * num_landmarks * num_coords:
                seq = seq.reshape(num_frames, num_landmarks, num_coords)
            else:
                return {
                    "valid": False,
                    "score": 0.0,
                    "feedback": f"Unexpected shape {seq.shape}. Expected ({num_frames}, {num_landmarks}, {num_coords}).",
                }

        if seq.shape[0] != num_frames:
            return {
                "valid": False,
                "score": 0.0,
                "feedback": f"Expected {num_frames} frames, got {seq.shape[0]}.",
            }

        flat = seq.reshape(num_frames, -1)

        zero_frames = np.all(np.abs(flat) < 1e-6, axis=1).sum()
        zero_frame_ratio = zero_frames / num_frames
        if zero_frame_ratio >= SampleQualityChecker.ZERO_FRAME_RATIO_THRESHOLD:
            return {
                "valid": False,
                "score": 15.0,
                "feedback": f"Too many empty frames ({zero_frame_ratio*100:.0f}%). Hold your hands steady in view.",
            }

        frame_norms = np.linalg.norm(flat, axis=1)
        active_frames = np.sum(frame_norms > 1e-4)
        if active_frames < SampleQualityChecker.MIN_ACTIVE_FRAMES:
            return {
                "valid": False,
                "score": 30.0,
                "feedback": f"Too few active frames ({active_frames}/{num_frames}). Perform the sign with more motion.",
            }

        hand1 = seq[:, :21, :]
        hand2 = seq[:, 21:, :]
        hand1_norms = np.linalg.norm(hand1, axis=2)
        hand2_norms = np.linalg.norm(hand2, axis=2)
        hand1_detected = np.any(hand1_norms > SampleQualityChecker.MIN_HAND_DISTANCE, axis=1)
        hand2_detected = np.any(hand2_norms > SampleQualityChecker.MIN_HAND_DISTANCE, axis=1)
        either_hand = np.logical_or(hand1_detected, hand2_detected)
        if not np.any(either_hand):
            return {
                "valid": False,
                "score": 20.0,
                "feedback": "No hand detected. Ensure your hands are visible and well-lit.",
            }

        hand1_ratio = np.mean(hand1_detected)
        hand2_ratio = np.mean(hand2_detected)
        if hand1_ratio < 0.3 and hand2_ratio < 0.3:
            return {
                "valid": False,
                "score": 25.0,
                "feedback": "Hands barely visible. Try repositioning in the frame.",
            }

        temporal_std = np.std(frame_norms)
        if temporal_std < 0.005:
            return {
                "valid": False,
                "score": 40.0,
                "feedback": "Not enough hand movement. Perform the sign with more dynamic motion.",
            }

        scores = []
        scores.append(min(active_frames / num_frames * 100, 100.0) * 0.5)
        scores.append(min(temporal_std * 500, 100.0) * 0.3)
        scores.append(max(0, (1.0 - zero_frame_ratio) * 100) * 0.2)
        quality_score = sum(scores)

        if quality_score >= 80:
            feedback = "Good sample ✓"
        elif quality_score >= 60:
            feedback = "Acceptable sample — try for better clarity next time."
        else:
            feedback = "Sample quality is low. Consider recording again."

        return {
            "valid": True,
            "score": round(quality_score, 1),
            "feedback": feedback,
            "zero_frame_ratio": round(zero_frame_ratio, 3),
            "active_frames": int(active_frames),
            "temporal_std": round(float(temporal_std), 5),
        }
