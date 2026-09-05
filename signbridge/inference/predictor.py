from pathlib import Path
from typing import List, Dict, Any, Union, Optional
import numpy as np
import torch
import torch.nn.functional as F

from signbridge.inference.model import GRUClassifier, load_signbridge_model, get_default_device
from signbridge.preprocessing.normalize import normalize_sequence
from signbridge.preprocessing.landmarks import extract_hand_sequence


class SignBridgePredictor:
    """
    High-level production predictor for SignBridge model.
    """

    def __init__(
        self,
        checkpoint_path: Path | str,
        device: Optional[torch.device] = None,
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.device = device or get_default_device()
        self.model, self.class_names, self.metadata = load_signbridge_model(
            self.checkpoint_path,
            device=self.device,
        )
        self.input_size = self.metadata["input_size"]

    def predict_sequence(
        self,
        sequence: np.ndarray,
        is_normalized: bool = False,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        Run inference on numpy landmark sequence.
        
        Args:
            sequence: np.ndarray of shape (T, 42, 3) or (T, 126)
            is_normalized: Set True if input sequence is already wrist & scale normalized
            top_k: Number of top predictions to return
            
        Returns:
            Dict containing top prediction gloss, confidence %, and list of top_k predictions.
        """
        sequence = np.nan_to_num(sequence, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        if not is_normalized:
            sequence = normalize_sequence(sequence)

        # Flatten sequence to (T, 126) if (T, 42, 3)
        if sequence.ndim == 3 and sequence.shape[1:] == (42, 3):
            sequence = sequence.reshape(sequence.shape[0], 126)

        if sequence.shape[1] != self.input_size:
            raise ValueError(
                f"Feature dimension mismatch: expected feature size {self.input_size}, got {sequence.shape[1]}"
            )

        tensor_input = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor_input)
            probs = F.softmax(logits, dim=1)[0]
            top_probs, top_indices = torch.topk(probs, k=min(top_k, len(self.class_names)))

        predictions = []
        for prob, idx in zip(top_probs.cpu().numpy(), top_indices.cpu().numpy()):
            predictions.append(
                {
                    "class_index": int(idx),
                    "gloss": self.class_names[idx],
                    "confidence": float(prob * 100.0),
                }
            )

        return {
            "predicted_gloss": predictions[0]["gloss"],
            "confidence": predictions[0]["confidence"],
            "top_k": predictions,
        }

    def predict_cache(
        self,
        cache_path: Path | str,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        Run inference on raw landmark cache file (.npy).
        
        Args:
            cache_path: Path to cached .npy sequence file
            top_k: Number of top predictions to return
            
        Returns:
            Dict containing predictions and cache metadata.
        """
        cache_path = Path(cache_path)
        if not cache_path.exists():
            raise FileNotFoundError(f"Cache file not found: {cache_path}")

        raw_sequence = np.load(cache_path).astype(np.float32)
        result = self.predict_sequence(raw_sequence, is_normalized=False, top_k=top_k)
        result["cache_path"] = str(cache_path)
        return result

    def predict_video(
        self,
        video_path: Path | str,
        num_frames: int = 32,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        Run inference directly on input video file (.mp4, etc.).
        
        Args:
            video_path: Path to input video file
            num_frames: Target sequence length to sample
            top_k: Number of top predictions to return
            
        Returns:
            Dict containing predictions and video metadata.
        """
        video_path = Path(video_path)
        raw_sequence = extract_hand_sequence(video_path, num_frames=num_frames)
        result = self.predict_sequence(raw_sequence, is_normalized=False, top_k=top_k)
        result["video_path"] = str(video_path)
        return result

