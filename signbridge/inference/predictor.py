import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Dict, Any, Tuple, Union, Optional
import numpy as np
import torch
import torch.nn.functional as F

from signbridge.inference.model import GRUClassifier, load_signbridge_model, get_default_device
from signbridge.preprocessing.normalize import normalize_sequence
from signbridge.preprocessing.landmarks import (
    compute_sequence_activity,
    extract_hand_sequence,
    stabilize_hand_sequence,
)


class SignBridgePredictor:
    """
    High-level production predictor for SignBridge model.
    Supports legacy 126/252/261D models and Kinematic-v2 476D Conformer/PyramidBiGRU
    models with OOD NO_SIGN gating.
    """

    def __init__(
        self,
        checkpoint_path: Path | str,
        device: Optional[torch.device] = None,
        ood_threshold: Optional[float] = None,
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self._explicit_device = device is not None
        self.device = device or get_default_device()
        self.model, self.class_names, self.metadata = load_signbridge_model(
            self.checkpoint_path,
            device=self.device,
        )
        # Single-sample RNN inference is faster on Apple M2 CPU (~9ms) than MPS (~30ms)
        if not self._explicit_device and self.metadata.get("architecture") == "SignPyramidBiGRUClassifier":
            self.device = torch.device("cpu")
            self.model = self.model.to(self.device)
            self.model.eval()

        self.input_size = int(self.metadata["input_size"])
        self.stabilize_landmarks = bool(
            self.metadata.get("stabilize_landmarks", self.input_size == 476)
        )
        self.has_ood_head = hasattr(self.model, "forward_with_aux")
        default_ood = max(0.15, float(self.metadata.get("ood_threshold", 0.15)))
        self.ood_threshold = float(ood_threshold if ood_threshold is not None else default_ood)

    def _to_raw_array(self, sequence: np.ndarray) -> np.ndarray:
        seq = np.nan_to_num(sequence, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        if seq.ndim == 2 and seq.shape[1] == 126:
            seq = seq.reshape(seq.shape[0], 42, 3)
        return seq

    def _prepare_tensor(self, sequence: np.ndarray, is_normalized: bool = False) -> torch.Tensor:
        sequence = self._to_raw_array(sequence)

        if sequence.ndim == 3 and sequence.shape[1:] == (42, 3):
            if self.input_size == 476:
                from signbridge.preprocessing.kinematics import build_kinematic_v2_features
                seq_in = stabilize_hand_sequence(sequence) if self.stabilize_landmarks else sequence
                sequence = build_kinematic_v2_features(seq_in)
            else:
                raw_sequence = sequence.copy()
                if not is_normalized:
                    norm_sequence = normalize_sequence(sequence)
                else:
                    norm_sequence = sequence

                if self.input_size == 252:
                    from signbridge.preprocessing.features import build_xyz_velocity_features
                    sequence = build_xyz_velocity_features(norm_sequence)
                elif self.input_size in [255, 258, 261]:
                    from signbridge.preprocessing.features import build_exp14_features
                    mode_map = {255: "a", 258: "b", 261: "c"}
                    sequence = build_exp14_features(raw_sequence, norm_sequence, feature_mode=mode_map[self.input_size])
                else:
                    sequence = norm_sequence.reshape(norm_sequence.shape[0], 126)

        if sequence.shape[1] != self.input_size:
            raise ValueError(
                f"Feature dimension mismatch: expected feature size {self.input_size}, got {sequence.shape[1]}"
            )
        return torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(self.device)

    def extract_embedding(
        self,
        sequence: np.ndarray,
        is_normalized: bool = False,
    ) -> np.ndarray:
        tensor_input = self._prepare_tensor(sequence, is_normalized)
        with torch.no_grad():
            embedding = self.model.extract_embedding(tensor_input)
        return embedding[0].cpu().numpy()

    def _forward_logits_and_aux(
        self,
        sequence: np.ndarray,
        is_normalized: bool = False,
    ) -> Tuple[np.ndarray, float, Optional[np.ndarray], Dict[str, Any]]:
        raw_arr = self._to_raw_array(sequence)
        activity = (
            compute_sequence_activity(raw_arr)
            if raw_arr.ndim == 3 and raw_arr.shape[1:] == (42, 3)
            else {"active_frames": len(raw_arr), "active_ratio": 1.0, "motion_score": 1.0}
        )
        tensor_input = self._prepare_tensor(raw_arr, is_normalized)
        with torch.no_grad():
            if self.has_ood_head:
                logits_t, ood_t, emb_t = self.model.forward_with_aux(tensor_input, None)
                ood_prob = float(torch.sigmoid(ood_t[0]).item())
                emb = emb_t[0].cpu().numpy()
            else:
                logits_t = self.model(tensor_input)
                ood_prob = 1.0
                emb = self.model.extract_embedding(tensor_input)[0].cpu().numpy() if hasattr(self.model, "extract_embedding") else None
        return logits_t[0].cpu().numpy(), ood_prob, emb, activity

    def predict_sequence(
        self,
        sequence: np.ndarray,
        is_normalized: bool = False,
        top_k: int = 5,
        temperature: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Run inference on numpy landmark sequence with kinematic OOD gating.
        """
        t0 = time.perf_counter()
        logits_np, ood_prob, embedding, activity = self._forward_logits_and_aux(sequence, is_normalized)

        logits_t = torch.from_numpy(logits_np) / max(1e-3, float(temperature))
        probs = F.softmax(logits_t, dim=0)
        top_probs, top_indices = torch.topk(probs, k=min(top_k, len(self.class_names)))

        predictions = []
        for prob, idx in zip(top_probs.numpy(), top_indices.numpy()):
            predictions.append(
                {
                    "class_index": int(idx),
                    "gloss": self.class_names[idx],
                    "confidence": float(prob * 100.0),
                }
            )

        raw_gloss = predictions[0]["gloss"]
        raw_conf = predictions[0]["confidence"]
        is_no_sign = _evaluate_no_sign(
            self.has_ood_head, activity, ood_prob, raw_conf, self.ood_threshold
        )

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "predicted_gloss": "NO_SIGN" if is_no_sign else raw_gloss,
            "raw_predicted_gloss": raw_gloss,
            "confidence": round((1.0 - ood_prob) * 100.0, 1) if is_no_sign else raw_conf,
            "raw_confidence": raw_conf,
            "is_no_sign": is_no_sign,
            "ood_prob": round(ood_prob, 4),
            "motion_score": round(float(activity["motion_score"]), 4),
            "active_frames": int(activity["active_frames"]),
            "top_k": predictions,
            "latency_ms": round(elapsed_ms, 1),
            "embedding": embedding.astype(np.float32).tolist() if embedding is not None else [],
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


def _evaluate_no_sign(
    has_ood_head: bool,
    activity: Dict[str, Any],
    ood_prob: float,
    raw_conf: float,
    ood_threshold: float,
) -> bool:
    """Apply OOD gate while preserving high-confidence actively moving strokes."""
    if not has_ood_head:
        return False
    motion_too_low = activity["active_frames"] < 2 or activity["motion_score"] < 0.05
    if motion_too_low:
        return True
    confident_in_motion = (
        activity["active_frames"] >= 5
        and activity["motion_score"] >= 0.09
        and raw_conf >= 62.0
        and ood_prob >= 0.01
    )
    return bool(ood_prob < ood_threshold and not confident_in_motion)


class EnsemblePredictor:
    """
    Weighted probability/logit ensemble across multiple diverse SignBridge predictors
    (e.g., SignConformer on MPS + SignPyramidBiGRU on CPU).
    """

    def __init__(
        self,
        predictors: List[Tuple[SignBridgePredictor, float]],
        temperatures: Optional[List[float]] = None,
        ood_threshold: Optional[float] = None,
    ):
        if not predictors:
            raise ValueError("EnsemblePredictor requires at least one predictor.")
        self.predictors = predictors
        total_w = sum(max(0.0, float(w)) for _, w in predictors)
        self.weights = [max(0.0, float(w)) / max(1e-8, total_w) for _, w in predictors]
        self.temperatures = temperatures or [1.0] * len(predictors)
        self.primary = predictors[0][0]
        self.class_names = self.primary.class_names
        self.input_size = self.primary.input_size
        self.device = self.primary.device
        self.metadata = {
            "architecture": "Ensemble(" + "+".join(
                str(p.metadata.get("architecture", "model")) for p, _ in predictors
            ) + ")",
            "input_size": self.input_size,
            "num_classes": len(self.class_names),
        }
        self.has_ood_head = any(p.has_ood_head for p, _ in predictors)
        self.ood_threshold = float(
            ood_threshold if ood_threshold is not None else max(p.ood_threshold for p, _ in predictors)
        )
        self._executor = ThreadPoolExecutor(max_workers=min(3, len(predictors)))

    def extract_embedding(
        self,
        sequence: np.ndarray,
        is_normalized: bool = False,
    ) -> np.ndarray:
        return self.primary.extract_embedding(sequence, is_normalized)

    def predict_sequence(
        self,
        sequence: np.ndarray,
        is_normalized: bool = False,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        t0 = time.perf_counter()
        if len(self.predictors) == 1:
            return self.primary.predict_sequence(sequence, is_normalized=is_normalized, top_k=top_k)

        futures = [
            self._executor.submit(pred._forward_logits_and_aux, sequence, is_normalized)
            for pred, _ in self.predictors
        ]
        results = [f.result() for f in futures]

        blended_probs: Optional[torch.Tensor] = None
        weighted_ood = 0.0
        ood_w_sum = 0.0
        primary_emb = None
        activity = results[0][3]

        for idx, ((pred, _), (logits_np, ood_prob, emb, act), w, temp) in enumerate(
            zip(self.predictors, results, self.weights, self.temperatures)
        ):
            probs_t = F.softmax(torch.from_numpy(logits_np) / max(1e-3, float(temp)), dim=0)
            blended_probs = probs_t * w if blended_probs is None else blended_probs + probs_t * w
            if pred.has_ood_head:
                weighted_ood += ood_prob * w
                ood_w_sum += w
            if idx == 0:
                primary_emb = emb
                activity = act

        final_ood = weighted_ood / ood_w_sum if ood_w_sum > 1e-6 else 1.0
        top_probs, top_indices = torch.topk(blended_probs, k=min(top_k, len(self.class_names)))
        predictions = [
            {
                "class_index": int(idx),
                "gloss": self.class_names[idx],
                "confidence": float(prob * 100.0),
            }
            for prob, idx in zip(top_probs.numpy(), top_indices.numpy())
        ]

        raw_gloss = predictions[0]["gloss"]
        raw_conf = predictions[0]["confidence"]
        is_no_sign = _evaluate_no_sign(
            self.has_ood_head, activity, final_ood, raw_conf, self.ood_threshold
        )

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "predicted_gloss": "NO_SIGN" if is_no_sign else raw_gloss,
            "raw_predicted_gloss": raw_gloss,
            "confidence": round((1.0 - final_ood) * 100.0, 1) if is_no_sign else raw_conf,
            "raw_confidence": raw_conf,
            "is_no_sign": is_no_sign,
            "ood_prob": round(final_ood, 4),
            "motion_score": round(float(activity["motion_score"]), 4),
            "active_frames": int(activity["active_frames"]),
            "top_k": predictions,
            "latency_ms": round(elapsed_ms, 1),
            "embedding": primary_emb.astype(np.float32).tolist() if primary_emb is not None else [],
        }

    def predict_cache(self, cache_path: Path | str, top_k: int = 5) -> Dict[str, Any]:
        raw_sequence = np.load(Path(cache_path)).astype(np.float32)
        res = self.predict_sequence(raw_sequence, is_normalized=False, top_k=top_k)
        res["cache_path"] = str(cache_path)
        return res

    def predict_video(self, video_path: Path | str, num_frames: int = 32, top_k: int = 5) -> Dict[str, Any]:
        raw_sequence = extract_hand_sequence(Path(video_path), num_frames=num_frames)
        res = self.predict_sequence(raw_sequence, is_normalized=False, top_k=top_k)
        res["video_path"] = str(video_path)
        return res

