import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import numpy as np

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
WEIGHTS_PATH = PROJECT_ROOT / "trained_models" / "fingerspelling" / "asl_alphabet_weights.npz"
TFLITE_PATH = PROJECT_ROOT / "trained_models" / "fingerspelling" / "ishara_v1.tflite"
ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


class FingerspellingEngine:
    """
    Hybrid Real-Time ASL Fingerspelling Engine.
    - Uses LiteRT XNNPACK (`ishara_v1.tflite` Squeezeformer+Conformer CTC) for
      92-landmark holistic/pose+hand dictionary/word sequences.
    - Uses fast calibrated 21-hand MLP (`asl_alphabet_weights.npz`) with temporal
      smoothing and degenerate/rest hand gating for 42x3 browser hand streams.
    """

    def __init__(
        self,
        weights_path: Optional[Union[str, Path]] = None,
        model_path: Optional[Union[str, Path]] = None,
        num_threads: int = 4,
    ):
        self.num_threads = int(num_threads)
        self.weights_path = Path(weights_path) if weights_path is not None else WEIGHTS_PATH
        self.model_path = Path(model_path) if model_path is not None else TFLITE_PATH

        self.w1: Optional[np.ndarray] = None
        self.b1: Optional[np.ndarray] = None
        self.w2: Optional[np.ndarray] = None
        self.b2: Optional[np.ndarray] = None
        if self.weights_path.exists():
            data = np.load(self.weights_path)
            self.w1 = data["w1"]  # (63, 128)
            self.b1 = data["b1"]  # (128,)
            self.w2 = data["w2"]  # (128, 26)
            self.b2 = data["b2"]  # (26,)

        from signbridge.fingerspelling.preprocessing import FingerspellingPreprocessor
        from signbridge.fingerspelling.decoder import FingerspellingDecoder

        self.preprocessor = FingerspellingPreprocessor()
        self.decoder = FingerspellingDecoder()
        self._interpreter = None
        self._signature_runner = None
        if self.model_path.exists():
            try:
                from ai_edge_litert.interpreter import Interpreter
                self._interpreter = Interpreter(model_path=str(self.model_path), num_threads=self.num_threads)
                self._signature_runner = self._interpreter.get_signature_runner("serving_default")
            except Exception as e:
                logger.warning(f"LiteRT fingerspelling model unavailable ({self.model_path.name}): {e}")

        if self.w1 is None and self._signature_runner is None:
            raise FileNotFoundError(
                f"Neither weights ({self.weights_path}) nor TFLite model ({self.model_path}) could be loaded."
            )

        logger.info(
            f"FingerspellingEngine initialized (mlp={self.w1 is not None}, tflite={self._signature_runner is not None})"
        )

    def _extract_active_hand(self, frame_landmarks: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract the 21 landmarks of the primary signing hand from a frame.
        Supports inputs of shape (21, 3), (42, 3), or flat arrays.
        """
        arr = np.array(frame_landmarks, dtype=np.float32)

        if arr.ndim == 2:
            if arr.shape == (21, 3):
                # Single hand
                if np.count_nonzero(arr) >= 10:
                    return arr
                return None
            elif arr.shape == (42, 3):
                # Two hands: 0..20 right hand, 21..41 left hand
                rh = arr[:21]
                lh = arr[21:]
                rh_active = np.count_nonzero(rh) >= 10
                lh_active = np.count_nonzero(lh) >= 10

                if rh_active and not lh_active:
                    return rh
                elif lh_active and not rh_active:
                    return lh
                elif rh_active and lh_active:
                    # Pick hand with larger spread / movement
                    rh_std = np.std(rh)
                    lh_std = np.std(lh)
                    return rh if rh_std >= lh_std else lh
                return None
        elif arr.ndim == 1:
            if len(arr) == 63:
                return arr.reshape(21, 3)
            elif len(arr) >= 126:
                return self._extract_active_hand(arr[:126].reshape(42, 3))

        return None

    def _normalize_hand(self, hand_21x3: np.ndarray) -> np.ndarray:
        """Normalize a single hand's 21x3 landmarks into an isotropic [0, 1] bounding box."""
        pts = np.asarray(hand_21x3, dtype=np.float32).copy()
        if pts.shape != (21, 3):
            return pts
        xy_min = pts[:, :2].min(axis=0)
        xy_max = pts[:, :2].max(axis=0)
        span = float(max((xy_max - xy_min).max(), 1e-6))
        pts[:, :2] = (pts[:, :2] - xy_min) / span
        pts[:, 2] = pts[:, 2] - pts[0, 2]
        return pts

    def predict_hand_frame(self, hand_21x3: np.ndarray) -> np.ndarray:
        """Forward pass on 21 hand landmarks, returning 26 class probabilities."""
        norm_hand = self._normalize_hand(hand_21x3)
        x = norm_hand.flatten()  # 63 features
        h = np.maximum(0.0, np.dot(x, self.w1) + self.b1)
        logits = np.dot(h, self.w2) + self.b2
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
        return probs

    def predict(
        self,
        sequence: Union[List[Any], np.ndarray],
        min_char_prob: float = 0.25,
        target_len: Optional[int] = 64,
    ) -> Dict[str, Any]:
        """
        Run inference on sequence of streaming frames.
        
        Args:
            sequence: List or array of frames [T, 42, 3] or [T, 21, 3].
            min_char_prob: Minimum confidence threshold.
            target_len: Unused parameter kept for API signature compatibility.
            
        Returns:
            Dictionary with recognized letter, confidence, stability, and top-k.
        """
        t0 = time.perf_counter()

        # Route holistic (92, 3), pre-flattened (276,), or frame-dict sequences through LiteRT XNNPACK
        use_tflite = False
        if isinstance(sequence, list) and len(sequence) > 0 and isinstance(sequence[0], dict):
            use_tflite = True
        else:
            probe = np.asarray(sequence, dtype=np.float32)
            if probe.size == 0:
                return {
                    "text": "",
                    "char": None,
                    "confidence": 0.0,
                    "stability": 0.0,
                    "is_active": False,
                    "latency_ms": 0.1,
                    "top_k": [],
                    "device": "Apple Silicon M2 Pro",
                }
            if probe.ndim == 3 and probe.shape[1] == 92:
                use_tflite = True
            elif probe.ndim == 2 and probe.shape[1] == 276:
                use_tflite = True
            elif self.w1 is None:
                use_tflite = True

        if use_tflite and self._signature_runner is not None:
            features = self.preprocessor.preprocess_sequence(
                sequence, target_len=target_len or 64, min_len=16
            )
            outputs = self._signature_runner(inputs=features)["outputs"]
            decoded = self.decoder.decode_greedy(outputs, min_char_prob=min_char_prob)
            latency = (time.perf_counter() - t0) * 1000.0
            clean_text = decoded["text"].replace("~", "").strip()
            last_char = clean_text[-1].upper() if clean_text else None
            return {
                "text": clean_text,
                "char": last_char,
                "confidence": round(float(decoded["confidence"]) * 100.0, 1),
                "stability": round(float(decoded["stability"]) * 100.0, 1),
                "is_active": len(clean_text) > 0,
                "latency_ms": round(latency, 2),
                "top_k": [{"char": last_char, "confidence": round(float(decoded["confidence"]) * 100.0, 1)}] if last_char else [],
                "frames_evaluated": int(features.shape[0]),
                "backend": "LiteRT XNNPACK",
                "device": "Apple Silicon M2 Pro",
            }

        seq_arr = np.array(sequence, dtype=np.float32)
        if len(seq_arr) == 0:
            return {
                "text": "",
                "char": None,
                "confidence": 0.0,
                "stability": 0.0,
                "is_active": False,
                "latency_ms": 0.1,
                "top_k": [],
                "device": "Apple Silicon M2 Pro",
            }

        # Analyze the most recent 16 frames and ignore degenerate/collapsed hand shapes
        recent_frames = seq_arr[-16:] if len(seq_arr) > 16 else seq_arr
        valid_probs: List[np.ndarray] = []

        for frame in recent_frames:
            hand = self._extract_active_hand(frame)
            if hand is not None and float(np.ptp(hand[:, :2], axis=0).sum()) >= 0.035:
                p = self.predict_hand_frame(hand)
                valid_probs.append(p)

        latency = (time.perf_counter() - t0) * 1000

        if not valid_probs:
            return {
                "text": "",
                "char": None,
                "confidence": 0.0,
                "stability": 0.0,
                "is_active": False,
                "latency_ms": round(latency, 2),
                "top_k": [],
                "device": "Apple Silicon M2 Pro",
            }

        # Temporal aggregation: exponential weighting towards the most recent frame
        weights = np.exp(np.linspace(-1.0, 0.0, len(valid_probs)))
        weights /= np.sum(weights)
        smooth_probs = np.sum([p * w for p, w in zip(valid_probs, weights)], axis=0)

        best_idx = int(np.argmax(smooth_probs))
        best_char = ALPHABET[best_idx]
        best_conf = float(smooth_probs[best_idx])

        # Top-3 predictions
        sorted_indices = np.argsort(smooth_probs)[::-1]
        top_k = [
            {"char": ALPHABET[i], "confidence": round(float(smooth_probs[i]) * 100, 1)}
            for i in sorted_indices[:3]
        ]

        # Stability: margin between 1st and 2nd top class
        margin = float(smooth_probs[sorted_indices[0]] - smooth_probs[sorted_indices[1]])
        stability = float(np.clip(margin * 1.4, 0.0, 1.0))

        is_confident = best_conf >= min_char_prob
        emitted_text = best_char if is_confident else ""

        return {
            "text": emitted_text,
            "char": best_char if is_confident else None,
            "confidence": round(best_conf * 100, 1),
            "stability": round(stability * 100, 1),
            "is_active": True,
            "latency_ms": round(latency, 2),
            "top_k": top_k,
            "frames_evaluated": len(valid_probs),
            "device": "Apple Silicon M2 Pro",
        }
