from pathlib import Path
from typing import List, Dict, Any, Union, Optional
import json
import logging
import numpy as np

logger = logging.getLogger(__name__)

try:
    import onnxruntime as ort
except ImportError:
    ort = None

from signbridge.preprocessing.normalize import normalize_sequence
from signbridge.preprocessing.features import build_exp14_features, build_xyz_velocity_features


class SignBridgeONNXPredictor:
    """
    Ultra-lightweight, zero-PyTorch production predictor using ONNX Runtime.
    Ideal for serverless deployments (Vercel, AWS Lambda, Cloud Run).
    """

    def __init__(
        self,
        model_path: Optional[Union[Path, str]] = None,
        class_names_path: Optional[Union[Path, str]] = None,
    ):
        base_dir = Path(__file__).resolve().parent.parent.parent
        self.model_path = Path(model_path) if model_path else base_dir / "trained_models" / "production" / "model.onnx"
        self.class_names_path = Path(class_names_path) if class_names_path else base_dir / "trained_models" / "production" / "class_names.json"

        if not self.class_names_path.exists():
            raise FileNotFoundError(f"Class names file not found at: {self.class_names_path}")

        with open(self.class_names_path, "r", encoding="utf-8") as f:
            self.class_names: List[str] = json.load(f)

        self.num_classes = len(self.class_names)
        self.session: Optional[Any] = None
        self.input_name: Optional[str] = None
        self.output_name: Optional[str] = None
        self.input_size: int = 261

        if ort is not None and self.model_path.exists():
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 2
            opts.inter_op_num_threads = 1
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.session = ort.InferenceSession(str(self.model_path), sess_options=opts)
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            input_shape = self.session.get_inputs()[0].shape
            if len(input_shape) >= 3 and isinstance(input_shape[2], int):
                self.input_size = input_shape[2]
            logger.info("ONNX runtime initialized with %d classes, input size %d", self.num_classes, self.input_size)
        else:
            logger.warning(
                "ONNX model or onnxruntime not available at %s. Predictor will run in heuristic fallback mode.",
                self.model_path,
            )

    def _prepare_features(self, sequence: np.ndarray, is_normalized: bool = False) -> np.ndarray:
        sequence = np.nan_to_num(sequence, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        if sequence.ndim == 2:
            sequence = sequence.reshape(sequence.shape[0], 42, 3)

        if sequence.ndim != 3 or sequence.shape[1:] != (42, 3):
            raise ValueError(f"Expected sequence shape (T, 42, 3) or (T, 126), got {sequence.shape}")

        raw_sequence = sequence.copy()
        norm_sequence = sequence if is_normalized else normalize_sequence(sequence)

        if self.input_size == 261:
            features = build_exp14_features(raw_sequence, norm_sequence, feature_mode="c")
        elif self.input_size == 258:
            features = build_exp14_features(raw_sequence, norm_sequence, feature_mode="b")
        elif self.input_size == 255:
            features = build_exp14_features(raw_sequence, norm_sequence, feature_mode="a")
        elif self.input_size == 252:
            features = build_xyz_velocity_features(norm_sequence)
        else:
            features = norm_sequence.reshape(norm_sequence.shape[0], 126)

        return np.expand_dims(features.astype(np.float32), axis=0)  # Shape (1, T, input_size)

    def predict_sequence(
        self,
        sequence: np.ndarray,
        is_normalized: bool = False,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        Run inference on landmark sequence.
        Returns Top-K glosses with confidence percentages matching SignBridgePredictor format.
        """
        features_batch = self._prepare_features(sequence, is_normalized=is_normalized)

        if self.session is not None:
            logits = self.session.run([self.output_name], {self.input_name: features_batch})[0]  # (1, num_classes)
            # Numerically stable softmax
            shift_logits = logits - np.max(logits, axis=1, keepdims=True)
            exp_logits = np.exp(shift_logits)
            probs = (exp_logits / np.sum(exp_logits, axis=1, keepdims=True))[0]
        else:
            # Fallback uniform distribution
            probs = np.ones(self.num_classes, dtype=np.float32) / self.num_classes

        top_k = max(1, min(top_k, self.num_classes))
        top_indices = np.argsort(probs)[::-1][:top_k]

        predictions = []
        for idx in top_indices:
            predictions.append(
                {
                    "class_index": int(idx),
                    "gloss": self.class_names[idx],
                    "confidence": float(probs[idx] * 100.0),
                }
            )

        top1 = predictions[0]
        return {
            "predicted_gloss": top1["gloss"],
            "prediction": top1["gloss"],
            "confidence": round(top1["confidence"], 2),
            "top_k": predictions,
            "embedding": [],
            "engine": "onnxruntime",
            "model_architecture": "Exp11AdapterClassifier (ONNX)",
            "num_classes": self.num_classes,
        }
