from pathlib import Path
from typing import List, Dict, Any, Union, Optional, Tuple
import json
import logging
import os
import time
import numpy as np

logger = logging.getLogger(__name__)

try:
    import onnxruntime as ort
except ImportError:
    ort = None

from signbridge.inference.postprocess import evaluate_no_sign
from signbridge.preprocessing.kinematics import build_kinematic_v2_features
from signbridge.preprocessing.landmarks import compute_sequence_activity, stabilize_hand_sequence
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


class KinematicEnsembleONNXPredictor:
    """
    Zero-PyTorch Ensemble V2 predictor powered by ONNX Runtime (`exp16.onnx` + `exp17.onnx`).
    Runs 476-D kinematic features, temperature-scaled probability blending, OOD gating,
    and 512-D normalized embedding extraction under < 400 MB RSS.
    """

    def __init__(
        self,
        ensemble_config_path: Optional[Union[Path, str]] = None,
        onnx_dir: Optional[Union[Path, str]] = None,
        class_names_path: Optional[Union[Path, str]] = None,
    ):
        if ort is None:
            raise RuntimeError("onnxruntime is required for KinematicEnsembleONNXPredictor")

        base_dir = Path(__file__).resolve().parent.parent.parent
        self.ensemble_config_path = (
            Path(ensemble_config_path)
            if ensemble_config_path
            else base_dir / "trained_models" / "full_2731" / "ensemble_v2" / "ensemble_config.json"
        )
        self.onnx_dir = Path(onnx_dir) if onnx_dir else self.ensemble_config_path.parent
        self.class_names_path = (
            Path(class_names_path)
            if class_names_path
            else base_dir / "trained_models" / "production" / "class_names.json"
        )
        if not self.class_names_path.exists():
            alt_classes = self.onnx_dir / "class_names.json"
            if alt_classes.exists():
                self.class_names_path = alt_classes

        if not self.class_names_path.exists():
            raise FileNotFoundError(f"Class names file not found at: {self.class_names_path}")

        with open(self.class_names_path, "r", encoding="utf-8") as f:
            self.class_names: List[str] = json.load(f)

        cfg = json.loads(self.ensemble_config_path.read_text(encoding="utf-8")) if self.ensemble_config_path.exists() else {}
        self.num_classes = len(self.class_names)
        self.input_size = 476
        self.device = "onnx-cpu"
        self.has_ood_head = True
        self.ood_threshold = float(cfg.get("ood_threshold", 0.4842))

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        opts.enable_cpu_mem_arena = False
        opts.enable_mem_pattern = False
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC

        default_members = [
            {"name": "exp16_kinematic_conformer", "onnx": "exp16.onnx", "weight": 0.55, "temperature": 1.15},
            {"name": "exp17_pyramid_bigru", "onnx": "exp17.onnx", "weight": 0.45, "temperature": 1.15},
        ]
        entries = cfg.get("members", default_members)
        loaded: List[Tuple[Any, float, float, str]] = []
        for entry in entries:
            w = max(0.0, float(entry.get("weight", 1.0)))
            if w <= 0.0:
                continue
            onnx_file = self._resolve_onnx_path(base_dir, entry)
            if onnx_file is None or not onnx_file.exists():
                continue
            sess = ort.InferenceSession(str(onnx_file), sess_options=opts, providers=["CPUExecutionProvider"])
            temp = float(entry.get("temperature", 1.0))
            loaded.append((sess, w, temp, str(entry.get("name", onnx_file.stem))))

        if not loaded:
            raise FileNotFoundError(
                f"No Ensemble V2 ONNX models found under {self.onnx_dir} (expected exp16.onnx / exp17.onnx)."
            )

        total_w = sum(item[1] for item in loaded)
        self.members: List[Tuple[Any, float, float, str]] = [
            (sess, w / max(1e-8, total_w), temp, name) for sess, w, temp, name in loaded
        ]
        self.metadata = {
            "architecture": "EnsembleONNX(" + "+".join(name for _, _, _, name in self.members) + ")",
            "input_size": self.input_size,
            "num_classes": self.num_classes,
            "engine": "onnxruntime",
        }
        logger.info(
            "Initialized KinematicEnsembleONNXPredictor with %d members (%d classes).",
            len(self.members),
            self.num_classes,
        )

    def _resolve_onnx_path(self, base_dir: Path, entry: Dict[str, Any]) -> Optional[Path]:
        candidates: List[Path] = []
        prefer_int8 = os.environ.get("SIGNBRIDGE_ONNX_INT8", "").strip().lower() in {"1", "true", "yes"}
        if entry.get("onnx"):
            p = Path(entry["onnx"])
            candidates.append(p if p.is_absolute() else base_dir / p)
            candidates.append(self.onnx_dir / p.name)
        name = str(entry.get("name", ""))
        short_base = "exp16" if "exp16" in name else ("exp17" if "exp17" in name else name)
        filenames = (
            [f"{short_base}.int8.onnx", f"{name}.int8.onnx", f"{name}.onnx", f"{short_base}.onnx"]
            if prefer_int8
            else [f"{name}.onnx", f"{short_base}.onnx", f"{short_base}.int8.onnx", f"{name}.int8.onnx"]
        )
        for search_dir in (self.onnx_dir, self.ensemble_config_path.parent, base_dir / "trained_models" / "onnx_v2"):
            for fn in filenames:
                if fn and fn != ".onnx":
                    candidates.append(search_dir / fn)
        for cand in candidates:
            if cand.exists():
                return cand
        return candidates[0] if candidates else None

    @staticmethod
    def _to_raw_array(sequence: np.ndarray) -> np.ndarray:
        arr = np.nan_to_num(np.asarray(sequence, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        if arr.ndim == 2 and arr.shape[1] == 126:
            arr = arr.reshape(arr.shape[0], 42, 3)
        return arr

    def _prepare_v2_batch(self, sequence: np.ndarray) -> np.ndarray:
        raw_arr = self._to_raw_array(sequence)
        if raw_arr.ndim == 3 and raw_arr.shape[1:] == (42, 3):
            feats = build_kinematic_v2_features(stabilize_hand_sequence(raw_arr))
        elif raw_arr.ndim == 2 and raw_arr.shape[1] == self.input_size:
            feats = raw_arr
        else:
            raise ValueError(
                f"Expected shape (T, 42, 3), (T, 126), or (T, {self.input_size}), got {raw_arr.shape}"
            )
        return np.expand_dims(feats.astype(np.float32), axis=0)

    def extract_embedding(
        self,
        sequence: np.ndarray,
        is_normalized: bool = False,
    ) -> np.ndarray:
        _ = is_normalized
        feats_batch = self._prepare_v2_batch(sequence)
        sess = self.members[0][0]
        _, _, emb = sess.run(None, {"features": feats_batch})
        return np.asarray(emb[0], dtype=np.float32)

    def predict_sequence(
        self,
        sequence: np.ndarray,
        is_normalized: bool = False,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        _ = is_normalized
        t0 = time.perf_counter()
        raw_arr = self._to_raw_array(sequence)
        activity = (
            compute_sequence_activity(raw_arr)
            if raw_arr.ndim == 3 and raw_arr.shape[1:] == (42, 3)
            else {"active_frames": len(raw_arr), "active_ratio": 1.0, "motion_score": 1.0}
        )
        feats_batch = self._prepare_v2_batch(raw_arr)

        blended_probs: Optional[np.ndarray] = None
        weighted_ood = 0.0
        primary_emb: Optional[np.ndarray] = None

        for idx, (sess, w, temp, _) in enumerate(self.members):
            logits, ood_logit, emb = sess.run(None, {"features": feats_batch})
            scaled = logits[0].astype(np.float32) / max(1e-3, float(temp))
            scaled = scaled - np.max(scaled)
            exp_vals = np.exp(scaled)
            probs = exp_vals / np.sum(exp_vals)
            blended_probs = probs * w if blended_probs is None else blended_probs + probs * w

            ood_val = float(ood_logit[0])
            ood_prob = float(1.0 / (1.0 + np.exp(-np.clip(ood_val, -60.0, 60.0))))
            weighted_ood += ood_prob * w
            if idx == 0:
                primary_emb = np.asarray(emb[0], dtype=np.float32)

        assert blended_probs is not None
        k = max(1, min(top_k, self.num_classes))
        top_indices = np.argsort(blended_probs)[::-1][:k]
        predictions = [
            {
                "class_index": int(idx),
                "gloss": self.class_names[int(idx)],
                "confidence": float(blended_probs[int(idx)] * 100.0),
            }
            for idx in top_indices
        ]

        raw_gloss = predictions[0]["gloss"]
        raw_conf = predictions[0]["confidence"]
        is_no_sign = evaluate_no_sign(
            self.has_ood_head, activity, weighted_ood, raw_conf, self.ood_threshold
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "predicted_gloss": "NO_SIGN" if is_no_sign else raw_gloss,
            "raw_predicted_gloss": raw_gloss,
            "confidence": round((1.0 - weighted_ood) * 100.0, 1) if is_no_sign else raw_conf,
            "raw_confidence": raw_conf,
            "is_no_sign": is_no_sign,
            "ood_prob": round(weighted_ood, 4),
            "motion_score": round(float(activity["motion_score"]), 4),
            "active_frames": int(activity["active_frames"]),
            "top_k": predictions,
            "latency_ms": round(elapsed_ms, 1),
            "embedding": primary_emb.tolist() if primary_emb is not None else [],
            "engine": "onnxruntime",
        }

    def predict_cache(self, cache_path: Path | str, top_k: int = 5) -> Dict[str, Any]:
        raw_sequence = np.load(Path(cache_path)).astype(np.float32)
        res = self.predict_sequence(raw_sequence, is_normalized=False, top_k=top_k)
        res["cache_path"] = str(cache_path)
        return res

    def predict_video(self, video_path: Path | str, num_frames: int = 32, top_k: int = 5) -> Dict[str, Any]:
        from signbridge.preprocessing.landmarks import extract_hand_sequence

        raw_sequence = extract_hand_sequence(Path(video_path), num_frames=num_frames)
        res = self.predict_sequence(raw_sequence, is_normalized=False, top_k=top_k)
        res["video_path"] = str(video_path)
        return res
