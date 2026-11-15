import os
import platform
import random
import shutil
import subprocess
import tempfile
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

logger = logging.getLogger(__name__)

import csv
import numpy as np
import re
from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

try:
    import torch as _torch  # noqa: F401
    from signbridge.inference.predictor import EnsemblePredictor, SignBridgePredictor
    _TORCH_AVAILABLE = True
except ImportError:
    EnsemblePredictor = Any  # type: ignore[misc,assignment]
    SignBridgePredictor = Any  # type: ignore[misc,assignment]
    _TORCH_AVAILABLE = False

from signbridge.inference.onnx_predictor import KinematicEnsembleONNXPredictor
from signbridge.continuous.session import ContinuousCSLRSessionManager
from signbridge.personalization.manager import PersonalizationManager
from signbridge.personalization.router import PersonalizationRouter
from signbridge.language.qwen_service import QwenLanguageService
from signbridge.language.groq_service import GroqLanguageService
from signbridge.language.asl_grammar import asl_rules_to_english
from signbridge.tts.service import KokoroTTSService

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PRODUCTION_CHECKPOINT = PROJECT_ROOT / "trained_models" / "production" / "best_model.pth"
EXP17_CHECKPOINT = PROJECT_ROOT / "trained_models" / "full_2731" / "exp17_pyramid_bigru" / "best_model.pth"
EXP16_CHECKPOINT = PROJECT_ROOT / "trained_models" / "full_2731" / "exp16_kinematic_conformer" / "best_model.pth"
ENSEMBLE_CONFIG_PATH = PROJECT_ROOT / "trained_models" / "full_2731" / "ensemble_v2" / "ensemble_config.json"
EXP13_CHECKPOINT = PROJECT_ROOT / "trained_models" / "full_2731" / "exp13_arcface035_skeletonwarp" / "best_model.pth"
EXP11_CHECKPOINT = PROJECT_ROOT / "trained_models" / "full_2731" / "exp11_adapter_velocity" / "best_model.pth"
EXP9_CHECKPOINT = PROJECT_ROOT / "trained_models" / "full_2731" / "exp9_true_arcface" / "best_model.pth"
FALLBACK_CHECKPOINT = PROJECT_ROOT / "trained_models" / "best_gru_normalized.pth"
DEFAULT_CHECKPOINT = PRODUCTION_CHECKPOINT if PRODUCTION_CHECKPOINT.exists() else FALLBACK_CHECKPOINT
VIDEOS_DIR = PROJECT_ROOT / "dataset" / "ASL_Citizen" / "videos"
SPLITS_DIR = PROJECT_ROOT / "dataset" / "ASL_Citizen" / "splits"
STATIC_DIR = Path(__file__).resolve().parent / "static"
TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"

app = FastAPI(
    title="SignBridge API & Application",
    description="Sign Language Recognition & Learning System API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR.mkdir(parents=True, exist_ok=True)
TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)
ASSETS_DIR = STATIC_DIR / "assets"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/assets", StaticFiles(directory=str(ASSETS_DIR)), name="assets")

_predictors: Dict[str, Any] = {}
_gloss_to_videos: Dict[str, List[str]] = {}
_demo_ready_glosses: set = set()
_elite_glosses: set = set()
_personalization_manager = PersonalizationManager(
    data_dir=PROJECT_ROOT / "data" / "personalization"
)
_personalization_manager.set_config(_personalization_manager.config)
_personalization_router = PersonalizationRouter(_personalization_manager)

_groq_service = GroqLanguageService()
_qwen_service = QwenLanguageService()
_tts_service = KokoroTTSService()
_continuous_manager = ContinuousCSLRSessionManager()

MODEL_CHECKPOINTS = {
    "production_exp13": PRODUCTION_CHECKPOINT,
    "production_v2": ENSEMBLE_CONFIG_PATH,
    "exp16_conformer": EXP16_CHECKPOINT,
    "exp17_bigru": EXP17_CHECKPOINT,
    "legacy_exp14c": PRODUCTION_CHECKPOINT,
    "production_exp7": PRODUCTION_CHECKPOINT,
    "candidate_exp13": EXP13_CHECKPOINT,
    "candidate_exp11": EXP11_CHECKPOINT,
    "candidate_exp9": EXP9_CHECKPOINT,
    "demo_50": PROJECT_ROOT / "trained_models" / "demo_50" / "exp3_augmented" / "best_model.pth",
}


def _use_onnx_backend() -> bool:
    force_onnx = os.environ.get("SIGNBRIDGE_USE_ONNX", "").strip().lower() in {"1", "true", "yes"}
    return force_onnx or not _TORCH_AVAILABLE


def _onnx_ensemble_available() -> bool:
    onnx_dirs = [ENSEMBLE_CONFIG_PATH.parent, PROJECT_ROOT / "trained_models" / "onnx_v2"]
    names = ("exp16.onnx", "exp17.onnx", "exp16.int8.onnx", "exp17.int8.onnx")
    return any((d / fn).exists() for d in onnx_dirs for fn in names)


def _load_onnx_ensemble() -> Optional[KinematicEnsembleONNXPredictor]:
    onnx_dir = ENSEMBLE_CONFIG_PATH.parent
    names = ("exp16.onnx", "exp17.onnx", "exp16.int8.onnx", "exp17.int8.onnx")
    if not any((onnx_dir / fn).exists() for fn in names):
        alt_dir = PROJECT_ROOT / "trained_models" / "onnx_v2"
        if any((alt_dir / fn).exists() for fn in names):
            onnx_dir = alt_dir
    return KinematicEnsembleONNXPredictor(
        ensemble_config_path=ENSEMBLE_CONFIG_PATH,
        onnx_dir=onnx_dir,
    )


def _load_ensemble_from_config(cfg_path: Path):
    if _use_onnx_backend():
        return _load_onnx_ensemble()
    cfg = json.loads(cfg_path.read_text())
    members = []
    temps = []
    for entry in cfg.get("members", []):
        ckpt_p = Path(entry["checkpoint"])
        if not ckpt_p.is_absolute():
            ckpt_p = PROJECT_ROOT / ckpt_p
        if not ckpt_p.exists():
            continue
        w = float(entry.get("weight", 1.0))
        if w <= 0.0:
            continue
        members.append((SignBridgePredictor(checkpoint_path=ckpt_p, ood_threshold=cfg.get("ood_threshold")), w))
        temps.append(float(entry.get("temperature", 1.0)))
    if not members:
        return None
    if len(members) == 1:
        return members[0][0]
    return EnsemblePredictor(
        members,
        temperatures=temps,
        ood_threshold=cfg.get("ood_threshold"),
    )


def get_predictor(model_variant: str = "production_exp13") -> Any:
    if model_variant not in MODEL_CHECKPOINTS:
        model_variant = "production_exp13"
    if model_variant in _predictors:
        return _predictors[model_variant]

    if _use_onnx_backend() and _onnx_ensemble_available():
        try:
            ens_onnx = _load_onnx_ensemble()
            if ens_onnx is not None:
                _predictors[model_variant] = ens_onnx
                return ens_onnx
        except Exception as e:
            logger.warning(f"Failed loading ONNX ensemble_v2: {e}")
            if not _TORCH_AVAILABLE:
                raise

    if not _TORCH_AVAILABLE:
        raise RuntimeError(
            "PyTorch is not installed and Ensemble V2 ONNX weights could not be loaded."
        )

    # Route production_exp13 / production_v2 through promoted ensemble_v2 when ready
    if model_variant in ("production_exp13", "production_v2") and ENSEMBLE_CONFIG_PATH.exists():
        try:
            cfg = json.loads(ENSEMBLE_CONFIG_PATH.read_text())
            if cfg.get("promoted", False):
                ens = _load_ensemble_from_config(ENSEMBLE_CONFIG_PATH)
                if ens is not None:
                    _predictors[model_variant] = ens
                    return ens
        except Exception as e:
            logger.warning(f"Failed loading ensemble_v2 config, falling back cleanly: {e}")

    checkpoint = MODEL_CHECKPOINTS[model_variant]
    if checkpoint == ENSEMBLE_CONFIG_PATH:
        checkpoint = EXP16_CHECKPOINT if EXP16_CHECKPOINT.exists() else DEFAULT_CHECKPOINT
    elif not checkpoint.exists():
        checkpoint = DEFAULT_CHECKPOINT
    _predictors[model_variant] = SignBridgePredictor(checkpoint_path=checkpoint)
    return _predictors[model_variant]


def _read_csv_col(path: Path, col: str) -> List[str]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return [str(row[col]).strip() for row in csv.DictReader(f) if row.get(col)]


def init_dataset_mappings():
    global _gloss_to_videos, _demo_ready_glosses, _elite_glosses
    if _gloss_to_videos:
        return

    for split in ["train.csv", "val.csv", "test.csv"]:
        csv_path = SPLITS_DIR / split
        if not csv_path.exists():
            continue
        with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                g = (row.get("Gloss") or "").strip().upper()
                vf = (row.get("Video file") or "").strip()
                if g and vf:
                    _gloss_to_videos.setdefault(g, []).append(vf)

    _demo_ready_glosses = {g.upper() for g in _read_csv_col(PROJECT_ROOT / "signbridge_final_demo_glosses.csv", "Gloss")}
    _elite_glosses = {g.upper() for g in _read_csv_col(PROJECT_ROOT / "signbridge_elite_glosses.csv", "Gloss")}


@app.on_event("startup")
async def startup_event():
    try:
        init_dataset_mappings()
        predictor = get_predictor()
        print(f"✅ SignBridge loaded {len(predictor.class_names)} glosses and dataset video mapping.")
    except Exception as e:
        print(f"⚠️ Startup warning: {e}")


# -------------------------------------------------------------------
# Pydantic Schemas
# -------------------------------------------------------------------

class SequenceRequest(BaseModel):
    sequence: List[List[List[float]]] = Field(..., description="Shape (T, 42, 3)")
    top_k: Optional[int] = Field(default=5, ge=1, le=20)
    mode: Optional[str] = Field(default=None, description="Capture mode for diagnostics")
    window_number: Optional[int] = Field(default=None, ge=1)
    hand_stats: Dict[str, Any] = Field(default_factory=dict)
    diagnostic: bool = False
    model_variant: str = "production_exp7"
    user_id: str = "default"
    personalization_enabled: bool = False


class FlatSequenceRequest(BaseModel):
    sequence: List[List[float]] = Field(..., description="Shape (T, 126)")
    top_k: Optional[int] = Field(default=5, ge=1, le=20)
    mode: Optional[str] = Field(default=None, description="Capture mode for diagnostics")
    window_number: Optional[int] = Field(default=None, ge=1)
    hand_stats: Dict[str, Any] = Field(default_factory=dict)
    diagnostic: bool = False
    model_variant: str = "production_exp7"
    user_id: str = "default"
    personalization_enabled: bool = False


class FingerspellingRequest(BaseModel):
    sequence: List[Any] = Field(..., description="List of frames containing landmarks: (T, 92, 3), (T, 276), (T, 42, 3), or frame dicts")
    min_char_prob: Optional[float] = Field(default=0.25, ge=0.0, le=1.0)
    target_len: Optional[int] = Field(default=64, ge=16, le=384)


class PersonalizationSampleRequest(BaseModel):
    sequence: List[List[List[float]]] = Field(..., description="Shape (32, 42, 3)")
    gloss: str = Field(..., min_length=1, max_length=100)
    user_id: str = Field(default="default", min_length=1, max_length=64)
    model_variant: str = "production_exp13"


class PersonalizationUserRequest(BaseModel):
    user_id: str = Field(default="default", min_length=1, max_length=64)


def _safe_user_id(user_id: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", user_id):
        raise HTTPException(status_code=400, detail="user_id may contain only letters, numbers, '_' and '-'.")
    return user_id


def _personalized_result(
    result: Dict[str, Any],
    user_id: str,
    enabled: bool,
    sequence_np: Optional[Any] = None,
) -> Dict[str, Any]:
    uid = _safe_user_id(user_id)
    if not enabled:
        return _personalization_router._no_personalization(result.get("top_k", []))
    prototypes = _personalization_manager.get_all_prototypes(uid)
    if not prototypes:
        return _personalization_router._no_personalization(result.get("top_k", []))

    raw_emb = result.get("embedding")
    if (not raw_emb or len(raw_emb) == 0) and sequence_np is not None:
        # Ensure consistent 512-d Conformer embedding even when running Demo 50 fast variant
        try:
            pro_pred = get_predictor("production_exp13")
            emb_vec = pro_pred.extract_embedding(sequence_np, is_normalized=False)
            raw_emb = emb_vec.astype(np.float32)
        except Exception:
            raw_emb = None

    sample_banks = _personalization_manager.get_all_sample_banks(uid)
    return _personalization_router.route_prediction(
        result.get("top_k", []),
        np.asarray(raw_emb, dtype=np.float32) if raw_emb is not None and len(raw_emb) > 0 else None,
        prototypes,
        user_id=uid,
        sample_banks=sample_banks,
    )


def _apply_personalization_override(
    result: Dict[str, Any],
    sequence_np: Any,
    user_id: str,
    enabled: bool,
) -> Dict[str, Any]:
    pers = _personalized_result(result, user_id, enabled, sequence_np=sequence_np)
    result["personalization"] = pers
    result["is_custom_sign"] = False
    has_motion = (
        int(result.get("active_frames", 0)) >= 6
        and float(result.get("motion_score", 0.0)) >= 0.008
    )
    can_override = (
        bool(pers.get("matched"))
        and bool(pers.get("override_gloss"))
        and (not result.get("is_no_sign", False) or (pers.get("mode") == "override" and has_motion))
    )
    if can_override:
        ov_g = str(pers["override_gloss"])
        ov_c = float(pers.get("confidence", result.get("confidence", 0.0)))
        merged_conf = round(max(float(result.get("confidence", 0.0)) if not result.get("is_no_sign", False) else 0.0, ov_c), 2)
        result["predicted_gloss"] = ov_g
        result["gloss"] = ov_g
        result["confidence"] = merged_conf
        result["is_no_sign"] = False
        result["is_custom_sign"] = True
        result["top_k"] = [{"gloss": ov_g, "confidence": merged_conf, "custom": True}] + [
            c for c in result.get("top_k", []) if str(c.get("gloss", "")).upper() != ov_g.upper()
        ][:4]
    return result


# -------------------------------------------------------------------
# Core Web & API Endpoints
# -------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_path = TEMPLATES_DIR / "index.html"
    if not index_path.exists():
        return HTMLResponse("<h1>SignBridge API Running</h1>")
    with open(index_path, "r", encoding="utf-8") as f:
        content = f.read()
    return HTMLResponse(content=content)


_cached_hardware_device: Optional[str] = None
_hardware_checked: bool = False


def _detect_hardware_device() -> Optional[str]:
    """Read local hardware compute/chip identifier without inventing values."""
    global _cached_hardware_device, _hardware_checked
    if _hardware_checked:
        return _cached_hardware_device
    _hardware_checked = True
    try:
        if platform.system() == "Darwin":
            out = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=1.5,
            ).strip()
            if out:
                _cached_hardware_device = out.removeprefix("Apple ").strip()
                return _cached_hardware_device
        proc = platform.processor().strip()
        if proc:
            _cached_hardware_device = proc
    except Exception:
        _cached_hardware_device = None
    return _cached_hardware_device


def _load_evaluation_metrics() -> Dict[str, Any]:
    """Load verified held-out evaluation metrics from saved experiment artifacts."""
    metrics: Dict[str, Any] = {}
    exp13_eval_path = EXP13_CHECKPOINT.parent / "eval_results.json"
    if exp13_eval_path.exists():
        try:
            with open(exp13_eval_path, "r", encoding="utf-8") as f:
                exp13_data = json.load(f)
            metrics["exp13"] = {
                "experiment": exp13_data.get("experiment", "exp13_arcface035_skeletonwarp"),
                "top1_accuracy": round(float(exp13_data["test_top1_accuracy"]), 2),
                "top3_accuracy": round(float(exp13_data.get("test_top3_accuracy", 0.0)), 2),
                "top5_accuracy": round(float(exp13_data["test_top5_accuracy"]), 2),
            }
        except Exception:
            pass

    prod_meta_path = PRODUCTION_CHECKPOINT.parent / "model_metadata.json"
    if prod_meta_path.exists():
        try:
            with open(prod_meta_path, "r", encoding="utf-8") as f:
                prod_meta = json.load(f)
            test_m = prod_meta.get("test_metrics", {})
            metrics["production"] = {
                "experiment": prod_meta.get("source_experiment", "production"),
                "top1_accuracy": round(float(test_m["test_top1_accuracy"]), 2) if "test_top1_accuracy" in test_m else None,
                "top5_accuracy": round(float(test_m["test_top5_accuracy"]), 2) if "test_top5_accuracy" in test_m else None,
            }
        except Exception:
            pass

    if ENSEMBLE_CONFIG_PATH.exists():
        try:
            cfg = json.loads(ENSEMBLE_CONFIG_PATH.read_text(encoding="utf-8"))
            tsum = cfg.get("test_summary", {})
            if cfg.get("promoted") and "test_top1" in tsum:
                metrics["production"] = {
                    "experiment": cfg.get("version", "ensemble_v2_kinematic_476"),
                    "top1_accuracy": round(float(tsum["test_top1"]), 2),
                    "top3_accuracy": round(float(tsum.get("test_top3", 0.0)), 2),
                    "top5_accuracy": round(float(tsum["test_top5"]), 2),
                    "macro_f1": round(float(tsum.get("test_macro_f1", 0.0)), 2),
                    "mirrored_2k_top1": round(float(tsum.get("mirrored_2k_top1", 0.0)), 2),
                    "ood_auroc": round(float(tsum.get("ood_auroc", 0.0)), 5),
                }
        except Exception:
            pass
    return metrics


@app.get("/api/health")
async def health_check():
    predictor = get_predictor()
    init_dataset_mappings()
    fs_weights = PROJECT_ROOT / "trained_models" / "fingerspelling" / "asl_alphabet_weights.npz"
    onnx_ready = _use_onnx_backend() and _onnx_ensemble_available()
    return {
        "status": "healthy",
        "device": str(predictor.device),
        "hardware_device": _detect_hardware_device(),
        "engine": "onnxruntime" if str(predictor.device).startswith("onnx") else "pytorch",
        "num_classes": len(predictor.class_names),
        "total_videos_mapped": len(_gloss_to_videos),
        "fingerspelling_ready": fs_weights.exists(),
        "evaluation_metrics": _load_evaluation_metrics(),
        "model_variants": {
            name: (
                path.exists()
                or (onnx_ready and name in ("production_exp13", "production_v2", "exp16_conformer", "exp17_bigru"))
            )
            for name, path in MODEL_CHECKPOINTS.items()
        },
    }


# -------------------------------------------------------------------
# Learn & Practice Endpoints
# -------------------------------------------------------------------

@app.get("/api/classes")
async def get_all_classes():
    """Return complete list of supported ASL gloss classes."""
    predictor = get_predictor()
    return {"count": len(predictor.class_names), "classes": predictor.class_names}


_dictionary_cache: Optional[Dict[str, Any]] = None


def _get_dictionary_cache() -> Optional[Dict[str, Any]]:
    """Lazily load precomputed 2,731-gloss 3D hand exemplar cache."""
    global _dictionary_cache
    if _dictionary_cache is not None:
        return _dictionary_cache
    npz_path = PROJECT_ROOT / "data" / "dictionary_exemplars_32.npz"
    if not npz_path.exists():
        return None
    with np.load(npz_path, allow_pickle=True) as raw:
        names = [str(g).upper() for g in raw["class_names"].tolist()]
        frames_arr = raw["frames"]
        frames = frames_arr if frames_arr.dtype == np.float16 else frames_arr.astype(np.float16)
        _dictionary_cache = {
            "index": {g: i for i, g in enumerate(names)},
            "frames": frames,
            "two_hand": raw["two_hand"],
            "handshapes": raw["handshapes"],
            "steps_json": raw["steps_json"],
        }
    return _dictionary_cache


@app.get("/api/vocabulary")
async def get_vocabulary(
    q: Optional[str] = Query(None, description="Search query string"),
    limit: int = Query(60, ge=1, le=10000),
    category: Optional[str] = Query(None, description="Filter by category (elite, demo, one_hand, two_hand, all)"),
):
    """Browse or search supported sign vocabulary."""
    predictor = get_predictor("production_exp13")
    init_dataset_mappings()
    dict_cache = _get_dictionary_cache()

    glosses = predictor.class_names
    query = (q or "").strip().upper()

    if query:
        glosses = [g for g in glosses if query in g]

    if category == "elite" and _elite_glosses:
        glosses = [g for g in glosses if g in _elite_glosses]
    elif category == "demo" and _demo_ready_glosses:
        glosses = [g for g in glosses if g in _demo_ready_glosses]
    elif category in ("two_hand", "one_hand") and dict_cache is not None:
        want_two = category == "two_hand"
        idx_map = dict_cache["index"]
        th_arr = dict_cache["two_hand"]
        glosses = [g for g in glosses if (g in idx_map and bool(th_arr[idx_map[g]]) == want_two)]

    glosses_slice = glosses[:limit]

    items = []
    for g in glosses_slice:
        video_count = len(_gloss_to_videos.get(g, []))
        is_elite = g in _elite_glosses
        is_demo = g in _demo_ready_glosses
        tier = "High Precision" if is_elite else ("Verified Demo" if is_demo else "Standard")
        two_hand = False
        handshape = ""
        if dict_cache is not None and g in dict_cache["index"]:
            ci = dict_cache["index"][g]
            two_hand = bool(dict_cache["two_hand"][ci])
            handshape = str(dict_cache["handshapes"][ci])

        items.append({
            "gloss": g,
            "video_count": video_count,
            "has_demonstration": video_count > 0,
            "tier": tier,
            "two_hand": two_hand,
            "handshape": handshape,
        })

    return {
        "total_matches": len(glosses),
        "returned": len(items),
        "items": items,
    }


@app.get("/api/demonstration/{gloss}")
async def get_demonstration(gloss: str):
    """Get ASL Citizen dataset demonstration video details for a given gloss."""
    predictor = get_predictor()
    init_dataset_mappings()

    gloss_upper = gloss.strip().upper()
    if gloss_upper not in predictor.class_names:
        raise HTTPException(status_code=404, detail=f"Gloss '{gloss}' not in model vocabulary")

    videos = _gloss_to_videos.get(gloss_upper, [])
    if not videos:
        return {
            "gloss": gloss_upper,
            "has_demonstration": False,
            "message": "No demonstration video available for this sign yet.",
            "video_url": None,
        }

    demo_video = videos[0]

    return {
        "gloss": gloss_upper,
        "has_demonstration": True,
        "video_filename": demo_video,
        "video_url": f"/api/video/{demo_video}",
        "total_available_examples": len(videos),
        "tier": "High Precision" if gloss_upper in _elite_glosses else "Standard",
    }


@app.api_route("/api/video/{filename}", methods=["GET", "HEAD"])
async def stream_video(filename: str):
    """Stream ASL Citizen dataset video file for learning demonstration."""
    video_path = VIDEOS_DIR / Path(filename).name
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video asset not found")
    return FileResponse(path=video_path, media_type="video/mp4", filename=video_path.name)


@app.get("/api/teach-me")
async def teach_me_something():
    """🎲 Randomly pick a high-quality sign for the user to learn."""
    init_dataset_mappings()
    predictor = get_predictor()

    pool = list(_elite_glosses or _demo_ready_glosses or predictor.class_names)
    selected_gloss = random.choice(pool)

    return await get_demonstration(selected_gloss)


@app.get("/api/dictionary/sign/{gloss:path}")
async def get_dictionary_sign(gloss: str):
    """Return 32-frame 3D landmark sequence & anatomical signing guide for any of 2,731 glosses."""
    predictor = get_predictor()
    init_dataset_mappings()
    gloss_upper = gloss.strip().upper()
    if gloss_upper not in predictor.class_names:
        raise HTTPException(status_code=404, detail=f"Gloss '{gloss}' not in model vocabulary")

    cache = _get_dictionary_cache()
    if cache is None or gloss_upper not in cache["index"]:
        raise HTTPException(status_code=404, detail=f"3D exemplar cache unavailable for '{gloss_upper}'")

    ci = cache["index"][gloss_upper]
    frames_arr = np.round(cache["frames"][ci].astype(np.float32), 4)
    steps = json.loads(str(cache["steps_json"][ci]))
    videos = _gloss_to_videos.get(gloss_upper, [])

    return {
        "gloss": gloss_upper,
        "tier": "High Precision" if gloss_upper in _elite_glosses else "Standard",
        "video_count": len(videos),
        "video_url": f"/api/video/{videos[0]}" if videos else None,
        "two_hand": bool(cache["two_hand"][ci]),
        "handshape": str(cache["handshapes"][ci]),
        "steps": steps,
        "total_frames": int(frames_arr.shape[0]),
        "frames": frames_arr.tolist(),
    }


# -------------------------------------------------------------------
# Prediction Endpoints
# -------------------------------------------------------------------

@app.post("/api/predict/sequence")
async def predict_sequence_endpoint(request: Union[SequenceRequest, FlatSequenceRequest]):
    """
    Run inference on live camera landmark sequence.
    Sequence is validated, normalized via official normalize_sequence(), and classified.
    """
    try:
        predictor = get_predictor(request.model_variant)
        import numpy as np

        sequence_np = np.array(request.sequence, dtype=np.float32)
        top_k = request.top_k or 5

        if sequence_np.ndim not in (2, 3):
            raise HTTPException(status_code=400, detail=f"Invalid sequence array dimensions: {sequence_np.ndim}")

        # Ensure (T, 42, 3) shape
        if sequence_np.ndim == 2 and sequence_np.shape[1] == 126:
            sequence_np = sequence_np.reshape(sequence_np.shape[0], 42, 3)

        if sequence_np.shape[1:] != (42, 3):
            raise HTTPException(
                status_code=400,
                detail=f"Expected shape (T, 42, 3) or (T, 126), got {sequence_np.shape}"
            )

        result = predictor.predict_sequence(sequence_np, is_normalized=False, top_k=top_k)
        _apply_personalization_override(
            result, sequence_np, request.user_id, request.personalization_enabled
        )
        if request.diagnostic:
            result["diagnostics"] = build_prediction_diagnostics(
                sequence_np,
                result,
                mode=request.mode,
                window_number=request.window_number,
                hand_stats=request.hand_stats,
                model_variant=request.model_variant,
            )
        return JSONResponse(content=result)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sequence prediction error: {str(e)}")


class ContinuousStreamRequest(BaseModel):
    session_id: str = Field(default="default", min_length=1, max_length=64)
    frames: List[List[List[float]]] = Field(..., description="Batch of incremental (42, 3) landmark frames")
    timestamps_ms: Optional[List[float]] = None
    model_variant: str = "production_exp13"
    user_id: str = "default"
    personalization_enabled: bool = False
    auto_finalize_on_pause: bool = True
    auto_translate_on_pause: bool = True
    top_k: int = Field(default=5, ge=1, le=20)


class ContinuousFinalizeRequest(BaseModel):
    session_id: str = Field(default="default", min_length=1, max_length=64)
    model_variant: str = "production_exp13"
    user_id: str = "default"
    personalization_enabled: bool = True
    translate: bool = True
    clear_after: bool = True


class ContinuousResetRequest(BaseModel):
    session_id: str = Field(default="default", min_length=1, max_length=64)


def _translate_gloss_sequence(
    glosses: List[str],
    hypotheses: Optional[List[List[Dict[str, Any]]]] = None,
    non_manual: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    cleaned = [g.strip().upper().rstrip("0123456789").removesuffix("/IT") for g in glosses if g.strip()]
    if not cleaned:
        return {"english": "", "glosses": [], "latency_ms": 0.0, "engine": "None", "status": "empty"}

    res = _groq_service.translate_glosses(cleaned, hypotheses=hypotheses, non_manual=non_manual)
    if not res.get("english"):
        res = _qwen_service.translate_glosses(cleaned, hypotheses=hypotheses, non_manual=non_manual)
    if res.get("status") == "success" and res.get("english"):
        return {
            "english": res["english"],
            "glosses": cleaned,
            "latency_ms": res["latency_ms"],
            "engine": res.get("engine", "Groq AI"),
            "llm_available": _groq_service.is_available() or _qwen_service.is_available(),
            "status": "success",
            "error": None,
        }
    english = asl_rules_to_english(cleaned, non_manual=non_manual)
    return {
        "english": english,
        "glosses": cleaned,
        "latency_ms": res.get("latency_ms", 0.0),
        "engine": "Fast Local ASL-Grammar NLP",
        "llm_available": False,
        "status": "fallback",
        "error": res.get("error"),
    }


def _make_continuous_predict_fn(predictor: Any, user_id: str, personalization_enabled: bool):
    def _predict_fn(seq32: np.ndarray, k: int) -> Dict[str, Any]:
        res = predictor.predict_sequence(seq32, is_normalized=False, top_k=k)
        return _apply_personalization_override(res, seq32, user_id, personalization_enabled)
    return _predict_fn


@app.post("/api/continuous/stream")
async def continuous_stream_endpoint(req: ContinuousStreamRequest):
    """Stream incremental webcam frames through the kinematic stroke segmenter and CSLR decoder."""
    try:
        sess_id = _safe_user_id(req.session_id)
        arr = np.asarray(req.frames, dtype=np.float32)
        if arr.size == 0:
            raise HTTPException(status_code=400, detail="frames must be non-empty")
        if arr.ndim == 2 and arr.shape == (42, 3):
            arr = arr[None, :, :]
        elif arr.ndim == 2 and arr.shape[1] == 126:
            arr = arr.reshape(arr.shape[0], 42, 3)
        if arr.ndim != 3 or arr.shape[1:] != (42, 3):
            raise HTTPException(status_code=400, detail=f"Expected (N, 42, 3) frames, got {arr.shape}")

        predictor = get_predictor(req.model_variant)
        predict_fn = _make_continuous_predict_fn(predictor, req.user_id, req.personalization_enabled)
        state = _continuous_manager.push_frames(
            session_id=sess_id,
            frames=arr,
            predict_fn=predict_fn,
            timestamps_ms=req.timestamps_ms,
            auto_finalize_on_pause=req.auto_finalize_on_pause,
            top_k=req.top_k,
        )
        if state.get("finalized_utterance") and req.auto_translate_on_pause:
            fin = state["finalized_utterance"]
            fin["translation"] = _translate_gloss_sequence(
                fin.get("glosses", []),
                hypotheses=fin.get("hypotheses"),
            )
        return JSONResponse(content=state)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Continuous stream error: {str(e)}")


@app.post("/api/continuous/finalize")
async def continuous_finalize_endpoint(req: ContinuousFinalizeRequest):
    """Flush pending active stroke and finalize the current continuous utterance."""
    try:
        sess_id = _safe_user_id(req.session_id)
        predictor = get_predictor(req.model_variant)
        predict_fn = _make_continuous_predict_fn(predictor, req.user_id, req.personalization_enabled)
        out = _continuous_manager.finalize(sess_id, predict_fn=predict_fn, clear_after=req.clear_after)
        if req.translate and out.get("glosses"):
            out["translation"] = _translate_gloss_sequence(
                out.get("glosses", []),
                hypotheses=out.get("hypotheses"),
            )
        return JSONResponse(content=out)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Continuous finalize error: {str(e)}")


@app.post("/api/continuous/reset")
async def continuous_reset_endpoint(req: ContinuousResetRequest):
    """Reset the continuous CSLR streaming buffer and active glosses."""
    sess_id = _safe_user_id(req.session_id)
    _continuous_manager.reset(sess_id)
    return JSONResponse(content={"session_id": sess_id, "status": "reset"})


# -------------------------------------------------------------------
# ASL Fingerspelling Recognition Endpoints (Google ASLFR)
# -------------------------------------------------------------------

_fingerspelling_engine = None


def get_fingerspelling_engine():
    global _fingerspelling_engine
    if _fingerspelling_engine is None:
        from signbridge.fingerspelling.engine import FingerspellingEngine
        _fingerspelling_engine = FingerspellingEngine()
    return _fingerspelling_engine


@app.post("/api/fingerspelling/predict")
async def predict_fingerspelling_endpoint(request: FingerspellingRequest):
    """
    Run continuous ASL fingerspelling recognition on streaming landmark sequence.
    Powered by Squeezeformer/LiteRT on Apple Silicon M2 Pro.
    """
    try:
        engine = get_fingerspelling_engine()
        result = engine.predict(
            sequence=request.sequence,
            min_char_prob=request.min_char_prob or 0.25,
            target_len=request.target_len or 64,
        )
        return JSONResponse(content=result)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Fingerspelling prediction failure: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Fingerspelling prediction error: {str(e)}")


@app.get("/api/fingerspelling/status")
async def fingerspelling_status_endpoint():
    """Return fingerspelling model runtime info and checkpoint status."""
    try:
        engine = get_fingerspelling_engine()
        return {
            "status": "ready",
            "model": engine.weights_path.name,
            "architecture": "ASL Alphabet Landmark MLP (63->128->26) + Isotropic BBox Norm",
            "device": _detect_hardware_device() or platform.machine(),
            "vocab_size": 26,
            "num_landmarks": 21,
            "num_features": 63,
            "weights_verified": engine.weights_path.exists(),
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
        }


@app.get("/api/personalization/profile")
async def personalization_profile(user_id: str = Query("default")):
    uid = _safe_user_id(user_id)
    return {
        "profile": _personalization_manager.get_profile(uid),
        "signs": _personalization_manager.list_personalized_signs(uid),
        "enabled": _personalization_manager.get_model_state(uid).get("enabled", False),
    }


@app.post("/api/personalization/enable")
async def personalization_enable(request: PersonalizationUserRequest):
    uid = _safe_user_id(request.user_id)
    state = _personalization_manager.get_model_state(uid)
    state["enabled"] = True
    state["last_updated"] = datetime.now(timezone.utc).isoformat()
    user_dir = _personalization_manager._user_dir(uid)
    user_dir.mkdir(parents=True, exist_ok=True)
    with open(_personalization_manager._model_state_path(uid), "w") as f:
        json.dump(state, f, indent=2)
    return {"user_id": uid, "enabled": True}


@app.post("/api/personalization/disable")
async def personalization_disable(request: PersonalizationUserRequest):
    uid = _safe_user_id(request.user_id)
    state = _personalization_manager.get_model_state(uid)
    state["enabled"] = False
    state["last_updated"] = datetime.now(timezone.utc).isoformat()
    user_dir = _personalization_manager._user_dir(uid)
    user_dir.mkdir(parents=True, exist_ok=True)
    with open(_personalization_manager._model_state_path(uid), "w") as f:
        json.dump(state, f, indent=2)
    return {"user_id": uid, "enabled": False}


@app.post("/api/personalization/train")
async def personalization_train(request: PersonalizationUserRequest):
    uid = _safe_user_id(request.user_id)
    return _personalization_manager.train_custom_model(uid)


@app.post("/api/personalization/sample")
async def personalization_sample(request: PersonalizationSampleRequest):
    uid = _safe_user_id(request.user_id)
    raw_gloss = re.sub(r"[\s-]+", "_", request.gloss.strip().upper())
    if not raw_gloss or not re.fullmatch(r"[A-Z0-9_]{1,40}", raw_gloss):
        raise HTTPException(
            status_code=400,
            detail="Custom sign name must be 1-40 letters, numbers, spaces, hyphens, or underscores.",
        )
    gloss = raw_gloss
    predictor = get_predictor("production_exp13")
    sequence_np = np.asarray(request.sequence, dtype=np.float32)
    if sequence_np.shape != (32, 42, 3):
        raise HTTPException(status_code=400, detail=f"Expected sequence shape (32, 42, 3), got {sequence_np.shape}.")
    quality = _personalization_manager.check_sample_quality(sequence_np)
    if not quality["valid"]:
        return JSONResponse(status_code=422, content={"accepted": False, "quality": quality})
    result = predictor.predict_sequence(sequence_np, is_normalized=False, top_k=5)
    emb = np.asarray(result.get("embedding"), dtype=np.float32)
    if emb.size == 0:
        emb = predictor.extract_embedding(sequence_np, is_normalized=False)
    saved = _personalization_manager.save_sample(
        gloss=gloss,
        sequence=sequence_np,
        embedding=emb,
        quality=quality,
        user_id=uid,
    )
    state = _personalization_manager.get_model_state(uid)
    state["enabled"] = True
    state["last_updated"] = datetime.now(timezone.utc).isoformat()
    with open(_personalization_manager._model_state_path(uid), "w") as f:
        json.dump(state, f, indent=2)
    return {
        "accepted": True,
        "gloss": gloss,
        "in_base_vocab": gloss in predictor.class_names,
        "sample": saved,
        "learned": saved["total_samples"] >= _personalization_manager.config.min_samples_required,
        "quality": quality,
    }


@app.delete("/api/personalization/sign/{gloss}")
async def personalization_forget(gloss: str, user_id: str = Query("default")):
    uid = _safe_user_id(user_id)
    return _personalization_manager.forget_sign(gloss.strip().upper(), uid)


@app.delete("/api/personalization")
async def personalization_reset(user_id: str = Query("default")):
    uid = _safe_user_id(user_id)
    remove_dir = uid != _personalization_manager.default_user
    return _personalization_manager.delete_all_personalization(uid, remove_user_dir=remove_dir)


# -------------------------------------------------------------------
# DEBUG ENDPOINT — proves exactly what the browser is sending
# Returns full diagnostic data about ONE received payload.
# Does NOT modify model or normalization. Remove in production.
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# DIAGNOSTIC ENDPOINTS FOR WEBCAM BOTTLENECKS
# -------------------------------------------------------------------

class DiagnosticSequenceRequest(BaseModel):
    sequence: List[List[List[float]]] = Field(..., description="Shape (T, 42, 3)")
    mode: str = Field(..., description="'sign_now' or 'continuous_live'")
    window_number: int = Field(..., description="Inference window counter")
    hand_stats: Dict[str, Any] = Field(default_factory=dict)
    model_variant: str = "production_exp7"


@app.post("/api/diagnostic/predict")
async def diagnostic_predict_endpoint(request: DiagnosticSequenceRequest):
    """
    Diagnostic prediction endpoint that logs raw model behavior.
    Returns comprehensive diagnostic data including:
    1. Raw model predictions (Top-5)
    2. Active frame statistics
    3. Filtering outcomes
    4. Timestamp for temporal analysis
    """
    import numpy as np
    from datetime import datetime

    try:
        predictor = get_predictor(request.model_variant)
        seq_np = np.array(request.sequence, dtype=np.float32)

        # Basic shape validation
        if seq_np.ndim != 3 or seq_np.shape[1:] != (42, 3):
            raise HTTPException(
                status_code=400,
                detail=f"Expected shape (T, 42, 3), got {seq_np.shape}"
            )

        T = seq_np.shape[0]

        # Run inference to get raw predictions
        result = predictor.predict_sequence(seq_np, is_normalized=False, top_k=5)
        diagnostics = build_prediction_diagnostics(
            seq_np,
            result,
            mode=request.mode,
            window_number=request.window_number,
            hand_stats=request.hand_stats,
            model_variant=request.model_variant,
        )
        top1_class = diagnostics["raw_predictions"]["top1"]["class"]
        top1_confidence = diagnostics["raw_predictions"]["top1"]["confidence"]
        total_active_frames = diagnostics["frame_stats"]["active_frames"]
        zero_frame_percentage = diagnostics["frame_stats"]["zero_frame_percentage"]
        passes_filter = diagnostics["filter_analysis"]["would_pass_filter"]

        # Log to console for immediate visibility
        print(f"\n{'='*60}")
        print(f"DIAGNOSTIC [{request.mode}] Window #{request.window_number}")
        print(f"Raw Top-1: {top1_class} @ {top1_confidence:.2f}%")
        print(f"Active frames: {total_active_frames}/{T} ({zero_frame_percentage:.1f}% zero)")
        print(f"Filter pass: {passes_filter} ({'C+M' if passes_filter else 'C' if passes_confidence else 'M' if passes_margin else 'none'})")
        print(f"{'='*60}\n")

        # Combine with original result
        combined_result = result.copy()
        combined_result["diagnostics"] = diagnostics

        return JSONResponse(content=combined_result)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Diagnostic prediction error: {str(e)}")


def build_prediction_diagnostics(
    seq_np: np.ndarray,
    result: Dict[str, Any],
    mode: Optional[str] = None,
    window_number: Optional[int] = None,
    hand_stats: Optional[Dict[str, Any]] = None,
    model_variant: str = "production_exp7",
) -> Dict[str, Any]:
    """Build raw model telemetry without applying frontend temporal filtering."""
    from datetime import datetime, timezone

    T = int(seq_np.shape[0])
    hand0_active = ~np.all(seq_np[:, :21, :] == 0, axis=(1, 2))
    hand1_active = ~np.all(seq_np[:, 21:, :] == 0, axis=(1, 2))
    active = hand0_active | hand1_active
    predictions = result.get("top_k", [])
    top1 = predictions[0] if predictions else {}
    top2 = predictions[1] if len(predictions) > 1 else {}
    top1_confidence = float(top1.get("confidence", 0.0))
    top2_confidence = float(top2.get("confidence", 0.0))
    margin = top1_confidence - top2_confidence
    confidence_threshold = 45.0
    margin_threshold = 4.0
    passes_confidence = top1_confidence >= confidence_threshold
    passes_margin = margin >= margin_threshold

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "window_number": window_number,
        "mode": mode,
        "model_variant": model_variant,
        "frame_stats": {
            "total_frames": T,
            "active_frames": int(np.sum(active)),
            "active_frame_percentage": round(float(np.mean(active) * 100) if T else 0.0, 1),
            "zero_frame_percentage": round(float(np.mean(~active) * 100) if T else 0.0, 1),
            "hand0_active_frames": int(np.sum(hand0_active)),
            "hand1_active_frames": int(np.sum(hand1_active)),
            "hand_stats": hand_stats or {},
        },
        "raw_predictions": {
            "top1": {
                "class": top1.get("gloss"),
                "confidence": round(top1_confidence, 4),
            },
            "top2": {
                "class": top2.get("gloss"),
                "confidence": round(top2_confidence, 4),
            },
            "margin": round(margin, 4),
            "all_predictions": predictions,
        },
        "filter_analysis": {
            "current_thresholds": {
                "confidence": confidence_threshold,
                "margin": margin_threshold,
            },
            "passes_confidence": passes_confidence,
            "passes_margin": passes_margin,
            "would_pass_filter": passes_confidence and passes_margin,
            "reason": (
                "both thresholds met"
                if passes_confidence and passes_margin
                else "confidence below threshold"
                if not passes_confidence
                else "margin below threshold"
            ),
        },
        "hand_stability": analyze_hand_stability(seq_np),
    }


def analyze_hand_stability(seq_np: np.ndarray) -> Dict[str, Any]:
    """Analyze hand ordering stability across frames."""
    T = seq_np.shape[0]
    hand_swaps = 0

    for t in range(T):
        # Check if both hands are present in this frame
        h0_present = not np.allclose(seq_np[t, :21], 0)
        h1_present = not np.allclose(seq_np[t, 21:], 0)

        if h0_present and h1_present:
            # Get wrist X coordinates
            h0_wrist_x = seq_np[t, 0, 0]
            h1_wrist_x = seq_np[t, 21, 0]

            # Check if ordering is consistent (left hand should have smaller X)
            if t > 0:
                # Compare with previous frame if both were present
                prev_h0_present = not np.allclose(seq_np[t-1, :21], 0)
                prev_h1_present = not np.allclose(seq_np[t-1, 21:], 0)

                if prev_h0_present and prev_h1_present:
                    prev_h0_wrist_x = seq_np[t-1, 0, 0]
                    prev_h1_wrist_x = seq_np[t-1, 21, 0]

                    # Check if hand identities swapped
                    h0_was_left = prev_h0_wrist_x < prev_h1_wrist_x
                    h0_is_left = h0_wrist_x < h1_wrist_x

                    if h0_was_left != h0_is_left:
                        hand_swaps += 1

    stability_score = 100 * (1 - hand_swaps / max(1, T))

    return {
        "total_frames_analyzed": T,
        "hand_swaps_detected": hand_swaps,
        "swap_percentage": round(100 * hand_swaps / max(1, T), 1),
        "stability_score": round(stability_score, 1),
        "interpretation": "excellent" if stability_score > 95 else
                         "good" if stability_score > 85 else
                         "poor" if stability_score > 70 else "unstable"
    }


@app.post("/api/debug/sequence")
async def debug_sequence_endpoint(request: Union[SequenceRequest, FlatSequenceRequest]):
    """
    Debug endpoint: inspect every detail of the received landmark payload.
    Returns shape, wrist coords, zero-fill stats, normalization preview.
    No inference side-effects — predictor is never called.
    """
    import numpy as np
    from signbridge.preprocessing.normalize import normalize_sequence

    try:
        seq = np.array(request.sequence, dtype=np.float32)

        # ── Basic shape info ────────────────────────────────────────────────
        shape_received = list(seq.shape)
        ndim = seq.ndim

        # Normalise to (T, 42, 3) for analysis
        if ndim == 2 and seq.shape[1] == 126:
            seq3d = seq.reshape(seq.shape[0], 42, 3)
            shape_note = f"Flat (T,126) reshaped to (T,42,3)"
        elif ndim == 3 and seq.shape[1:] == (42, 3):
            seq3d = seq
            shape_note = "Correct 3-D (T,42,3)"
        elif ndim == 3 and seq.shape[1:] == (21, 3):
            seq3d = None
            shape_note = "WRONG: only 21 landmarks (single hand slot) — expected 42"
        else:
            seq3d = None
            shape_note = f"UNEXPECTED shape: {seq.shape}"

        if seq3d is None:
            return JSONResponse(content={
                "error": "Unhandled shape",
                "shape_received": shape_received,
                "shape_note": shape_note,
            }, status_code=400)

        T = seq3d.shape[0]

        # ── Frame-level stats ───────────────────────────────────────────────
        # Count frames where hand0 is zero (missing)
        hand0_zero_frames = int(np.sum(np.all(seq3d[:, :21, :] == 0, axis=(1, 2))))
        hand1_zero_frames = int(np.sum(np.all(seq3d[:, 21:, :] == 0, axis=(1, 2))))
        frames_both_zero  = int(np.sum(
            np.all(seq3d[:, :21, :] == 0, axis=(1, 2)) &
            np.all(seq3d[:, 21:, :] == 0, axis=(1, 2))
        ))

        # ── First frame ─────────────────────────────────────────────────────
        f0 = seq3d[0]
        first_frame_hand0_wrist = f0[0].tolist()   # landmark 0, xyz
        first_frame_hand1_wrist = f0[21].tolist()  # landmark 21, xyz
        first_frame_hand0_all   = f0[:21].tolist()
        first_frame_hand1_all   = f0[21:].tolist()

        # ── Last frame ──────────────────────────────────────────────────────
        fL = seq3d[-1]
        last_frame_hand0_wrist = fL[0].tolist()
        last_frame_hand1_wrist = fL[21].tolist()

        # ── Hand ordering (first non-zero frame) ───────────────────────────
        ordering_note = "N/A"
        for fi in range(T):
            h0w = seq3d[fi, 0, 0]   # hand0 wrist X
            h1w = seq3d[fi, 21, 0]  # hand1 wrist X
            h0_present = not np.allclose(seq3d[fi, :21], 0)
            h1_present = not np.allclose(seq3d[fi, 21:], 0)
            if h0_present and h1_present:
                ordering_note = (
                    f"Frame {fi}: hand0.wristX={h0w:.4f}, hand1.wristX={h1w:.4f} "
                    f"→ {'✅ correct left<right' if h0w < h1w else '⚠️ hand0 is RIGHT of hand1'}"
                )
                break
            elif h0_present:
                ordering_note = f"Frame {fi}: only hand0 detected (wristX={h0w:.4f})"
                break
            elif h1_present:
                ordering_note = f"Frame {fi}: only hand1 detected (wristX={h1w:.4f})"
                break

        # ── Normalization preview (first non-zero frame only) ───────────────
        try:
            normed = normalize_sequence(seq3d)
            normed_shape = list(normed.shape)
            normed_f0_h0_wrist = normed[0, 0].tolist()   # should be [0,0,0] after wrist subtraction
            normed_f0_h0_mid   = normed[0, 9].tolist()   # middle MCP, scaled
            norm_ok = np.allclose(normed[0, 0], 0) if not np.allclose(seq3d[0, :21], 0) else True
            norm_note = "✅ hand0 wrist is [0,0,0] after normalization" if norm_ok else "⚠️ wrist not zeroed"
        except Exception as ne:
            normed_shape = None
            normed_f0_h0_wrist = None
            normed_f0_h0_mid = None
            norm_note = f"normalize_sequence() raised: {ne}"

        # ── Value ranges ────────────────────────────────────────────────────
        global_min = float(np.min(seq3d))
        global_max = float(np.max(seq3d))
        x_range = [float(np.min(seq3d[:,:,0])), float(np.max(seq3d[:,:,0]))]
        y_range = [float(np.min(seq3d[:,:,1])), float(np.max(seq3d[:,:,1]))]
        z_range = [float(np.min(seq3d[:,:,2])), float(np.max(seq3d[:,:,2]))]

        result = {
            # ── Shape ──────────────────────────────────────────────────────
            "shape_received":     shape_received,
            "shape_note":         shape_note,
            "num_frames":         T,
            "num_landmarks":      42,
            "num_coords":         3,
            "model_expects":      "[1, 32, 126]",

            # ── Frame zero-fill analysis ────────────────────────────────────
            "hand0_zero_frames":  hand0_zero_frames,
            "hand1_zero_frames":  hand1_zero_frames,
            "frames_both_zero":   frames_both_zero,
            "real_frames":        T - frames_both_zero,

            # ── Coordinate ranges ───────────────────────────────────────────
            "global_value_min":   round(global_min, 6),
            "global_value_max":   round(global_max, 6),
            "x_range":            [round(v, 6) for v in x_range],
            "y_range":            [round(v, 6) for v in y_range],
            "z_range":            [round(v, 6) for v in z_range],

            # ── Hand ordering ───────────────────────────────────────────────
            "hand_ordering_check": ordering_note,

            # ── First frame ─────────────────────────────────────────────────
            "first_frame": {
                "hand0_wrist_xyz":  [round(v, 6) for v in first_frame_hand0_wrist],
                "hand1_wrist_xyz":  [round(v, 6) for v in first_frame_hand1_wrist],
                "hand0_all_21_landmarks": [[round(v, 6) for v in lm] for lm in first_frame_hand0_all],
                "hand1_all_21_landmarks": [[round(v, 6) for v in lm] for lm in first_frame_hand1_all],
            },

            # ── Last frame ──────────────────────────────────────────────────
            "last_frame": {
                "hand0_wrist_xyz": [round(v, 6) for v in last_frame_hand0_wrist],
                "hand1_wrist_xyz": [round(v, 6) for v in last_frame_hand1_wrist],
            },

            # ── Normalization preview ────────────────────────────────────────
            "normalization": {
                "output_shape":          normed_shape,
                "hand0_wrist_after_norm": [round(v, 6) for v in normed_f0_h0_wrist] if normed_f0_h0_wrist else None,
                "hand0_middle_mcp_after_norm": [round(v, 6) for v in normed_f0_h0_mid] if normed_f0_h0_mid else None,
                "note":                  norm_note,
            },
        }

        # Also print to server stdout for terminal inspection
        import json as _json
        print("\n" + "="*60)
        print("DEBUG /api/debug/sequence received payload:")
        print(_json.dumps({
            "shape_received": shape_received,
            "shape_note": shape_note,
            "real_frames": T - frames_both_zero,
            "hand0_zero_frames": hand0_zero_frames,
            "hand1_zero_frames": hand1_zero_frames,
            "hand_ordering_check": ordering_note,
            "x_range": x_range,
            "y_range": y_range,
            "z_range": z_range,
        }, indent=2))
        print("="*60 + "\n")

        return JSONResponse(content=result)

    except Exception as e:
        import traceback
        return JSONResponse(content={"error": str(e), "traceback": traceback.format_exc()}, status_code=500)


@app.post("/api/predict/video")
async def predict_video_endpoint(
    file: UploadFile = File(...),
    top_k: int = Form(5),
    model_variant: str = Form("production_exp13"),
):
    """Run inference on uploaded video file."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in [".mp4", ".mov", ".webm", ".avi", ".m4v"]:
        raise HTTPException(status_code=400, detail="Unsupported video format")

    temp_dir = tempfile.mkdtemp()
    temp_video_path = Path(temp_dir) / f"upload_{file.filename}"

    try:
        with open(temp_video_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        predictor = get_predictor(model_variant)
        result = predictor.predict_video(temp_video_path, top_k=top_k)
        result["filename"] = file.filename
        return JSONResponse(content=result)

    except RuntimeError as e:
        raise HTTPException(status_code=501, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Video processing error: {str(e)}")

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


class NLPSmoothRequest(BaseModel):
    glosses: List[str] = Field(..., description="List of recognized ASL gloss tokens")
    hypotheses: Optional[List[List[Dict[str, Any]]]] = Field(
        default=None,
        description="Optional per-slot Top-K candidate distributions from continuous CSLR",
    )
    non_manual: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional non-manual markers (wh_question, yn_question, negative)",
    )


class TTSRequest(BaseModel):
    text: str = Field(..., description="English sentence to convert to speech")
    voice: Optional[str] = Field("af_sarah", description="Kokoro voice name")


@app.post("/api/nlp/smooth")
async def nlp_smooth_endpoint(req: NLPSmoothRequest):
    """
    NLP Gloss-to-English translation endpoint with Top-K hypothesis disambiguation
    and ASL grammar ordering.
    """
    raw_glosses = req.glosses
    if not raw_glosses:
        return JSONResponse(content={"english": "", "glosses": [], "latency_ms": 0.0, "engine": "None"})

    return JSONResponse(
        content=_translate_gloss_sequence(
            raw_glosses,
            hypotheses=req.hypotheses,
            non_manual=req.non_manual,
        )
    )


@app.post("/api/tts/speak")
async def tts_speak_endpoint(req: TTSRequest):
    """
    Kokoro ONNX TTS endpoint. Synthesizes input text to speech.
    """
    res = _tts_service.speak(req.text, req.voice)
    return JSONResponse(content=res)


@app.get("/api/tts/status")
async def tts_status_endpoint():
    """
    Returns current Kokoro TTS status.
    """
    return JSONResponse(content=_tts_service.get_status())


@app.post("/api/tts/stop")
async def tts_stop_endpoint():
    """
    Stops active playback and clears pending speech queue.
    """
    _tts_service.stop()
    return JSONResponse(content={"status": "stopped", "message": "Playback stopped and queue cleared"})


@app.get("/api/pipeline/status")
async def pipeline_status_endpoint():
    """
    Returns real status of Groq AI, Qwen, and Kokoro TTS services.
    """
    return JSONResponse(content={
        "groq": {
            "available": _groq_service.is_available(),
            "model": _groq_service.model_name,
            "api_key_configured": bool(_groq_service.api_key),
            "engine": "Groq Cloud AI"
        },
        "qwen": {
            "available": _qwen_service.is_available(),
            "model": _qwen_service.model_name,
            "device": _qwen_service.device,
            "engine": "Direct PyTorch/Transformers"
        },
        "tts": _tts_service.get_status()
    })

