import os
import random
import shutil
import tempfile
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd
import re
from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from signbridge.inference.predictor import SignBridgePredictor
from signbridge.personalization.manager import PersonalizationManager
from signbridge.personalization.router import PersonalizationRouter
from signbridge.language.qwen_service import QwenLanguageService
from signbridge.language.groq_service import GroqLanguageService
from signbridge.tts.service import KokoroTTSService

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PRODUCTION_CHECKPOINT = PROJECT_ROOT / "trained_models" / "production" / "best_model.pth"
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

_predictors: Dict[str, SignBridgePredictor] = {}
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


MODEL_CHECKPOINTS = {
    "production_exp7": PRODUCTION_CHECKPOINT,
    "candidate_exp9": EXP9_CHECKPOINT,
}


def get_predictor(model_variant: str = "production_exp7") -> SignBridgePredictor:
    if model_variant not in MODEL_CHECKPOINTS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model variant '{model_variant}'. Choose from {sorted(MODEL_CHECKPOINTS)}.",
        )
    checkpoint = MODEL_CHECKPOINTS[model_variant]
    if not checkpoint.exists():
        raise FileNotFoundError(f"Model checkpoint not found at {checkpoint}")
    if model_variant not in _predictors:
        _predictors[model_variant] = SignBridgePredictor(checkpoint_path=checkpoint)
    return _predictors[model_variant]


def init_dataset_mappings():
    global _gloss_to_videos, _demo_ready_glosses, _elite_glosses
    if _gloss_to_videos:
        return

    dfs = []
    for split in ["train.csv", "val.csv", "test.csv"]:
        csv_path = SPLITS_DIR / split
        if csv_path.exists():
            dfs.append(pd.read_csv(csv_path))

    if dfs:
        combined_df = pd.concat(dfs, ignore_index=True)
        for gloss, group in combined_df.groupby("Gloss"):
            _gloss_to_videos[str(gloss).upper()] = group["Video file"].tolist()

    demo_csv = PROJECT_ROOT / "signbridge_final_demo_glosses.csv"
    if demo_csv.exists():
        df_demo = pd.read_csv(demo_csv)
        _demo_ready_glosses = set(df_demo["Gloss"].str.upper().tolist())

    elite_csv = PROJECT_ROOT / "signbridge_elite_glosses.csv"
    if elite_csv.exists():
        df_elite = pd.read_csv(elite_csv)
        _elite_glosses = set(df_elite["Gloss"].str.upper().tolist())


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


class PersonalizationSampleRequest(BaseModel):
    sequence: List[List[List[float]]] = Field(..., description="Shape (32, 42, 3)")
    gloss: str = Field(..., min_length=1, max_length=100)
    user_id: str = Field(default="default", min_length=1, max_length=64)
    model_variant: str = "candidate_exp9"


class PersonalizationUserRequest(BaseModel):
    user_id: str = Field(default="default", min_length=1, max_length=64)


def _safe_user_id(user_id: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", user_id):
        raise HTTPException(status_code=400, detail="user_id may contain only letters, numbers, '_' and '-'.")
    return user_id


def _personalized_result(result: Dict[str, Any], user_id: str, enabled: bool) -> Dict[str, Any]:
    uid = _safe_user_id(user_id)
    if not enabled:
        return _personalization_router._no_personalization(result.get("top_k", []))
    prototypes = _personalization_manager.get_all_prototypes(uid)
    if not prototypes:
        return _personalization_router._no_personalization(result.get("top_k", []))
    return _personalization_router.route_prediction(
        result.get("top_k", []),
        np.asarray(result.get("embedding"), dtype=np.float32),
        prototypes,
        user_id=uid,
    )


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


@app.get("/api/health")
async def health_check():
    predictor = get_predictor()
    init_dataset_mappings()
    return {
        "status": "healthy",
        "device": str(predictor.device),
        "num_classes": len(predictor.class_names),
        "total_videos_mapped": len(_gloss_to_videos),
        "model_variants": {
            name: path.exists() for name, path in MODEL_CHECKPOINTS.items()
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


@app.get("/api/vocabulary")
async def get_vocabulary(
    q: Optional[str] = Query(None, description="Search query string"),
    limit: int = Query(60, ge=1, le=2731),
    category: Optional[str] = Query(None, description="Filter by category (elite, demo, all)"),
):
    """Browse or search supported sign vocabulary."""
    predictor = get_predictor("production_exp7")
    init_dataset_mappings()

    glosses = predictor.class_names
    query = (q or "").strip().upper()

    if query:
        glosses = [g for g in glosses if query in g]

    if category == "elite" and _elite_glosses:
        glosses = [g for g in glosses if g in _elite_glosses]
    elif category == "demo" and _demo_ready_glosses:
        glosses = [g for g in glosses if g in _demo_ready_glosses]

    glosses_slice = glosses[:limit]

    items = []
    for g in glosses_slice:
        video_count = len(_gloss_to_videos.get(g, []))
        is_elite = g in _elite_glosses
        is_demo = g in _demo_ready_glosses
        tier = "High Precision" if is_elite else ("Verified Demo" if is_demo else "Standard")

        items.append({
            "gloss": g,
            "video_count": video_count,
            "has_demonstration": video_count > 0,
            "tier": tier,
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
    video_path = VIDEOS_DIR / filename
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video asset not found")
    return FileResponse(path=video_path, media_type="video/mp4", filename=filename)


@app.get("/api/teach-me")
async def teach_me_something():
    """🎲 Randomly pick a high-quality sign for the user to learn."""
    init_dataset_mappings()
    predictor = get_predictor()

    pool = list(_elite_glosses or _demo_ready_glosses or predictor.class_names)
    selected_gloss = random.choice(pool)

    return await get_demonstration(selected_gloss)


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
        result["personalization"] = _personalized_result(
            result, request.user_id, request.personalization_enabled
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


@app.post("/api/personalization/sample")
async def personalization_sample(request: PersonalizationSampleRequest):
    uid = _safe_user_id(request.user_id)
    gloss = request.gloss.strip().upper()
    predictor = get_predictor("candidate_exp9")
    sequence_np = np.asarray(request.sequence, dtype=np.float32)
    if sequence_np.shape != (32, 42, 3):
        raise HTTPException(status_code=400, detail=f"Expected sequence shape (32, 42, 3), got {sequence_np.shape}.")
    if gloss not in predictor.class_names:
        raise HTTPException(status_code=400, detail=f"Gloss '{gloss}' is not in the 2,731-class vocabulary.")
    quality = _personalization_manager.check_sample_quality(sequence_np)
    if not quality["valid"]:
        return JSONResponse(status_code=422, content={"accepted": False, "quality": quality})
    result = predictor.predict_sequence(sequence_np, is_normalized=False, top_k=5)
    saved = _personalization_manager.save_sample(
        gloss=gloss,
        sequence=sequence_np,
        embedding=np.asarray(result["embedding"], dtype=np.float32),
        quality=quality,
        user_id=uid,
    )
    state = _personalization_manager.get_model_state(uid)
    state["enabled"] = True
    with open(_personalization_manager._model_state_path(uid), "w") as f:
        json.dump(state, f, indent=2)
    return {
        "accepted": True,
        "gloss": gloss,
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
    return _personalization_manager.delete_all_personalization(uid)


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

        predictor = get_predictor()
        result = predictor.predict_video(temp_video_path, top_k=top_k)
        result["filename"] = file.filename
        return JSONResponse(content=result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Video processing error: {str(e)}")

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


class NLPSmoothRequest(BaseModel):
    glosses: List[str] = Field(..., description="List of recognized ASL gloss tokens")


class TTSRequest(BaseModel):
    text: str = Field(..., description="English sentence to convert to speech")
    voice: Optional[str] = Field("af_sarah", description="Kokoro voice name")


@app.post("/api/nlp/smooth")
async def nlp_smooth_endpoint(req: NLPSmoothRequest):
    """
    NLP Gloss-to-English translation endpoint using local PyTorch Qwen Instruct.
    Converts raw ASL gloss sequence into natural, fluent English.
    """
    raw_glosses = req.glosses
    if not raw_glosses:
        return JSONResponse(content={"english": "", "glosses": [], "latency_ms": 0.0, "engine": "None"})

    cleaned = [g.strip().upper().rstrip("0123456789").removesuffix("/IT") for g in raw_glosses if g.strip()]
    if not cleaned:
        return JSONResponse(content={"english": "", "glosses": [], "latency_ms": 0.0, "engine": "None"})

    # Primary: Groq AI Language Service (or local NLP fallback)
    res = _groq_service.translate_glosses(cleaned)
    if not res.get("english"):
        res = _qwen_service.translate_glosses(cleaned)

    if res.get("status") == "success" and res.get("english"):
        return JSONResponse(content={
            "english": res["english"],
            "glosses": cleaned,
            "latency_ms": res["latency_ms"],
            "engine": res.get("engine", "Groq AI"),
            "llm_available": _groq_service.is_available() or _qwen_service.is_available(),
            "status": "success",
            "error": None
        })

    # Fallback to rule assembly only if Qwen is unreachable / error
    logger.warning(f"Qwen translation failed: {qwen_res.get('error')}. Using rule fallback.")
    idioms = {
        ("NICE", "MEET", "YOU"): "Nice to meet you!",
        ("SEE", "YOU", "LATER"): "See you later!",
        ("GOOD", "MORNING"): "Good morning!",
        ("GOOD", "NIGHT"): "Good night!",
        ("HOW", "YOU"): "How are you?",
        ("YOU", "HOW"): "How are you?",
        ("NAME", "YOU", "WHAT"): "What is your name?",
        ("YOU", "NAME", "WHAT"): "What is your name?",
        ("TIME", "WHAT"): "What time is it?",
        ("WHERE", "BATHROOM"): "Where is the restroom?",
        ("BATHROOM", "WHERE"): "Where is the bathroom?",
        ("HELP", "ME"): "Can you please help me?",
        ("THANKYOU",): "Thank you very much!",
        ("SORRY",): "I am sorry.",
        ("PLEASE",): "Please.",
        ("HELLO",): "Hello!",
        ("YES",): "Yes, absolutely.",
        ("NO",): "No, thank you."
    }

    key = tuple(cleaned)
    if key in idioms:
        english = idioms[key]
    else:
        pronouns = {"ME": "I", "I": "I", "MY": "my", "YOU": "you", "YOUR": "your", "WE": "we", "THEY": "they"}
        tokens = [pronouns.get(w, w.lower()) for w in cleaned]
        english = " ".join(tokens).capitalize() + "."

    return JSONResponse(content={
        "english": english,
        "glosses": cleaned,
        "latency_ms": qwen_res.get("latency_ms", 0.0),
        "engine": "Rule Fallback",
        "llm_available": False,
        "status": qwen_res.get("status", "fallback"),
        "error": qwen_res.get("error")
    })


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

