import os
import random
import shutil
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

import pandas as pd
from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from signbridge.inference.predictor import SignBridgePredictor

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CHECKPOINT = PROJECT_ROOT / "trained_models" / "best_gru_normalized.pth"
VIDEOS_DIR = PROJECT_ROOT / "dataset" / "ASL_Citizen" / "videos"
SPLITS_DIR = PROJECT_ROOT / "dataset" / "ASL_Citizen" / "splits"
STATIC_DIR = Path(__file__).resolve().parent / "static"
TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"

app = FastAPI(
    title="SignBridge API & Application",
    description="Sign Language Recognition & Learning System API",
    version="2.0.0",
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
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

_predictor: Optional[SignBridgePredictor] = None
_gloss_to_videos: Dict[str, List[str]] = {}
_demo_ready_glosses: set = set()
_elite_glosses: set = set()


def get_predictor() -> SignBridgePredictor:
    global _predictor
    if _predictor is None:
        if not DEFAULT_CHECKPOINT.exists():
            raise FileNotFoundError(f"Model checkpoint not found at {DEFAULT_CHECKPOINT}")
        _predictor = SignBridgePredictor(checkpoint_path=DEFAULT_CHECKPOINT)
    return _predictor


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


class FlatSequenceRequest(BaseModel):
    sequence: List[List[float]] = Field(..., description="Shape (T, 126)")
    top_k: Optional[int] = Field(default=5, ge=1, le=20)


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
    predictor = get_predictor()
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
    """Run inference on live camera landmark sequence."""
    try:
        predictor = get_predictor()
        import numpy as np

        sequence_np = np.array(request.sequence, dtype=np.float32)
        top_k = request.top_k or 5

        if sequence_np.ndim not in (2, 3):
            raise HTTPException(status_code=400, detail=f"Invalid sequence array dimensions: {sequence_np.ndim}")

        result = predictor.predict_sequence(sequence_np, is_normalized=False, top_k=top_k)
        return JSONResponse(content=result)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sequence prediction error: {str(e)}")


# -------------------------------------------------------------------
# DEBUG ENDPOINT — proves exactly what the browser is sending
# Returns full diagnostic data about ONE received payload.
# Does NOT modify model or normalization. Remove in production.
# -------------------------------------------------------------------

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


@app.post("/api/nlp/smooth")
async def nlp_smooth_endpoint(req: NLPSmoothRequest):
    """
    NLP Gloss-to-English translation endpoint.
    Converts raw ASL gloss sequence into natural, fluent English.
    """
    raw_glosses = req.glosses
    if not raw_glosses:
        return JSONResponse(content={"english": "", "glosses": []})

    cleaned = [g.strip().upper().rstrip("0123456789").removesuffix("/IT") for g in raw_glosses if g.strip()]

    # Quick rule matching
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
        return JSONResponse(content={"english": idioms[key], "glosses": cleaned, "idiom": True})

    # Basic grammatical assembly
    pronouns = {"ME": "I", "I": "I", "MY": "my", "YOU": "you", "YOUR": "your", "WE": "we", "THEY": "they"}
    tokens = [pronouns.get(w, w.lower()) for w in cleaned]

    if len(tokens) >= 2 and tokens[0] in ["I", "you", "we", "they"]:
        subj = tokens[0]
        rest = tokens[1:]
        if rest[0] in ["want", "like", "need", "eat", "have"]:
            verb = rest[0]
            obj = " ".join(rest[1:])
            article = "an" if obj and obj[0] in "aeiou" else "a"
            if verb == "want":
                english = f"{subj} would like {article} {obj}." if obj else f"{subj} want that."
            elif verb == "eat":
                english = f"{subj} would like to eat {article} {obj}." if obj else f"{subj} am eating."
            else:
                english = f"{subj} {verb} {obj}." if obj else f"{subj} {verb} it."
        else:
            english = " ".join(tokens).capitalize() + "."
    else:
        english = " ".join(tokens).capitalize() + "."

    return JSONResponse(content={"english": english, "glosses": cleaned, "idiom": False})
