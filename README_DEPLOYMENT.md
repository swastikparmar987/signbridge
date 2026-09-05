# SignBridge: Production Deployment & Application Guide

SignBridge is a human-centered, high-performance sign language recognition and communication interface built on the ASL Citizen dataset (2,731 gloss classes).

---

## 🚀 Quick Start

### 1. Environment Setup

Ensure Python 3.11+ is installed. Activate your virtual environment and install dependencies:

```bash
cd /Users/pro/Desktop/projects/signbridge
source .venv/bin/activate
uv pip install --python .venv -r requirements.txt
```

### 2. Start the SignBridge Web Application

Run the production web server:

```bash
PYTHONPATH=. .venv/bin/python -m uvicorn signbridge.web.app:app --host 0.0.0.0 --port 8000
```

Open your browser at: `http://localhost:8000`

---

## 🧠 Model Architecture & Frozen ML Pipeline

- **Checkpoint Path**: `trained_models/best_gru_normalized.pth`
- **Architecture**: `GRUClassifier`
  - Input Features: `126` (2 hands × 21 landmarks × 3 coordinates x, y, z)
  - Hidden Size: `256`
  - GRU Layers: `2`
  - Number of Classes: `2,731`
- **Evaluation Accuracy** (32,941 ASL Citizen Test Samples):
  - **Top-1 Accuracy**: **34.84%**
  - **Top-5 Accuracy**: **61.54%**

---

## 📦 Package Architecture

```
signbridge/
├── __init__.py
├── preprocessing/
│   ├── __init__.py
│   ├── normalize.py       # Single source of truth for wrist & scale normalization
│   └── landmarks.py       # OpenCV & MediaPipe landmark extraction (32 frames)
├── inference/
│   ├── __init__.py
│   ├── model.py           # GRUClassifier & checkpoint loader
│   ├── predictor.py       # High-level SignBridgePredictor API
│   ├── predict_cache.py   # CLI tool for raw .npy landmark cache files
│   ├── predict_video.py   # CLI tool for .mp4 video files
│   └── verify_pipeline.py # End-to-end forensic verification suite
└── web/
    ├── __init__.py
    ├── app.py             # FastAPI backend server & REST API
    ├── static/
    │   ├── css/
    │   │   └── style.css  # Accessible, warm, human-centered UI design system
    │   └── js/
    │       └── app.js     # Live WebRTC camera, MediaPipe overlay, temporal debouncing
    └── templates/
        └── index.html     # Single Page Application HTML layout
```

---

## 🌐 Web Application Features

1. **🎥 Live Camera Mode (Primary)**:
   - Live WebRTC camera feed with MediaPipe Hands overlay in browser.
   - Live 32-frame rolling buffer sent to `/api/predict/sequence`.
   - UI temporal smoothing & debouncing: prevents UI flickering and stabilizes predictions.
   - Human-centered status guidance: *"Looking for a sign..."*, *"Hand detected — Analyzing sign..."*.
2. **↑ Video Upload Mode (Secondary)**:
   - Drag-and-drop zone for `.mp4`, `.mov`, or `.webm` files.
   - Automatic video processing via `/api/predict/video`.
3. **Hero Prediction Presentation**:
   - Prominent, elegant display of predicted gloss with confidence badge (*"High confidence"*, *"Not completely sure"*).
   - Ranked Top-5 prediction list with progress indicators.
4. **Accessibility (WCAG 2.2 AA)**:
   - Accessible contrast ratio, keyboard navigation (`Tab` / `Space`), visible focus rings, ARIA live region (`aria-live="polite"`), and `prefers-reduced-motion` support.

---

## 🧪 Automated Pipeline Verification

To verify full model parity, golden sample tests, class mapping alignment, and test set accuracy reproduction:

```bash
PYTHONPATH=. .venv/bin/python -m signbridge.inference.verify_pipeline
```

---

## 💻 REST API Endpoints

- `GET /api/health` — Check server status, active device (`mps`/`cuda`/`cpu`), and class count.
- `POST /api/predict/sequence` — Predict on array of 3D landmarks `(T, 42, 3)` or `(T, 126)`.
- `POST /api/predict/video` — Predict on uploaded video file (`.mp4`, `.mov`, `.webm`).
- `POST /api/predict/cache` — Predict on cached `.npy` sequence file by dataset index.
