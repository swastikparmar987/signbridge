# SignBridge 🤟

Real-time American Sign Language (ASL) recognition web app built with PyTorch, MediaPipe, and FastAPI.

## Features

- **Live webcam recognition** — MediaPipe Hands → 32-frame buffer → GRU inference → prediction in real time
- **Video file upload** — drag-and-drop a `.mp4` and get an instant sign prediction
- **2,731 ASL glosses** — trained on WLASL dataset
- **Learn mode** — browse vocabulary, watch demonstrations, practice signs with live feedback
- **Debug panel** — real-time pipeline status (Camera / MediaPipe / Buffer / API / Prediction)

## Architecture

```
Browser webcam
    ↓ MediaPipe Hands (42 landmarks / frame)
    ↓ Rolling 32-frame buffer (real frames only)
    ↓ POST /api/predict/sequence  [32, 42, 3]
FastAPI + PyTorch GRU
    ↓ normalize_sequence()
    ↓ reshape → [1, 32, 126]
    ↓ GRU (2,731-class output)
    → Prediction + confidence
```

## Model

| Property | Value |
|----------|-------|
| Architecture | GRU (Gated Recurrent Unit) |
| Input shape | `[1, 32, 126]` (32 frames × 42 landmarks × 3 coords) |
| Output classes | 2,731 ASL glosses |
| Checkpoint | `trained_models/best_gru_normalized.pth` |
| Top-1 accuracy | 34.84% |
| Top-5 accuracy | 61.54% |

## Setup

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/signbridge.git
cd signbridge

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the server
uvicorn signbridge.web.app:app --host 0.0.0.0 --port 8000

# 5. Open browser
open http://localhost:8000
```

## Requirements

- Python 3.10+
- PyTorch (MPS / CUDA / CPU)
- FastAPI + Uvicorn
- OpenCV
- MediaPipe

See `requirements.txt` for full list.

## Project Structure

```
signbridge/
├── inference/          # Standalone CLI prediction scripts
├── preprocessing/      # Landmark extraction & normalization
│   ├── landmarks.py    # MediaPipe hand landmark extraction
│   └── normalize.py    # normalize_sequence() used by GRU
├── training/           # Model definition (GRU)
└── web/                # FastAPI app + frontend
    ├── app.py          # REST API endpoints
    ├── static/
    │   ├── js/app.js   # Webcam pipeline + debug panel
    │   └── css/        # Styles
    └── templates/      # Jinja2 HTML templates

trained_models/
└── best_gru_normalized.pth   # Trained GRU checkpoint (16MB)
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/health` | Model health + class count |
| POST | `/api/predict/sequence` | Live webcam inference `[32,42,3]` |
| POST | `/api/predict/video` | Video file inference |
| POST | `/api/debug/sequence` | Payload diagnostic (shape, wrist coords, normalization) |
| GET | `/api/vocabulary` | Browse 2,731 glosses |
| GET | `/api/demonstration/{gloss}` | Get demo video for a gloss |

## Dataset

Trained on [WLASL (World Level American Sign Language)](https://dxli94.github.io/WLASL/) dataset.
The dataset is not included in this repository due to its size (~46GB).

## License

MIT
