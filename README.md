# SignBridge 🤟

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Real-time American Sign Language (ASL) recognition and learning web application powered by **MediaPipe Hands**, a **PyTorch GRU Neural Network**, and **FastAPI**. Trained on the comprehensive ASL Citizen / WLASL dataset supporting **2,731 ASL gloss classes**.

---

## ⚡ Quick Start: How to Run

Follow these step-by-step instructions to run SignBridge locally on your machine.

### 1. Prerequisites

- **Python 3.10 or higher** installed (`python3 --version`)
- A working **Webcam** (for live sign recognition)
- Modern web browser (Chrome, Edge, Firefox, or Safari)

---

### 2. Clone the Repository

```bash
git clone https://github.com/swastikparmar987/signbridge.git
cd signbridge
```

---

### 3. Create and Activate a Virtual Environment

#### On macOS / Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

#### On Windows (Command Prompt / PowerShell):
```cmd
python -m venv .venv
.venv\Scripts\activate
```

---

### 4. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **Note for Apple Silicon (M1/M2/M3/M4 Macs):** PyTorch will automatically utilize the high-speed Metal Performance Shaders (`mps`) hardware acceleration.
> **Note for NVIDIA GPUs:** PyTorch will automatically detect and leverage CUDA.

---

### 5. Launch the Web Application

Start the FastAPI server using Uvicorn:

```bash
python -m uvicorn signbridge.web.app:app --host 0.0.0.0 --port 8000 --reload
```

---

### 6. Open the Application in Your Browser

Open your browser and navigate to:

👉 **[http://localhost:8000](http://localhost:8000)**

---

## 📖 How to Use the Application

### 🎥 1. Live Webcam Recognition Mode (Default)
1. Click the **"Start Camera"** button on the web interface.
2. Allow camera access in your browser when prompted.
3. Position your hands clearly in front of the camera. MediaPipe will track 21 landmarks per hand with a visual skeleton overlay.
4. Perform an ASL sign. Once the 32-frame buffer fills with active signing motion, the system sends the sequence to the GRU model and displays:
   - **Top-1 Predicted Gloss** (with confidence percentage)
   - **Top-5 Ranked Alternatives** with probability distribution bars
5. **🛠 Diagnostic Debug Panel**:
   - Press the **`D`** key or click the **🛠 Debug** toggle in the camera header to view real-time pipeline telemetry (Camera status, MediaPipe state, Hands detected, Buffer fill count `XX/32`, API latency, and payload shapes).

### 📁 2. Video Upload Mode
1. Click on the **"Upload Video"** tab.
2. Drag and drop any `.mp4`, `.mov`, or `.webm` video recording of an ASL sign (or click to browse).
3. The server automatically extracts MediaPipe landmark sequences from the video frames, runs GRU inference, and renders the prediction.

### 📚 3. Learn & Practice Mode
1. Click **"Learn"** in the top navigation bar.
2. Search through **2,731 ASL vocabulary glosses**.
3. Select a sign to watch a demonstration video.
4. Click **"Try Signing This"** to enter Practice Mode — the app guides you to perform the sign in front of the webcam and checks your accuracy against the target!

---

## 🧪 CLI & Pipeline Verification

You can also run offline inference and automated verification tests directly from the terminal:

### Test and Verify the ML Pipeline
Verifies dataset alignment, GRU weights, and pipeline accuracy:
```bash
python -m signbridge.inference.verify_pipeline
```

### Run Prediction on a Local Video File
```bash
python -m signbridge.inference.predict_video --video path/to/sample.mp4
```

---

## 🧠 System Architecture

```
┌─────────────────┐       ┌────────────────────────┐       ┌─────────────────────────┐
│ Browser Webcam  │ ────> │ MediaPipe Hands (JS)   │ ────> │ 32-Frame Buffer [32,42,3]│
└─────────────────┘       └────────────────────────┘       └─────────────────────────┘
                                                                        │
                                                              POST /api/predict/sequence
                                                                        ▼
┌─────────────────────────┐       ┌────────────────────────┐       ┌─────────────────────────┐
│ Real-Time Web UI        │ <──── │ 2,731-Class Prediction │ <──── │ PyTorch GRU Classifier  │
│ Top-1 & Top-5 Rankings  │       │ + Confidence Score     │       │ [1, 32, 126] Tensor     │
└─────────────────────────┘       └────────────────────────┘       └─────────────────────────┘
```

### Model Specifications
- **Architecture**: Deep Gated Recurrent Unit (`GRUClassifier`)
- **Input Dimensions**: `[Batch, 32 frames, 126 features]` (2 hands × 21 landmarks × 3 coordinates `[x, y, z]`)
- **Hidden Dimensions**: `256`, 2 GRU layers with dropout
- **Classes**: `2,731` distinct ASL signs
- **Weights Included**: Pre-trained weights stored in [`trained_models/best_gru_normalized.pth`](trained_models/best_gru_normalized.pth) (16 MB)
- **Top-1 Accuracy**: **34.84%**
- **Top-5 Accuracy**: **61.54%**

---

## 🌐 REST API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Service health status, hardware accelerator (`mps`/`cuda`/`cpu`), and total classes |
| `POST` | `/api/predict/sequence` | Real-time prediction on a `(32, 42, 3)` sequence of coordinates |
| `POST` | `/api/predict/video` | Upload a video file (`.mp4`, `.mov`, `.webm`) for inference |
| `POST` | `/api/debug/sequence` | Diagnostic endpoint returning shape verification, wrist coordinates, and normalization details |
| `GET` | `/api/vocabulary` | Search and list vocabulary items with pagination |
| `GET` | `/api/demonstration/{gloss}` | Fetch demonstration video info for a specific ASL sign |
| `GET` | `/api/teach-me` | Returns a random sign for interactive learning |

---

## 🔧 Troubleshooting

- **Camera Not Starting**:
  Ensure your browser has permission to access your webcam. On macOS, make sure your browser has Camera permissions enabled under **System Settings > Privacy & Security > Camera**.
- **Port 8000 Already in Use**:
  Run on a different port:
  ```bash
  python -m uvicorn signbridge.web.app:app --host 0.0.0.0 --port 8080
  ```
- **ModuleNotFoundError**:
  Ensure your virtual environment is activated (`source .venv/bin/activate` or `.venv\Scripts\activate`) and run with `python -m uvicorn ...`.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
