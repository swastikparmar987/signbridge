<div align="center">

# 🤟 SignBridge

### *Bridging Silence and Sound with Real-Time AI*

**An ultra-low latency, real-time American Sign Language (ASL) recognition & learning platform.**  
Powered by **MediaPipe Hands**, a custom **PyTorch GRU Neural Network**, and a high-throughput **FastAPI** backend.

---

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Hands-007ACC?style=for-the-badge&logo=google&logoColor=white)](https://mediapipe.dev/)
[![License](https://img.shields.io/badge/License-MIT-F5A623?style=for-the-badge)](LICENSE)

[**Explore Features**](#-core-features) • [**Quick Start**](#-quick-start) • [**Architecture**](#-neural-architecture) • [**API Docs**](#-rest-api-reference) • [**Learn Mode**](#-interactive-learning--practice)

</div>

---

## 🌟 Highlights

- ⚡ **Real-Time Webcam Inference**: Continuous live video recognition with a 32-frame dynamic rolling buffer (~40ms inference on Apple Silicon / CUDA).
- 🎯 **Massive Vocabulary**: Recognizes **2,731 distinct ASL glosses** trained on the comprehensive ASL Citizen dataset.
- 🖐️ **Sub-Pixel Landmark Normalization**: Dual-hand wrist-centered origin translation and scale invariance for camera distance independence.
- 🎓 **Interactive Learning Suite**: Instant demonstration videos for every vocabulary sign with guided live practice and accuracy scoring.
- 🛠️ **Built-In Diagnostic Overlay**: Live telemetry HUD displaying camera state, hand landmark tracking, buffer health, and API latency.
- ♿ **Accessible by Design**: WCAG 2.2 AA compliant, dark-mode optimized interface with full keyboard navigation.

---

## ⚡ Quick Start

Get SignBridge up and running in **less than 2 minutes**.

### 1. Clone the Repository

```bash
git clone https://github.com/swastikparmar987/signbridge.git
cd signbridge
```

### 2. Set Up Virtual Environment

<details open>
<summary><b>macOS & Linux</b></summary>

```bash
python3 -m venv .venv
source .venv/bin/activate
```
</details>

<details>
<summary><b>Windows (Command Prompt / PowerShell)</b></summary>

```cmd
python -m venv .venv
.venv\Scripts\activate
```
</details>

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> 🚀 **Hardware Acceleration Note**:
> - **macOS (Apple Silicon M1/M2/M3/M4)**: Automatically runs on **Metal (`mps`)**.
> - **Linux / Windows with NVIDIA GPU**: Automatically runs on **CUDA**.
> - **CPU Fallback**: Gracefully executes vectorized CPU tensor operations.

### 4. Launch the Server

```bash
python -m uvicorn signbridge.web.app:app --host 0.0.0.0 --port 8000 --reload
```

### 5. Open in Browser

👉 Navigate to: **[http://localhost:8000](http://localhost:8000)**

---

## 🎮 How to Use

<table>
  <tr>
    <td width="33%" align="center">
      <h3>🎥 Live Recognition</h3>
      <p>Click <b>Start Camera</b>. Hold your hands up, sign naturally, and get immediate Top-1 and Top-5 ranked predictions.</p>
    </td>
    <td width="33%" align="center">
      <h3>📁 Video File Upload</h3>
      <p>Drag and drop any <code>.mp4</code>, <code>.mov</code>, or <code>.webm</code> video file to instantly analyze offline recordings.</p>
    </td>
    <td width="33%" align="center">
      <h3>📚 Learn & Practice</h3>
      <p>Browse 2,731 sign definitions, watch demo clips, and practice in real time with interactive score feedback.</p>
    </td>
  </tr>
</table>

> 💡 **Pro-Tip**: Press the **`D`** key at any time while the camera is active to toggle the **Real-Time Diagnostic Panel** to inspect exact tensor dimensions and latency.

---

## 🧠 Neural Architecture

The recognition pipeline is built for high accuracy and ultra-low temporal latency:

```mermaid
flowchart LR
    A[Webcam Feed] --> B[MediaPipe Hands]
    B -->|42 Landmarks x 3D| C[Rolling 32-Frame Buffer]
    C -->|POST Payload [32, 42, 3]| D[FastAPI Backend]
    D --> E[Wrist & Scale Normalization]
    E --> F[PyTorch GRU Tensor [1, 32, 126]]
    F --> G[Linear Classifier]
    G --> H[Softmax Top-5 Ranked Glosses]
```

### Model Specifications

| Parameter | Specification | Description |
|---|---|---|
| **Model Type** | Gated Recurrent Unit (`GRUClassifier`) | Optimized for sequential temporal gesture tracking |
| **Input Shape** | `[Batch, 32, 126]` | 32 Frames × (2 Hands × 21 Landmarks × 3 Coordinates) |
| **Hidden Units** | `256` units | 2 Bidirectional-aware recurrent layers with dropout |
| **Vocabulary Size** | `2,731` Classes | Full ASL Citizen gesture spectrum |
| **Checkpoint** | `trained_models/best_gru_normalized.pth` | Lightweight **15.98 MB** pre-trained weight distribution |
| **Top-1 / Top-5 Accuracy** | **34.84%** / **61.54%** | Benchmarked over 32,941 wild test samples |

---

## 🧪 Forensic Verification & CLI Tools

SignBridge includes built-in verification suites to test accuracy and offline inference:

#### Run Full Test Suite Verification
```bash
python -m signbridge.inference.verify_pipeline
```

#### Run CLI Prediction on Video Files
```bash
python -m signbridge.inference.predict_video --video path/to/sample_sign.mp4
```

---

## 🌐 REST API Reference

| Method | Route | Description |
|:---|:---|:---|
| `GET` | `/api/health` | Verifies health status, device accelerator (`mps`/`cuda`/`cpu`), and active classes |
| `POST` | `/api/predict/sequence` | Accepts JSON `{"sequence": [[[x, y, z], ...]]}` shape `(32, 42, 3)` |
| `POST` | `/api/predict/video` | Multipart file upload (`.mp4`, `.mov`, `.webm`) for batch inference |
| `POST` | `/api/debug/sequence` | Diagnostic inspection endpoint (reports wrist coordinates & zero-fills) |
| `GET` | `/api/vocabulary` | Paginated dictionary lookup with search query filtering |
| `GET` | `/api/demonstration/{gloss}` | Returns demonstration video metadata and target video source |
| `GET` | `/api/teach-me` | Returns a curated random sign for discovery |

---

## 📂 Project Structure

```
signbridge/
├── 📦 signbridge/
│   ├── 🧠 inference/          # Predictor engine, GRU model, pipeline verifier
│   ├── 📐 preprocessing/      # MediaPipe landmark extraction & coordinate normalizers
│   └── 🌐 web/                # FastAPI application, templates, CSS & interactive JS
├── 🏋️ trained_models/         # Pre-trained neural network weights (15.98 MB)
├── 📜 requirements.txt        # Production dependency manifest
├── 📖 README.md               # Visual project documentation
└── 🛡️ .gitignore              # Ignores local caches, venvs, and 46GB raw datasets
```

---

## 🛠️ Troubleshooting & FAQ

<details>
<summary><b>1. The webcam isn't starting in the browser</b></summary>

- Ensure your browser has been granted permission to access your camera.
- On **macOS**: Go to *System Settings → Privacy & Security → Camera* and ensure your browser (Chrome, Safari, etc.) is toggled ON.
- If using an external webcam, verify that it is selected as the default input device in your browser's site settings.
</details>

<details>
<summary><b>2. Error: Port 8000 is already in use</b></summary>

You can specify an alternate port when running Uvicorn:
```bash
python -m uvicorn signbridge.web.app:app --host 0.0.0.0 --port 8080
```
Then visit `http://localhost:8080`.
</details>

<details>
<summary><b>3. How do I verify my GPU is being used?</b></summary>

Hit the health check endpoint in your browser or terminal:
```bash
curl http://localhost:8000/api/health
```
Look for `"device": "mps"` (Apple Silicon) or `"device": "cuda"` (NVIDIA).
</details>

---

<div align="center">

Made with ❤️ for inclusive, accessible communication.  
**Contributions, bug reports, and feature requests are welcome!**

[![GitHub Stars](https://img.shields.io/github/stars/swastikparmar987/signbridge?style=social)](https://github.com/swastikparmar987/signbridge/stargazers)
[![GitHub Forks](https://img.shields.io/github/forks/swastikparmar987/signbridge?style=social)](https://github.com/swastikparmar987/signbridge/network/members)

</div>
