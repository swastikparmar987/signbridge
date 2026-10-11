# Deploying SignBridge on Vercel (Backend + Camera)

This guide details how **SignBridge** is deployed to **Vercel** with full backend recognition and live browser camera support.

---

## ⚡ Architecture Overview

```
[ User Browser ]
       │
       ├─► 1. WebRTC Camera Feed (`navigator.mediaDevices.getUserMedia`)
       │      • Served via Vercel SSL (HTTPS) — satisfies browser security
       │
       ├─► 2. Client-side MediaPipe Hands (CDN)
       │      • Real-time 42-landmark extraction across 32 frames
       │
       └─► 3. POST /api/predict/sequence
              │
              ▼
   [ Vercel Edge / Serverless ]
       │
       ├─► Static Assets (`/` & `/static/*`)
       │      • Cached globally on Vercel Edge CDN (0ms cold start)
       │
       └─► Python Serverless Function (`api/index.py`)
              • Engine: ONNX Runtime (CPU)
              • Model: Exp11AdapterClassifier (261 features, 2,731 ASL classes)
              • Total function payload: ~27MB (well under Vercel's 250MB limit)
              • Inference latency: ~25-40ms
```

---

## 🚀 Deployment Instructions

### Method 1: Git Push (Automatic Vercel Integration - Recommended)

If your repository `swastikparmar987/signbridge` is connected to Vercel:

1. **Stage and commit the deployment configuration:**
   ```bash
   git add api/ public/ vercel.json .vercelignore signbridge/inference/onnx_predictor.py
   git commit -m "feat: add Vercel serverless ONNX backend and static frontend"
   git push origin main
   ```

2. **Open Vercel Dashboard:**
   - Go to [vercel.com](https://vercel.com).
   - If not imported yet, click **"Add New..."** → **"Project"** → Select `signbridge`.
   - Leave Framework Preset as **"Other"**.
   - Root Directory: `./`.
   - Click **Deploy**.

---

### Method 2: Deploy via Vercel CLI

1. **Deploy preview deployment:**
   ```bash
   npx vercel
   ```
   Follow the interactive prompts:
   - *Set up and deploy?* `Y`
   - *Which scope?* Select your team/account
   - *Link to existing project?* `N` (or `Y` if already created)
   - *What's your project's name?* `signbridge`
   - *In which directory is your code located?* `./`

2. **Deploy directly to production:**
   ```bash
   npx vercel --prod
   ```

---

## 🔑 Environment Variables (Optional)

Configure these in **Vercel Dashboard → Project Settings → Environment Variables**:

| Variable | Description | Default |
|---|---|---|
| `GROQ_API_KEY` | *(Optional)* Groq API key for Llama 3.1 high-speed ASL gloss-to-English grammar translation | Falls back to rule-based ASL grammar engine |

---

## 📷 Camera & WebRTC Verification

1. When opening the deployed application (`https://<project-name>.vercel.app`):
   - Because Vercel serves the app over **HTTPS**, the browser will automatically allow camera access.
2. Click **"Turn On Camera"** in the Live Translate view.
3. Allow camera permissions when prompted by your browser.
4. Position your hands inside the framing guide:
   - Green bounding landmarks will highlight your hands.
   - Status will update to *"Listening for signs..."*.
   - Recognized ASL glosses appear with confidence ratings in the hero card.

---

## 🧪 Local Simulation & Testing

To test the Vercel serverless backend locally before pushing:

```bash
# Run local test suite against serverless API
.venv/bin/python -c "
from fastapi.testclient import TestClient
from api.index import app
client = TestClient(app)
print(client.get('/api/health').json())
"
```
