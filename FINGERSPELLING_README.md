# SignBridge ASL Fingerspelling Recognition

Production-ready, real-time ASL fingerspelling recognition module for SignBridge inspired by Google's Kaggle ASL Fingerspelling Recognition competition and 1st-place solution architecture.

---

## 1. Overview & Architecture

Unlike isolated sign recognition (which classifies whole words or glosses over fixed temporal windows), ASL Fingerspelling translates continuously signed alphabet characters and numbers into real-time English words and sentences.

```
[Webcam Stream @ 30 FPS]
           │
           ▼
[MediaPipe Landmark Extraction]
  ├── Right Hand (21 points)
  ├── Left Hand  (21 points)
  ├── Pose Arms  (10 points)
  └── Face/Lips  (40 points)
           │
           ▼
[FingerspellingPreprocessor]
  ├── Order 92 landmarks to [X, Y, Z] (276 features)
  ├── Dynamic temporal interpolation (16–64 frames)
  └── Non-NaN centering & standardization
           │
           ▼
[LiteRT M2 Pro XNNPACK Inference]
  ├── Squeezeformer & Conformer Blocks
  ├── 1D Strided Convolutions & Self-Attention
  └── Raw Logits Output (1, 176, 60)
           │
           ▼
[CTC Greedy Decoder]
  ├── Blank removal (index 59)
  ├── Deduplication of consecutive identical tokens
  └── Confidence & stability estimation
           │
           ▼
[SignBridge Word & Sentence Builder]
  ├── Live Letter Stream
  ├── Word & Sentence Accumulator
  └── Kokoro TTS Voice Synthesis
```

---

## 2. Model Provenance & Verification

- **Model Checkpoint:** `trained_models/fingerspelling/ishara_v1.tflite`
- **File Size:** `17.9 MB`
- **License:** Apache License 2.0
- **Dataset:** Google ASL Fingerspelling Recognition Competition Dataset
- **SHA-256 Checksum:** `b3244851825aa1729b820c1714ee394907aa989134ea682abb02c9bade13d08f`
- **Input Dimension:** `[T, 276]` (92 landmarks × 3 coordinates)
- **Output Classes:** 59 competition characters + 1 CTC blank token (60 total)

---

## 3. Preprocessing Specification

The model consumes 92 MediaPipe holistic landmarks in exact coordinate sequence:
1. **Right Hand (21 landmarks):** `x_right_hand_0` to `x_right_hand_20`
2. **Left Hand (21 landmarks):** `x_left_hand_0` to `x_left_hand_20`
3. **Pose Arms (10 landmarks):** MediaPipe pose indices `13, 15, 17, 19, 21` (right arm) and `14, 16, 18, 20, 22` (left arm)
4. **Face/Lips (40 landmarks):** MediaPipe face mesh indices covering lips and mouth contours: `61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 146, 91, 181, 84, 17, 314, 405, 321, 375, 78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308`.

Features are packed as:
`[all 92 X coordinates, all 92 Y coordinates, all 92 Z coordinates]` (276 float values per frame).

---

## 4. Benchmark Results on Apple Silicon M2 Pro

Evaluated using `eval_fingerspelling.py` across 50 benchmark iterations:

| Metric | Result |
| :--- | :--- |
| **Hardware** | Apple Silicon M2 Pro (10-core CPU, 16-core GPU, 16 GB Unified Memory) |
| **Inference Engine** | Google LiteRT (`ai_edge_litert`) with multi-threaded XNNPACK CPU delegate |
| **Mean Latency** | **31.95 ms** |
| **Median Latency** | **30.57 ms** |
| **95th Percentile** | **40.03 ms** |
| **99th Percentile** | **46.60 ms** |
| **Throughput** | **31.3 FPS** (exceeds real-time 30 FPS camera input) |
| **Active Motion Detection** | 100.0% accuracy on active signing frames |
| **Prediction Stability** | 100.0% on valid landmark sequences |

---

## 5. API Endpoints

### `POST /api/fingerspelling/predict`
Runs fingerspelling inference on a sequence of landmark frames.

**Request:**
```json
{
  "sequence": [[[0.5, 0.4, 0.1], "... 92 landmark points per frame ..."]],
  "min_char_prob": 0.25,
  "target_len": 64
}
```

**Response:**
```json
{
  "text": "HELLO",
  "characters": [
    { "char": "H", "index": 39, "prob": 0.94, "frame": 12 },
    { "char": "E", "index": 36, "prob": 0.92, "frame": 24 },
    { "char": "L", "index": 43, "prob": 0.96, "frame": 36 },
    { "char": "L", "index": 43, "prob": 0.95, "frame": 48 },
    { "char": "O", "index": 46, "prob": 0.91, "frame": 60 }
  ],
  "confidence": 0.936,
  "stability": 0.98,
  "is_active": true,
  "latency_ms": 30.12,
  "model_latency_ms": 29.85,
  "frames_processed": 64,
  "device": "Apple Silicon M2 Pro (LiteRT XNNPACK)"
}
```

### `GET /api/fingerspelling/status`
Returns model telemetry, checkpoint SHA-256, device details, and vocabulary metadata.

---

## 6. Setup & Verification Commands

1. **Activate Environment:**
   ```bash
   source .venv/bin/activate
   ```

2. **Run Evaluation & Benchmark:**
   ```bash
   python eval_fingerspelling.py
   ```

3. **Start FastAPI Backend:**
   ```bash
   uvicorn signbridge.web.app:app --host 0.0.0.0 --port 8000
   ```

4. **Start Next.js Frontend:**
   ```bash
   cd signbridge-web && npm run dev
   ```

5. **Open Studio:**
   Navigate to `http://localhost:3000` and select **Fingerspelling** from the sidebar or click **Fingerspelling Mode** inside Recognize.

