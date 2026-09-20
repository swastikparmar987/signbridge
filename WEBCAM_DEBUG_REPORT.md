# SignBridge Webcam Debug & Forensic Report

**Document**: Root Cause Analysis of Real-World Webcam vs Dataset Inference  
**Author**: SignBridge Engineering Team  
**Date**: September 6, 2026  

---

## 1. The Core Paradox

The offline model and pipeline achieve:
- **34.84% Top-1** & **61.54% Top-5** across all 32,941 ASL Citizen test sequences.
- **99.59% confidence** on raw dataset video files (e.g., `10506892472594931-APPLE.mp4`).
- **99.22% confidence** on cached numpy landmark files.

Yet, real-time webcam inference with live human signers historically produced unstable or low-confidence predictions.

---

## 2. Root Cause Breakdown

Our forensic audit isolated the following contributing factors:

### Factor A: Continuous Stream vs Isolated Gesture Mismatch (Primary Cause)
- **Training Data**: Every training video is an isolated sign clip ($1.5 - 3.0\text{s}$ total) starting with rest/preparation zeros, moving through the sign gesture peak, and returning to rest zeros. The 32 uniform samples capture the entire lifecycle of the sign.
- **Previous Webcam Approach**: Pushed frames into a continuous 32-frame buffer at camera FPS (~30 FPS).
- **Failure Mode**: 32 frames at 30 FPS captured only a ~1.0-second random window (e.g. half of the motion or transition state), completely breaking the recurrent GRU temporal representation.
- **Solution**: Implemented explicit **2.0s gesture capture mode ("Sign Now" / Spacebar)** that buffers the full motion including lead-in and lead-out, then uniformly downsamples to 32 frames.

### Factor B: Hand Ordering & Camera Mirroring
- In standard camera feeds, selfie preview is mirrored visually with CSS `transform: scaleX(-1)`, but the raw video frame dispatched to MediaPipe JS is the direct camera sensor output.
- Both Python and JS implementations order hands left-to-right by wrist $X$-coordinate (`wristX` ascending).
- For single-hand signs (e.g. APPLE), the active hand lands in slot 0 (indices 0..20) and slot 1 is zero-filled `(21..41)`.

### Factor C: Non-Zero Zero-Frame Preservation
- When signers transition between signs or pause, MediaPipe detects 0 hands.
- In earlier experimental code, zero frames were filtered out (`if (hasHands) push(frame)`). This stripped essential resting context and caused temporal distortion.
- **Fix**: All frames during the 2.0s capture window are recorded faithfully. If no hands are present, `zeros(42, 3)` are stored.

### Factor D: Preprocessing Single Source of Truth
- An ad-hoc ensemble trial had been introduced in `/api/predict/sequence` which performed multiple perturbed inferences (slot swap + temporal shift) and voted with summed confidence scores.
- **Fix**: Replaced with direct, standard `predictor.predict_sequence(sequence_np, is_normalized=False)` to guarantee mathematical identity with the verified offline pipeline.

---

## 3. Real-Time Telemetry & Diagnostic HUD (Key 'D')

The application now features real-time telemetry HUD displaying:
- **Camera Status & FPS**: Active stream resolution and connection state.
- **MediaPipe Detector**: Model complexity, initialization status, and tracking health.
- **Hand Counter**: Real-time detected hands count (0, 1, 2).
- **Temporal Buffering**: Current capture mode (Idle, Recording 2.0s, Sampling 32 frames).
- **Latency & Roundtrip**: API request dispatch and response latency in milliseconds.
- **Model Output**: Raw Top-1 gloss, raw confidence percentage, Top-5 candidate list, and zero-frame diagnostics.
