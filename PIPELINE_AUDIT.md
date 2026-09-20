# SignBridge Pipeline Forensic Audit & Architectural Verification

**Project**: SignBridge Isolated Sign Language Recognition  
**Auditor**: Senior ML / CV / MLOps Engineer  
**Date**: September 6, 2026  
**Status**: Verified Production Pipeline Baseline  

---

## Executive Summary

A comprehensive forensic audit of the SignBridge system was conducted across training artifacts, offline test caches, video extraction, inference pipelines, and web frontend/backend integration.

1. **Baseline Model & Test Set Reproduction**:
   - Model: 2-layer `GRUClassifier` (input: 126, hidden: 256, classes: 2,731).
   - Checkpoint: `trained_models/best_gru_normalized.pth` (verified epoch 30, val_acc: 45.38%).
   - Test set: ASL Citizen 32,941 samples.
   - **Top-1 Accuracy**: **34.84%** (11,477 / 32,941) — *Exact Reproduction Verified*.
   - **Top-5 Accuracy**: **61.54%** (20,271 / 32,941) — *Exact Reproduction Verified*.

2. **Video Inference**:
   - Video `10506892472594931-APPLE.mp4` -> MediaPipe Hands -> 32 uniform frames -> `normalize_sequence()` -> GRU.
   - Prediction: **APPLE (99.59% confidence)** — *Exact Video Inference Verified*.

3. **Parity Tests**:
   - Cache Pipeline vs Predictor Sequence Pipeline: **Max Difference = 0.000000** (Index 0, 339, 1000, 5000).
   - Class Mapping: 2,731 classes verified 100% sorted alphabetical match with `test.csv`.

---

## Detailed Forensic Audit Answers

### 1. What exact preprocessing was used during training?
- **Raw Input**: 42 hand landmarks per frame (21 landmarks $\times$ 2 hands $\times$ 3 coordinates: $x, y, z$).
- **Temporal Structure**: 32 frames uniformly sampled from isolated sign videos.
- **Missing Hands**: Filled with exact zeros `(21, 3)`.
- **Zero Frames**: Natural lead-in (resting/preparation) and lead-out (resting/completion) frames are preserved as zeros `(42, 3)`.
- **Wrist-Relative Normalization**: For each non-zero hand $(21, 3)$, subtract Landmark 0 (wrist):
  $$\text{hand}_{\text{rel}} = \text{hand} - \text{wrist}$$
- **Scale Normalization**: Scale by Euclidean distance from wrist (Landmark 0) to middle finger MCP (Landmark 9):
  $$\text{scale} = \|\text{middle\_mcp} - \text{wrist}\|_2, \quad \text{if scale} \ge 10^{-6}: \text{hand}_{\text{norm}} = \frac{\text{hand}_{\text{rel}}}{\text{scale}}$$
- **Flattening**: Reshaped from $(32, 42, 3)$ to $(32, 126)$.

### 2. What exact preprocessing is used during cache inference?
- Loads `.npy` file of raw coordinates of shape $(32, 42, 3)$.
- Calls `signbridge.preprocessing.normalize.normalize_sequence()` (single source of truth).
- Reshapes to $(32, 126)$ and passes to PyTorch GRU.

### 3. What exact preprocessing is used during video inference?
- OpenCV reads video file, extracts total frame count $N$.
- Uniformly samples 32 frame indices: `np.linspace(0, N - 1, 32).astype(int)`.
- Runs MediaPipe Hands (`min_detection_confidence=0.5`, `max_num_hands=2`).
- Detected hands are sorted left-to-right by wrist $x$-coordinate: `detected_hands.sort(key=lambda x: x[0])`.
- Primary hand in slot 0 (0..20), secondary hand in slot 1 (21..41).
- Sequences are passed through `normalize_sequence()` and flattened to $(32, 126)$.

### 4. What exact preprocessing is used during webcam inference?
- WebRTC camera captures frames into buffer.
- MediaPipe Hands JS extracts landmarks $[42, 3]$ (2 hands $\times$ 21 points, sorted by wrist $x$).
- Timed capture records full $2.0\text{s}$ gesture (~60 frames).
- Resampled to 32 frames via uniform index mapping: `Math.floor(i * (total - 1) / 31)`.
- Raw $(32, 42, 3)$ coordinates dispatched to `/api/predict/sequence`.
- Backend validates shape and executes `normalize_sequence()` before GRU inference.

### 5. Are all pipelines mathematically identical?
- **Yes**. `normalize_sequence()` in `signbridge/preprocessing/normalize.py` is the single source of truth used across all Python inference pathways and the FastAPI backend.
- Ad-hoc multi-trial heuristics previously introduced in `/api/predict/sequence` were removed during this audit to enforce strict pipeline purity.

### 6. Are hand ordering rules identical?
- **Python**: `detected_hands.sort(key=lambda x: x[0])` (ascending by wrist $X$).
- **JavaScript**: `detectedHands.sort((a, b) => a.wristX - b.wristX)` (ascending by wrist $X$).
- Both sort leftmost hand to slot 0 and rightmost hand to slot 1.

### 7. Are coordinates represented identically?
- Both Python MediaPipe and MediaPipe JS represent $x \in [0, 1]$, $y \in [0, 1]$, and $z$ as relative depth.

### 8. Are zero frames preserved?
- **Yes**. Lead-in preparation frames and trailing resting frames without hands are preserved as zeros $(42, 3)$. No frames are dropped.

### 9. Is normalization identical?
- **Yes**. Only the backend executes `normalize_sequence()`. Frontend sends raw landmarks.

### 10. Is the temporal sampling identical?
- **Yes**. Python `np.linspace(0, total-1, 32).astype(int)` and JS `Math.floor(i * (total - 1) / 31)` produce an identical index sequence for all frame counts from 32 to 300+.

### 11. Is the class mapping identical?
- **Yes**. Checkpoint contains 2,731 sorted unique classes identical to ASL Citizen `test.csv`.

---

## Test Verification Matrix

| Verification Step | Target / Reference | Measured Result | Status |
| :--- | :--- | :--- | :--- |
| **Test Set Top-1 Accuracy** | 34.84% | **34.84%** (11,477 / 32,941) | PASSED |
| **Test Set Top-5 Accuracy** | 61.54% | **61.54%** (20,271 / 32,941) | PASSED |
| **Cache Sample 339 (APPLE)** | Top-1 APPLE > 90% | **99.22%** | PASSED |
| **Video Inference (APPLE)** | Top-1 APPLE > 90% | **99.59%** | PASSED |
| **Tensor Parity Max Diff** | 0.000000 | **0.000000** | PASSED |
| **Class Vocabulary Parity** | 2,731 classes | **2,731 classes** (100% match) | PASSED |
