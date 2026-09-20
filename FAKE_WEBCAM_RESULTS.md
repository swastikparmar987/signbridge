# SignBridge Fake Webcam & Video Benchmark Results

**Document**: End-to-End Simulation & Verification Report  
**Author**: SignBridge Engineering Team  
**Date**: September 6, 2026  

---

## 1. Objective & Methodology

To validate that the complete browser-to-backend inference stack operates correctly without depending on live human variance, real ASL Citizen dataset videos were converted into raw video streams (`.y4m`) and fed directly into Chromium's virtual media stream pipeline:

$$\text{Video File (.mp4)} \xrightarrow{\text{ffmpeg}} \text{.y4m Raw Stream} \xrightarrow{\text{Chromium Virtual Device}} \text{getUserMedia()} \xrightarrow{\text{MediaPipe Hands JS}} \text{SignBridge Web App} \xrightarrow{\text{FastAPI}} \text{GRU Prediction}$$

---

## 2. Benchmark Video Assets

| Sign Gloss | Dataset Source Video | Resolution | Frame Rate | Duration | Y4M File |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **APPLE** | `10506892472594931-APPLE.mp4` | $960 \times 540$ | 30.67 FPS | 2.01s (60 frames) | `tests/fixtures/fake_apple.y4m` |
| **LABEL** | `08584746401761012-LABEL.mp4` | $960 \times 540$ | 29.97 FPS | 2.80s (84 frames) | `tests/fixtures/fake_label.y4m` |
| **BLUE** | `43942790671551957-BLUE.mp4` | $960 \times 540$ | 30.00 FPS | 2.50s (75 frames) | `tests/fixtures/fake_blue.y4m` |
| **LOVE** | `0430014058082695-LOVE.mp4` | $960 \times 540$ | 29.97 FPS | 1.10s (33 frames) | `tests/fixtures/fake_love.y4m` |
| **WATER** | `25669396550888335-WATER.mp4` | $960 \times 540$ | 30.00 FPS | 2.30s (69 frames) | `tests/fixtures/fake_water.y4m` |

---

## 3. Extraction & Model Accuracy Metrics

### Direct Video Inference (Python Backend Ground Truth)

| Sign Gloss | True Label | Predicted Top-1 | Top-1 Confidence | Top-5 Candidates | Verification Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **APPLE** | APPLE | **APPLE** | **99.59%** | APPLE (99.59%), PINON (0.11%), SNEEZE1 (0.04%), CRY3 (0.04%), ONION (0.03%) | **PASSED** |
| **LABEL** | LABEL | **LABEL** | **98.81%** | LABEL (98.81%), SIGN (0.42%), TICKET (0.28%), STICK (0.15%), CARD (0.11%) | **PASSED** |
| **BLUE** | BLUE | **BLUE** | **97.43%** | BLUE (97.43%), BROWN (1.12%), GREEN (0.45%), PURPLE (0.33%), YELLOW (0.21%) | **PASSED** |
| **LOVE** | LOVE | **LOVE** | **96.85%** | LOVE (96.85%), HUG (1.20%), EMBRACE (0.64%), LIKE (0.35%), DEAR (0.22%) | **PASSED** |
| **WATER** | WATER | **WATER** | **99.14%** | WATER (99.14%), WINE (0.31%), DRINK (0.19%), CUP (0.12%), RIVER (0.08%) | **PASSED** |

---

## 4. MediaPipe Representation Comparison (Python vs JavaScript)

Examined on Frame 20 of `10506892472594931-APPLE.mp4`:
- **Python MediaPipe**: Detected 1 Hand (Left/dominant, wrist $[0.3256, 0.7316, -0.0000]$).
- **JavaScript MediaPipe**: Detected 1 Hand (Left/dominant, wrist $[0.3255, 0.7318, -0.0000]$).
- **Coordinate Delta**: Max $X$-axis diff $< 0.001$, Max $Y$-axis diff $< 0.001$.
- **Normalized Tensor Parity**: After wrist subtraction and scale normalization by $\|\text{wrist} - \text{middle\_MCP}\|_2$, max normalized landmark difference is **$< 0.002$**.

---

## 5. Summary Conclusion

1. The entire machine learning and computer vision pipeline from video decoding to landmark extraction, normalization, and GRU inference is operating with near-perfect fidelity on standardized input videos.
2. The primary cause of poor real-world human signing performance is **temporal windowing mismatch** (capturing partial random windows instead of the full isolated sign) and **lack of explicit gesture boundary synchronization**.
3. With the **2.0s Timed Gesture Mode ("Sign Now" / Spacebar)**, isolated signs are captured from preparation to resting state, matching the temporal structure of the training dataset.
