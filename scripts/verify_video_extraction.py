from pathlib import Path
import cv2
import numpy as np

from signbridge.preprocessing.landmarks import extract_hand_sequence, get_sampled_frame_indices
from signbridge.preprocessing.normalize import normalize_sequence
from signbridge.inference.predictor import SignBridgePredictor

def verify_video_extraction():
    video_path = Path("dataset/ASL_Citizen/videos/10506892472594931-APPLE.mp4")
    cache_path = Path("landmark_cache_full/test/339.npy")

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    sampled_indices = get_sampled_frame_indices(total_frames, 32)
    extracted_seq = extract_hand_sequence(video_path, 32)
    cached_seq = np.load(cache_path).astype(np.float32)

    print("=" * 60)
    print("🎥 VIDEO EXTRACTION VERIFICATION REPORT")
    print("=" * 60)
    print("Video File:            ", video_path.name)
    print(f"1. Original FPS:        {fps:.2f}")
    print(f"2. Total frames:        {total_frames} ({width}x{height})")
    print(f"3. Sampled indices (32): {sampled_indices.tolist()}")
    print(f"4. Extracted shape:     {extracted_seq.shape}")
    print(f"   Cached shape:        {cached_seq.shape}")

    ext_h0_active = int(np.sum(~np.all(extracted_seq[:, :21, :] == 0, axis=(1, 2))))
    ext_h1_active = int(np.sum(~np.all(extracted_seq[:, 21:, :] == 0, axis=(1, 2))))
    cac_h0_active = int(np.sum(~np.all(cached_seq[:, :21, :] == 0, axis=(1, 2))))
    cac_h1_active = int(np.sum(~np.all(cached_seq[:, 21:, :] == 0, axis=(1, 2))))

    print(f"5. Extracted Hand0 active: {ext_h0_active}/32 frames | Hand1 active: {ext_h1_active}/32 frames")
    print(f"   Cached Hand0 active:    {cac_h0_active}/32 frames | Hand1 active: {cac_h1_active}/32 frames")

    raw_diff = float(np.max(np.abs(extracted_seq - cached_seq)))
    raw_mean_diff = float(np.mean(np.abs(extracted_seq - cached_seq)))
    print(f"6. Raw landmark max diff:  {raw_diff:.6f} | Mean diff: {raw_mean_diff:.6f}")

    norm_ext = normalize_sequence(extracted_seq)
    norm_cac = normalize_sequence(cached_seq)
    norm_diff = float(np.max(np.abs(norm_ext - norm_cac)))
    norm_mean_diff = float(np.mean(np.abs(norm_ext - norm_cac)))
    print(f"7. Norm sequence max diff: {norm_diff:.6f} | Mean diff: {norm_mean_diff:.6f}")

    predictor = SignBridgePredictor("trained_models/best_gru_normalized.pth")
    p_cache = predictor.predict_cache(cache_path)
    p_video = predictor.predict_video(video_path)

    print(f"8. Cached Prediction:   {p_cache['predicted_gloss']} ({p_cache['confidence']:.2f}%)")
    print("   Top-5 Cache:        ", [(p["gloss"], f"{p['confidence']:.2f}%") for p in p_cache["top_k"]])
    print(f"9. Video Prediction:    {p_video['predicted_gloss']} ({p_video['confidence']:.2f}%)")
    print("   Top-5 Video:        ", [(p["gloss"], f"{p['confidence']:.2f}%") for p in p_video["top_k"]])
    print("=" * 60)

if __name__ == "__main__":
    verify_video_extraction()
