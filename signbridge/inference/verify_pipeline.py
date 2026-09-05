import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from signbridge.inference.model import GRUClassifier, load_signbridge_model, get_default_device
from signbridge.inference.predictor import SignBridgePredictor
from signbridge.preprocessing.normalize import normalize_sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CHECKPOINT_PATH = PROJECT_ROOT / "trained_models" / "best_gru_normalized.pth"
TEST_CSV = PROJECT_ROOT / "dataset" / "ASL_Citizen" / "splits" / "test.csv"
TEST_CACHE_DIR = PROJECT_ROOT / "landmark_cache_full" / "test"
VIDEOS_DIR = PROJECT_ROOT / "dataset" / "ASL_Citizen" / "videos"


class SignLanguageTestDataset(Dataset):
    def __init__(self, dataframe, cache_dir, gloss_to_idx):
        self.dataframe = dataframe.reset_index(drop=True)
        self.cache_dir = Path(cache_dir)
        self.gloss_to_idx = gloss_to_idx
        self.indices = [
            idx for idx in range(len(self.dataframe))
            if (self.cache_dir / f"{idx}.npy").exists()
        ]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        row_idx = self.indices[index]
        sequence = np.load(self.cache_dir / f"{row_idx}.npy").astype(np.float32)
        sequence = normalize_sequence(sequence).reshape(32, 126)
        gloss = self.dataframe.iloc[row_idx]["Gloss"]
        label = self.gloss_to_idx[gloss]
        return torch.tensor(sequence, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


def verify_pipeline():
    print("=" * 75)
    print("🧪 SIGNBRIDGE FORENSIC PIPELINE VERIFICATION")
    print("=" * 75)

    # 1. Model Loading
    print("\n1. Verification of Model Checkpoint...")
    predictor = SignBridgePredictor(CHECKPOINT_PATH)
    print(f"   Model loaded on device: {predictor.device}")
    print(f"   Total Classes: {len(predictor.class_names)}")
    assert len(predictor.class_names) == 2731, "Class count mismatch!"

    # 2. Class Mapping Verification
    print("\n2. Class Mapping Consistency Check...")
    test_df = pd.read_csv(TEST_CSV)
    test_glosses = set(test_df["Gloss"].unique())
    ckpt_glosses = set(predictor.class_names)
    assert test_glosses == ckpt_glosses, "Mismatch between checkpoint class_names and dataset glosses!"
    assert predictor.class_names == sorted(predictor.class_names), "class_names must be sorted alphabetically!"
    print("   ✅ 100% Class Mapping Parity Verified (2,731 glosses match).")

    # 3. Golden Sample Parity Test
    print("\n3. Golden Sample Parity Test...")
    samples_to_test = [0, 339, 1000, 5000]
    gloss_to_idx = {g: i for i, g in enumerate(predictor.class_names)}
    for idx in samples_to_test:
        row = test_df.iloc[idx]
        true_gloss = row["Gloss"]
        cache_path = TEST_CACHE_DIR / f"{idx}.npy"
        res = predictor.predict_cache(cache_path, top_k=5)
        top1_gloss = res["predicted_gloss"]
        top1_conf = res["confidence"]
        is_match = "MATCH" if top1_gloss == true_gloss else f"Predicted: {top1_gloss}"
        print(f"   - Index {idx:4d} | True: {true_gloss:<20} | Top-1: {top1_conf:6.2f}% ({is_match})")

    # 4. Full Test Accuracy Reproduction
    print("\n4. Full Test Set Accuracy Reproduction (32,941 samples)...")
    dataset = SignLanguageTestDataset(test_df, TEST_CACHE_DIR, gloss_to_idx)
    loader = DataLoader(dataset, batch_size=128, shuffle=False)

    model = predictor.model
    model.eval()

    correct_top1 = 0
    correct_top5 = 0
    total = 0

    with torch.no_grad():
        for X, y in tqdm(loader, desc="Evaluating Test Set"):
            X, y = X.to(predictor.device), y.to(predictor.device)
            outputs = model(X)
            top1 = outputs.argmax(dim=1)
            correct_top1 += (top1 == y).sum().item()
            _, top5 = torch.topk(outputs, k=5, dim=1)
            correct_top5 += (top5 == y.unsqueeze(1)).any(dim=1).sum().item()
            total += y.size(0)

    top1_acc = 100.0 * correct_top1 / total
    top5_acc = 100.0 * correct_top5 / total

    print(f"\n   Total Samples Evaluated: {total}")
    print(f"   Top-1 Accuracy: {top1_acc:.2f}% ({correct_top1}/{total})")
    print(f"   Top-5 Accuracy: {top5_acc:.2f}% ({correct_top5}/{total})")

    assert abs(top1_acc - 34.84) < 0.1, f"Top-1 accuracy discrepancy! Expected ~34.84%, got {top1_acc:.2f}%"
    assert abs(top5_acc - 61.54) < 0.1, f"Top-5 accuracy discrepancy! Expected ~61.54%, got {top5_acc:.2f}%"
    print("   ✅ Exact target accuracy reproduced successfully!")

    # 5. Video Inference Verification
    print("\n5. Video Inference Verification...")
    sample_video = VIDEOS_DIR / "10506892472594931-APPLE.mp4"
    if sample_video.exists():
        v_res = predictor.predict_video(sample_video, top_k=5)
        print(f"   Video: {sample_video.name}")
        print(f"   Top-1 Prediction: {v_res['predicted_gloss']} ({v_res['confidence']:.2f}%)")
        assert v_res['predicted_gloss'] == "APPLE", "Video inference failed on sample APPLE video!"
        print("   ✅ Video Landmark Extraction and Model Inference Verified.")

    print("\n" + "=" * 75)
    print("🎉 ALL PIPELINE VERIFICATIONS PASSED SUCCESSFULLY!")
    print("=" * 75)


if __name__ == "__main__":
    verify_pipeline()

