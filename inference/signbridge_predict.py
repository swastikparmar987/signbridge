import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from signbridge.inference.predictor import SignBridgePredictor

CHECKPOINT_PATH = PROJECT_DIR / "trained_models" / "best_gru_normalized.pth"
CACHE_DIR = PROJECT_DIR / "landmark_cache_full" / "test"
TEST_CSV = PROJECT_DIR / "dataset" / "ASL_Citizen" / "splits" / "test.csv"
TEST_INDEX = 339


def main():
    print("=" * 70)
    print("🧠 SIGNBRIDGE PREDICT SCRIPT")
    print("=" * 70)

    predictor = SignBridgePredictor(checkpoint_path=CHECKPOINT_PATH)
    print(f"Device: {predictor.device}")

    # Check cache file
    cache_file = CACHE_DIR / f"{TEST_INDEX}.npy"
    print(f"\nTesting Cache File: {cache_file}")

    if not cache_file.exists():
        raise FileNotFoundError(f"Cache file not found: {cache_file}")

    result = predictor.predict_cache(cache_file, top_k=5)

    print("\n🔥 TOP-5 PREDICTIONS\n")
    for rank, pred in enumerate(result["top_k"], start=1):
        print(f"#{rank}: {pred['gloss']:<25} {pred['confidence']:6.2f}%")

    print("\n" + "=" * 70)
    print("🔍 CACHE INDEX VERIFICATION")
    print("=" * 70)

    if TEST_CSV.exists():
        import pandas as pd
        test_df = pd.read_csv(TEST_CSV)
        if TEST_INDEX < len(test_df):
            row = test_df.iloc[TEST_INDEX]
            print(f"CACHE INDEX: {TEST_INDEX}")
            print(f"TRUE GLOSS:  {row['Gloss']}")
            print(f"VIDEO FILE:  {row['Video file']}")
            print(f"\nMatch Result: {'✅ MATCH' if row['Gloss'] == result['predicted_gloss'] else '❌ DIFFERENT'}")
    else:
        print("Test CSV not found.")


if __name__ == "__main__":
    main()