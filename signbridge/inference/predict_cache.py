import argparse
from pathlib import Path
import pandas as pd

from signbridge.inference.predictor import SignBridgePredictor

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CHECKPOINT = PROJECT_ROOT / "trained_models" / "best_gru_normalized.pth"
TEST_CSV = PROJECT_ROOT / "dataset" / "ASL_Citizen" / "splits" / "test.csv"


def main():
    parser = argparse.ArgumentParser(description="SignBridge Cache Inference CLI")
    parser.add_argument(
        "--cache_path",
        type=str,
        required=True,
        help="Path to landmark cache .npy file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(DEFAULT_CHECKPOINT),
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of top predictions to display",
    )

    args = parser.parse_args()

    cache_path = Path(args.cache_path)
    predictor = SignBridgePredictor(checkpoint_path=args.checkpoint)

    print("=" * 70)
    print("🎯 SIGNBRIDGE CACHE INFERENCE")
    print("=" * 70)
    print(f"Cache File: {cache_path}")

    # Check ground truth metadata if cache_path stem is numeric index
    if cache_path.stem.isdigit() and TEST_CSV.exists():
        row_idx = int(cache_path.stem)
        test_df = pd.read_csv(TEST_CSV)
        if row_idx < len(test_df):
            row = test_df.iloc[row_idx]
            print(f"True Gloss: {row['Gloss']}")
            print(f"Original Video: {row['Video file']}")

    result = predictor.predict_cache(cache_path, top_k=args.top_k)

    print("\n🔥 TOP PREDICTIONS:\n")
    for rank, pred in enumerate(result["top_k"], start=1):
        print(f"#{rank}: {pred['gloss']:<25} {pred['confidence']:6.2f}%")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()

