import argparse
from pathlib import Path

from signbridge.inference.predictor import SignBridgePredictor

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CHECKPOINT = PROJECT_ROOT / "trained_models" / "best_gru_normalized.pth"


def main():
    parser = argparse.ArgumentParser(description="SignBridge Video Inference CLI")
    parser.add_argument(
        "--video_path",
        type=str,
        required=True,
        help="Path to sign language video file (.mp4)",
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

    video_path = Path(args.video_path)
    predictor = SignBridgePredictor(checkpoint_path=args.checkpoint)

    print("=" * 70)
    print("🎥 SIGNBRIDGE VIDEO INFERENCE")
    print("=" * 70)
    print(f"Video File: {video_path}")

    result = predictor.predict_video(video_path, top_k=args.top_k)

    print("\n🔥 TOP PREDICTIONS:\n")
    for rank, pred in enumerate(result["top_k"], start=1):
        print(f"#{rank}: {pred['gloss']:<25} {pred['confidence']:6.2f}%")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()

