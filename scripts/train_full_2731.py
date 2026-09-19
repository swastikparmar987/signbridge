#!/usr/bin/env python3
"""
SignBridge Full 2,731-Class Experiment 1: Augmented Fine-Tuning
"""

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from signbridge.training.config import get_full_experiment_config
from signbridge.training.train import run_full_training
from signbridge.training.evaluate import evaluate_full_model


def main():
    print("=" * 80)
    print("🚀 SIGNBRIDGE FULL 2,731-CLASS EXPERIMENT 1: AUGMENTED FINE-TUNING")
    print("=" * 80)

    config = get_full_experiment_config("exp1_augmented_finetune")
    print(f"Name:                  {config.name}")
    print(f"Pretrained Source:     {config.pretrained_checkpoint}")
    print(f"Output Directory:      {config.output_dir}")
    print(f"Batch Size:            {config.batch_size}")
    print(f"Fine-Tuning LR:        {config.phase2_lr}")
    print(f"Max Epochs:            {config.phase2_epochs}")
    print(f"Augmentation:          {config.augmentation.enabled}")
    print("=" * 80)

    # 1. Run full fine-tuning
    t_start = time.time()
    train_result = run_full_training(config)
    train_duration = time.time() - t_start

    # 2. Evaluate best checkpoint on untouched 32,941-sample test set
    best_ckpt = Path(train_result["checkpoint_path"])
    print("\n" + "=" * 80)
    print(f"🧪 EVALUATING BEST CHECKPOINT ON UNTOUCHED TEST SET (32,941 SAMPLES)...")
    print(f"   Checkpoint: {best_ckpt}")
    print("=" * 80)

    t_eval_start = time.time()
    eval_result = evaluate_full_model(best_ckpt, batch_size=256)
    eval_duration = time.time() - t_eval_start

    # Exclude huge raw confusion matrix from summary JSON, keep top confusions & per-class
    eval_summary = {
        "experiment": config.name,
        "checkpoint_path": str(best_ckpt),
        "best_epoch": train_result["best_epoch"],
        "best_val_top1": train_result["best_val_top1"],
        "test_top1_accuracy": eval_result["top1_accuracy"],
        "test_top3_accuracy": eval_result["top3_accuracy"],
        "test_top5_accuracy": eval_result["top5_accuracy"],
        "macro_f1": eval_result["macro_f1"],
        "weighted_f1": eval_result["weighted_f1"],
        "total_test_samples": eval_result["total_samples"],
        "training_duration_seconds": round(train_duration, 2),
        "evaluation_duration_seconds": round(eval_duration, 2),
        "top_confusions": eval_result["top_confusions"],
        "worst_classes": eval_result["worst_classes"],
        "best_classes": eval_result["best_classes"],
    }

    eval_file = config.output_dir / "eval_results.json"
    with open(eval_file, "w") as f:
        json.dump(eval_summary, f, indent=2)

    # 3. Print Final Benchmark Comparison
    baseline_top1 = 34.84
    baseline_top5 = 61.54
    new_top1 = eval_result["top1_accuracy"]
    new_top5 = eval_result["test_top5_accuracy"] if "test_top5_accuracy" in eval_result else eval_result["top5_accuracy"]
    delta_top1 = new_top1 - baseline_top1
    delta_top5 = new_top5 - baseline_top5

    print("\n" + "=" * 80)
    print("📊 FINAL BENCHMARK RESULTS & COMPARISON")
    print("=" * 80)
    print(f"Total Training Duration:   {train_duration / 60:.1f} minutes ({train_duration:.1f}s)")
    print(f"Best Validation Epoch:     Epoch {train_result['best_epoch']}")
    print(f"Best Validation Top-1:     {train_result['best_val_top1']:.2f}%")
    print(f"Test Top-1 Accuracy:       {new_top1:.2f}% ({'+' if delta_top1 >= 0 else ''}{delta_top1:.2f}%)")
    print(f"Test Top-3 Accuracy:       {eval_result['top3_accuracy']:.2f}%")
    print(f"Test Top-5 Accuracy:       {new_top5:.2f}% ({'+' if delta_top5 >= 0 else ''}{delta_top5:.2f}%)")
    print(f"Test Macro F1:             {eval_result['macro_f1']:.2f}%")
    print(f"Test Weighted F1:          {eval_result['weighted_f1']:.2f}%")
    print(f"Total Test Samples:        {eval_result['total_samples']:,}")
    print(f"Evaluation Time:           {eval_duration:.1f}s")
    print(f"Results Saved:             {eval_file}")
    print("=" * 80)

    if delta_top1 > 0:
        print(f"🎉 SUCCESS: Model improved over baseline Top-1 by +{delta_top1:.2f}%!")
    else:
        print(f"ℹ️ Result: Top-1 change is {delta_top1:.2f}% relative to baseline.")
    print("=" * 80)


if __name__ == "__main__":
    main()
