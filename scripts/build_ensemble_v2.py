#!/usr/bin/env python3
"""
Validation-Calibrated Ensemble Builder, Perturbation Auditor, and Promotion Gate.

Loads trained Kinematic-v2 checkpoints (e.g., EXP16 SignConformer + EXP17
SignPyramidBiGRU), optimizes ensemble weights, per-model temperatures, and
OOD threshold on the held-out validation split, and verifies final Ensemble
performance on all 32,941 untouched test.csv samples + full perturbation suite.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_exp16_kinematic_conformer import (
    compute_macro_f1,
    compute_roc_auroc_and_fpr95,
    eval_ood_scores,
    evaluate_features,
    mirror_split,
    precompute_features,
    stabilize_split,
)
from signbridge.inference.model import get_default_device, load_signbridge_model
from signbridge.inference.predictor import EnsemblePredictor, SignBridgePredictor
from signbridge.preprocessing.kinematics import build_kinematic_v2_torch
from signbridge.training.gpu_augment import synthesize_ood_negative_batch


def perturb_offset_crop(x_np: np.ndarray, dx: float = 0.05, dy: float = -0.04, scale: float = 0.92) -> np.ndarray:
    """Apply camera framing shift & slight zoom crop while keeping empty zeros intact."""
    out = x_np.copy()
    mask = np.abs(out).sum(axis=-1, keepdims=True) > 1e-4
    shifted = out.copy()
    shifted[..., 0] = (out[..., 0] - 0.5) * scale + 0.5 + dx
    shifted[..., 1] = (out[..., 1] - 0.5) * scale + 0.5 + dy
    return np.where(mask, shifted, 0.0).astype(np.float32)


def perturb_drop_frames(x_np: np.ndarray, keep_ratio: float = 0.80) -> np.ndarray:
    """Drop 20% of frames from boundaries/interior and linearly resample back to 32 frames."""
    n, t, j, c = x_np.shape
    keep_len = max(8, int(round(t * keep_ratio)))
    start = (t - keep_len) // 2
    sub = x_np[:, start : start + keep_len]
    idx = np.round(np.linspace(0, keep_len - 1, t)).astype(np.int64)
    return sub[:, idx].astype(np.float32)


@torch.no_grad()
def build_ood_features(raw_stabilized_x: np.ndarray, device: torch.device, num_ood: int = 300, seed: int = 123) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(raw_stabilized_x), size=min(num_ood, len(raw_stabilized_x)), replace=False)
    sample_t = torch.from_numpy(raw_stabilized_x[idx]).to(device=device, dtype=torch.float32)
    neg_raw = synthesize_ood_negative_batch(sample_t)
    return build_kinematic_v2_torch(neg_raw).cpu()


def score_blended_probs(
    logits_list: List[np.ndarray],
    weights: List[float],
    temps: List[float],
    y_np: np.ndarray,
) -> Dict[str, float]:
    total_w = sum(weights)
    norm_w = [w / max(1e-8, total_w) for w in weights]
    blended = None
    for logits_np, w, t in zip(logits_list, norm_w, temps):
        probs = F.softmax(torch.from_numpy(logits_np) / max(1e-3, t), dim=1)
        blended = probs * w if blended is None else blended + probs * w
    top5 = blended.topk(5, dim=1).indices.numpy()
    preds = top5[:, 0]
    top1 = float((preds == y_np).mean() * 100.0)
    top3 = float((top5[:, :3] == y_np[:, None]).any(axis=1).mean() * 100.0)
    top5_acc = float((top5 == y_np[:, None]).any(axis=1).mean() * 100.0)
    f1 = compute_macro_f1(preds, y_np, blended.shape[1])
    return {"top1": top1, "top3": top3, "top5": top5_acc, "macro_f1": f1}


def optimize_ensemble_on_val(
    val_outputs: List[Dict[str, Any]],
    val_mir_outputs: List[Dict[str, Any]],
    val_ood_scores: List[np.ndarray],
    val_y: np.ndarray,
) -> Dict[str, Any]:
    """Grid-search ensemble weights, softmax temperatures, and OOD threshold strictly on val."""
    n_models = len(val_outputs)
    if n_models == 1:
        w_best = [1.0]
        t_best = [1.0]
    else:
        best_score = -1.0
        w_best = [0.55, 0.45]
        t_best = [1.0, 1.0]
        val_logits = [o["logits"] for o in val_outputs]
        mir_logits = [o["logits"] for o in val_mir_outputs]
        for w0 in [0.35, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]:
            ws = [w0, 1.0 - w0]
            for t0 in [0.85, 1.0, 1.15]:
                for t1 in [0.85, 1.0, 1.15]:
                    ts = [t0, t1]
                    c = score_blended_probs(val_logits, ws, ts, val_y)
                    m = score_blended_probs(mir_logits, ws, ts, val_y)
                    # Optimize val accuracy + mirror invariance without overfitting
                    obj = 0.70 * c["top1"] + 0.15 * c["macro_f1"] + 0.15 * m["top1"]
                    if obj > best_score:
                        best_score = obj
                        w_best = ws
                        t_best = ts

    val_metrics = score_blended_probs([o["logits"] for o in val_outputs], w_best, t_best, val_y)
    mir_metrics = score_blended_probs([o["logits"] for o in val_mir_outputs], w_best, t_best, val_y)

    # Calibrate weighted OOD score threshold on val vs val_ood
    total_w = sum(w_best)
    pos_ood = sum(val_outputs[i]["ood_prob"] * (w_best[i] / total_w) for i in range(n_models))
    neg_ood = sum(val_ood_scores[i] * (w_best[i] / total_w) for i in range(n_models))
    auroc, fpr95 = compute_roc_auroc_and_fpr95(pos_ood, neg_ood)

    pos_p02 = float(np.quantile(pos_ood, 0.02))
    neg_p99 = float(np.quantile(neg_ood, 0.99))
    # Safe separation threshold preserving >=98% active sign recall while rejecting static/rest
    if pos_p02 > neg_p99:
        ood_thresh = round(0.5 * (pos_p02 + neg_p99), 4)
    else:
        ood_thresh = round(max(0.15, min(0.65, float(np.quantile(pos_ood, 0.03)))), 4)
    ood_thresh = float(max(0.15, min(0.75, ood_thresh)))

    return {
        "weights": w_best,
        "temperatures": t_best,
        "ood_threshold": ood_thresh,
        "val_metrics": val_metrics,
        "val_mirrored_metrics": mir_metrics,
        "val_ood_auroc": auroc,
        "val_ood_fpr95": fpr95,
        "val_pos_ood_p02": pos_p02,
        "val_neg_ood_p99": neg_p99,
    }


def benchmark_predictor_latency(pred: Any, test_x: np.ndarray, n_warmup: int = 10, n_trials: int = 60) -> Dict[str, float]:
    for i in range(n_warmup):
        pred.predict_sequence(test_x[i], is_normalized=False, top_k=5)
    lats = []
    for i in range(n_trials):
        t0 = time.perf_counter()
        pred.predict_sequence(test_x[n_warmup + i], is_normalized=False, top_k=5)
        lats.append((time.perf_counter() - t0) * 1000.0)
    return {
        "mean_ms": round(float(np.mean(lats)), 2),
        "p50_ms": round(float(np.percentile(lats, 50)), 2),
        "p95_ms": round(float(np.percentile(lats, 95)), 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build val-calibrated Ensemble v2 and run full 32,941 test suite")
    parser.add_argument(
        "--exp16",
        type=str,
        default="trained_models/full_2731/exp16_kinematic_conformer/best_model.pth",
    )
    parser.add_argument(
        "--exp17",
        type=str,
        default="trained_models/full_2731/exp17_pyramid_bigru/best_model.pth",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="trained_models/full_2731/ensemble_v2",
    )
    parser.add_argument("--force-promote", action="store_true", help="Promote even if strict gate isn't met (not recommended)")
    args = parser.parse_args()

    ckpts: List[Path] = []
    for p_str in (args.exp16, args.exp17):
        if p_str:
            p = ROOT / p_str if not Path(p_str).is_absolute() else Path(p_str)
            if p.exists():
                ckpts.append(p)

    if not ckpts:
        raise FileNotFoundError("No valid EXP16/EXP17 checkpoints found.")

    device = get_default_device()
    cache_dir = ROOT / "landmark_cache_packed"
    out_dir = ROOT / args.out_dir if not Path(args.out_dir).is_absolute() else Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading validation & test caches on {device} for {len(ckpts)} model(s)...", flush=True)
    val_x_stab = stabilize_split(np.load(cache_dir / "val_x.npy"))
    val_y = np.load(cache_dir / "val_y.npy").astype(np.int64)
    val_mir_stab = mirror_split(val_x_stab)

    val_feat = precompute_features(val_x_stab, device)
    val_mir_feat = precompute_features(val_mir_stab, device)
    val_ood_feat = build_ood_features(val_x_stab, device, num_ood=400, seed=101)

    val_outputs = []
    val_mir_outputs = []
    val_ood_scores = []
    models = []

    for p in ckpts:
        m, cls_names, meta = load_signbridge_model(p, device)
        print(f"Evaluating val split for {p.parent.name} ({meta.get('architecture')})...", flush=True)
        models.append((p, m, meta))
        v_clean = evaluate_features(m, val_feat, val_y, device, return_probs=True)
        v_mir = evaluate_features(m, val_mir_feat, val_y, device, return_probs=True)
        v_ood = eval_ood_scores(m, val_ood_feat, device)
        val_outputs.append(v_clean)
        val_mir_outputs.append(v_mir)
        val_ood_scores.append(v_ood)
        print(
            f"  [VAL {p.parent.name}] top1={v_clean['top1']:.2f}% top5={v_clean['top5']:.2f}% "
            f"f1={v_clean['macro_f1']:.2f}% mir={v_mir['top1']:.2f}%",
            flush=True,
        )

    opt = optimize_ensemble_on_val(val_outputs, val_mir_outputs, val_ood_scores, val_y)
    weights = opt["weights"]
    temps = opt["temperatures"]
    ood_threshold = opt["ood_threshold"]
    print(
        f"\n[VAL ENSEMBLE] weights={[round(w, 3) for w in weights]} temps={temps} "
        f"ood_thresh={ood_threshold:.4f}\n"
        f"  clean: top1={opt['val_metrics']['top1']:.2f}% top3={opt['val_metrics']['top3']:.2f}% "
        f"top5={opt['val_metrics']['top5']:.2f}% f1={opt['val_metrics']['macro_f1']:.2f}%\n"
        f"  mirror: top1={opt['val_mirrored_metrics']['top1']:.2f}% | "
        f"ood_auroc={opt['val_ood_auroc']:.4f} fpr95={opt['val_ood_fpr95']:.4f}",
        flush=True,
    )

    # Now evaluate strictly on ALL 32,941 untouched test.csv samples + full perturbation suite
    print("\nLoading untouched test.csv (32,941 samples) + perturbation splits...", flush=True)
    test_x_raw = np.load(cache_dir / "test_x.npy")
    test_x_stab = stabilize_split(test_x_raw)
    test_y = np.load(cache_dir / "test_y.npy").astype(np.int64)

    test_feat = precompute_features(test_x_stab, device)
    test_mir_2k_feat = precompute_features(mirror_split(test_x_stab[:2000]), device)
    test_offset_2k_feat = precompute_features(perturb_offset_crop(test_x_stab[:2000]), device)
    test_drop_2k_feat = precompute_features(perturb_drop_frames(test_x_stab[:2000], 0.80), device)
    test_ood_feat = build_ood_features(test_x_stab, device, num_ood=400, seed=999)

    test_outputs = []
    test_mir_outputs = []
    test_offset_outputs = []
    test_drop_outputs = []
    test_ood_scores = []
    per_model_test = {}

    for p, m, meta in models:
        t_clean = evaluate_features(m, test_feat, test_y, device, return_probs=True)
        t_mir = evaluate_features(m, test_mir_2k_feat, test_y[:2000], device, return_probs=True)
        t_off = evaluate_features(m, test_offset_2k_feat, test_y[:2000], device, return_probs=True)
        t_drp = evaluate_features(m, test_drop_2k_feat, test_y[:2000], device, return_probs=True)
        t_ood = eval_ood_scores(m, test_ood_feat, device)
        auroc_i, fpr95_i = compute_roc_auroc_and_fpr95(t_clean["ood_prob"], t_ood)

        test_outputs.append(t_clean)
        test_mir_outputs.append(t_mir)
        test_offset_outputs.append(t_off)
        test_drop_outputs.append(t_drp)
        test_ood_scores.append(t_ood)

        per_model_test[p.parent.name] = {
            "architecture": meta.get("architecture"),
            "test_top1": round(t_clean["top1"], 3),
            "test_top3": round(t_clean["top3"], 3),
            "test_top5": round(t_clean["top5"], 3),
            "test_macro_f1": round(t_clean["macro_f1"], 3),
            "mirrored_2k_top1": round(t_mir["top1"], 3),
            "offset_crop_2k_top1": round(t_off["top1"], 3),
            "drop_20pct_2k_top1": round(t_drp["top1"], 3),
            "ood_auroc": round(auroc_i, 5),
            "ood_fpr95": round(fpr95_i, 5),
        }
        print(
            f"[TEST {p.parent.name}] top1={t_clean['top1']:.2f}% top3={t_clean['top3']:.2f}% "
            f"top5={t_clean['top5']:.2f}% f1={t_clean['macro_f1']:.2f}% | "
            f"mir2k={t_mir['top1']:.2f}% off2k={t_off['top1']:.2f}% drop2k={t_drp['top1']:.2f}% auroc={auroc_i:.4f}",
            flush=True,
        )

    ens_clean = score_blended_probs([o["logits"] for o in test_outputs], weights, temps, test_y)
    ens_mir = score_blended_probs([o["logits"] for o in test_mir_outputs], weights, temps, test_y[:2000])
    ens_off = score_blended_probs([o["logits"] for o in test_offset_outputs], weights, temps, test_y[:2000])
    ens_drp = score_blended_probs([o["logits"] for o in test_drop_outputs], weights, temps, test_y[:2000])

    total_w = sum(weights)
    ens_pos_ood = sum(test_outputs[i]["ood_prob"] * (weights[i] / total_w) for i in range(len(models)))
    ens_neg_ood = sum(test_ood_scores[i] * (weights[i] / total_w) for i in range(len(models)))
    ens_auroc, ens_fpr95 = compute_roc_auroc_and_fpr95(ens_pos_ood, ens_neg_ood)

    # Free GPU memory cleanly before benchmarking EnsemblePredictor single-sample latency
    del test_feat, val_feat, val_mir_feat
    if device.type == "mps":
        torch.mps.empty_cache()

    ens_predictors = [
        (SignBridgePredictor(p, ood_threshold=ood_threshold), w)
        for (p, _, _), w in zip(models, weights)
    ]
    live_ens = EnsemblePredictor(ens_predictors, temperatures=temps, ood_threshold=ood_threshold)
    lat_stats = benchmark_predictor_latency(live_ens, test_x_raw)

    # Strict promotion gate vs EXP14c baseline (test_top1=52.70%, test_top5=77.46%, macro_f1=51.87%, mirror=17.60%)
    beats_baseline = (
        ens_clean["top1"] > 52.70
        and ens_clean["top5"] > 77.46
        and ens_clean["macro_f1"] > 51.87
        and ens_mir["top1"] > 45.0
        and ens_auroc >= 0.95
    )
    promoted = bool(beats_baseline or args.force_promote)

    config_payload = {
        "promoted": promoted,
        "version": "ensemble_v2_kinematic_476",
        "ood_threshold": ood_threshold,
        "members": [
            {
                "name": p.parent.name,
                "architecture": meta.get("architecture"),
                "checkpoint": str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p),
                "weight": round(float(w), 4),
                "temperature": round(float(t), 4),
            }
            for (p, _, meta), w, t in zip(models, weights, temps)
        ],
        "val_summary": opt,
        "test_summary": {
            "num_samples": int(len(test_y)),
            "test_top1": round(ens_clean["top1"], 3),
            "test_top3": round(ens_clean["top3"], 3),
            "test_top5": round(ens_clean["top5"], 3),
            "test_macro_f1": round(ens_clean["macro_f1"], 3),
            "mirrored_2k_top1": round(ens_mir["top1"], 3),
            "mirrored_2k_top5": round(ens_mir["top5"], 3),
            "offset_crop_2k_top1": round(ens_off["top1"], 3),
            "drop_20pct_2k_top1": round(ens_drp["top1"], 3),
            "ood_auroc": round(ens_auroc, 5),
            "ood_fpr95": round(ens_fpr95, 5),
            "latency": lat_stats,
        },
        "per_model_test": per_model_test,
    }

    (out_dir / "ensemble_config.json").write_text(json.dumps(config_payload, indent=2), encoding="utf-8")
    (out_dir / "eval_report.json").write_text(json.dumps(config_payload, indent=2), encoding="utf-8")

    print("\n" + "=" * 72)
    print(f"FINAL ENSEMBLE V2 (32,941 TEST SAMPLES) | PROMOTED={promoted}")
    print(
        f"  Top-1={ens_clean['top1']:.2f}% | Top-3={ens_clean['top3']:.2f}% | "
        f"Top-5={ens_clean['top5']:.2f}% | Macro-F1={ens_clean['macro_f1']:.2f}%"
    )
    print(
        f"  Perturbations (2K): Mirrored={ens_mir['top1']:.2f}% | "
        f"OffsetCrop={ens_off['top1']:.2f}% | Drop20%={ens_drp['top1']:.2f}%"
    )
    print(
        f"  OOD Rejection: AUROC={ens_auroc:.5f} | FPR95={ens_fpr95:.5f} | "
        f"Latency={lat_stats['mean_ms']:.2f}ms (p95={lat_stats['p95_ms']:.2f}ms)"
    )
    print("=" * 72, flush=True)


if __name__ == "__main__":
    main()
