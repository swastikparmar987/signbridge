from typing import Dict, Any, List, Tuple
import numpy as np
import torch
from sklearn.metrics import f1_score, confusion_matrix


def compute_topk_accuracies(
    logits: torch.Tensor,
    targets: torch.Tensor,
    topk: Tuple[int, ...] = (1, 3, 5),
) -> Dict[str, float]:
    """
    Compute Top-K accuracies for given logits and targets.
    
    Args:
        logits: (N, C) tensor
        targets: (N,) tensor
        topk: tuple of k values
        
    Returns:
        Dict mapping 'top1', 'top3', 'top5' to float percentages (0-100)
    """
    maxk = max(topk)
    batch_size = targets.size(0)
    if batch_size == 0:
        return {f"top{k}": 0.0 for k in topk}

    _, pred = logits.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1).expand_as(pred))

    res = {}
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True).item()
        res[f"top{k}"] = float((correct_k / batch_size) * 100.0)

    return res


def compute_comprehensive_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    class_names: List[str],
) -> Dict[str, Any]:
    """
    Compute full evaluation metrics:
    - Top-1, Top-3, Top-5 Accuracy
    - Macro F1, Weighted F1
    - Per-class accuracy
    - Top confused pairs
    - Full confusion matrix
    """
    topk_res = compute_topk_accuracies(logits, targets, topk=(1, 3, 5))
    preds = logits.argmax(dim=-1).cpu().numpy()
    y_true = targets.cpu().numpy()
    num_classes = len(class_names)

    macro_f1 = float(f1_score(y_true, preds, average="macro", zero_division=0) * 100.0)
    weighted_f1 = float(f1_score(y_true, preds, average="weighted", zero_division=0) * 100.0)

    # Confusion matrix
    cm = confusion_matrix(y_true, preds, labels=list(range(num_classes)))

    # Per-class accuracy
    per_class_acc: Dict[str, float] = {}
    for idx, name in enumerate(class_names):
        total_samples = int(cm[idx].sum())
        correct_samples = int(cm[idx, idx])
        acc = (correct_samples / total_samples * 100.0) if total_samples > 0 else 0.0
        per_class_acc[name] = float(acc)

    # Sort classes by accuracy
    sorted_classes = sorted(per_class_acc.items(), key=lambda x: x[1])
    worst_classes = sorted_classes[:10]
    best_classes = sorted_classes[-10:][::-1]

    # Find top confused pairs (where true != pred and count > 0)
    confusions = []
    for true_idx in range(num_classes):
        for pred_idx in range(num_classes):
            if true_idx != pred_idx and cm[true_idx, pred_idx] > 0:
                confusions.append({
                    "true_class": class_names[true_idx],
                    "predicted_class": class_names[pred_idx],
                    "count": int(cm[true_idx, pred_idx]),
                })
    confusions.sort(key=lambda x: x["count"], reverse=True)

    return {
        "top1_accuracy": topk_res["top1"],
        "top3_accuracy": topk_res["top3"],
        "top5_accuracy": topk_res["top5"],
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "total_samples": int(len(y_true)),
        "per_class_accuracy": per_class_acc,
        "best_classes": best_classes,
        "worst_classes": worst_classes,
        "top_confusions": confusions[:15],
        "confusion_matrix": cm.tolist(),
    }
