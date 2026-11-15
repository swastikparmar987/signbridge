"""Torch-free post-processing helpers shared by PyTorch and ONNX predictors."""

from typing import Any, Dict


def evaluate_no_sign(
    has_ood_head: bool,
    activity: Dict[str, Any],
    ood_prob: float,
    raw_conf: float,
    ood_threshold: float,
) -> bool:
    """Apply OOD gate while preserving high-confidence actively moving strokes."""
    if not has_ood_head:
        return False
    motion_too_low = activity["active_frames"] < 2 or activity["motion_score"] < 0.05
    if motion_too_low:
        return True
    confident_in_motion = (
        activity["active_frames"] >= 5
        and activity["motion_score"] >= 0.09
        and raw_conf >= 62.0
        and ood_prob >= 0.01
    )
    return bool(ood_prob < ood_threshold and not confident_in_motion)
