#!/usr/bin/env python3
"""Build a compact 2,731-gloss 3D hand animation exemplar & step-guide cache.

Selects the cleanest high-visibility 3D MediaPipe hand landmark sequence per gloss
from landmark_cache_packed, trims empty padding, repairs 1-2 frame dropouts,
smoothly resamples to 32 active frames, and derives anatomical signing steps.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from signbridge.continuous.segmenter import resample_sequence_aware

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKED_DIR = PROJECT_ROOT / "landmark_cache_packed"
DEFAULT_OUT = PROJECT_ROOT / "data" / "dictionary_exemplars_32.npz"

FINGER_CHAINS = {
    "Thumb": (1, 2, 3, 4),
    "Index": (5, 6, 7, 8),
    "Middle": (9, 10, 11, 12),
    "Ring": (13, 14, 15, 16),
    "Pinky": (17, 18, 19, 20),
}


def _active_mask(hand_seq: np.ndarray) -> np.ndarray:
    return np.abs(hand_seq).sum(axis=(-2, -1)) > 1e-4


def _score_candidate(seq: np.ndarray) -> float:
    """Rank exemplar candidates by visibility continuity and realistic motion."""
    m0 = _active_mask(seq[:, :21, :])
    m1 = _active_mask(seq[:, 21:42, :])
    any_m = m0 | m1
    active_count = int(any_m.sum())
    if active_count < 3:
        return -1e6

    # Prefer rich sequences without extreme dropout holes inside the active span
    nz = np.where(any_m)[0]
    span = int(nz[-1] - nz[0] + 1)
    hole_penalty = float((~any_m[nz[0] : nz[-1] + 1]).sum()) * 2.5

    # Penalize single-frame hand-slot flip or wrist teleportation > 0.22
    jump_penalty = 0.0
    motion_reward = 0.0
    for sl in (slice(0, 21), slice(21, 42)):
        h = seq[:, sl, :]
        m = _active_mask(h)
        if m.sum() >= 3:
            pts = h[m, 0, :2]
            diffs = np.linalg.norm(np.diff(pts, axis=0), axis=-1)
            jump_penalty += float((diffs > 0.22).sum()) * 3.0
            motion_reward += float(np.clip(diffs.sum(), 0.0, 1.2))
            motion_reward += float(m.mean()) * 1.5

    span_bonus = min(active_count, 26) * 0.25 + min(span, 28) * 0.05
    return span_bonus + motion_reward - hole_penalty - jump_penalty


def _repair_and_smooth_hand(hand_seq: np.ndarray) -> np.ndarray:
    """Fill short internal dropouts and apply mild binomial smoothing."""
    out = hand_seq.copy()
    mask = _active_mask(out)
    n = len(out)
    if mask.sum() < 2:
        return out

    idx = np.where(mask)[0]
    first, last = int(idx[0]), int(idx[-1])
    # Linearly bridge gaps up to 4 frames within active span
    i = first
    while i < last:
        if mask[i] and not mask[i + 1]:
            j = i + 1
            while j <= last and not mask[j]:
                j += 1
            if j <= last and (j - i) <= 5:
                for k in range(i + 1, j):
                    a = float(k - i) / float(j - i)
                    out[k] = (1.0 - a) * out[i] + a * out[j]
            i = j
        else:
            i += 1

    # Mild temporal smoothing [0.2, 0.6, 0.2] on continuously active triples
    mask2 = _active_mask(out)
    smoothed = out.copy()
    for t in range(1, n - 1):
        if mask2[t - 1] and mask2[t] and mask2[t + 1]:
            smoothed[t] = 0.2 * out[t - 1] + 0.6 * out[t] + 0.2 * out[t + 1]
    return smoothed


def _clean_exemplar(raw: np.ndarray) -> np.ndarray:
    seq = np.asarray(raw, dtype=np.float32)
    any_m = _active_mask(seq[:, :21, :]) | _active_mask(seq[:, 21:42, :])
    nz = np.where(any_m)[0]
    if len(nz) == 0:
        return np.zeros((32, 42, 3), dtype=np.float32)

    # Keep all active core frames so the 32-frame timeline continuously animates
    trimmed = seq[int(nz[0]) : int(nz[-1]) + 1].copy()
    trimmed[:, :21, :] = _repair_and_smooth_hand(trimmed[:, :21, :])
    trimmed[:, 21:42, :] = _repair_and_smooth_hand(trimmed[:, 21:42, :])

    res32 = resample_sequence_aware(trimmed, target_len=32)
    # One final gentle pass on active frames to keep 3D playback silky smooth
    res32[:, :21, :] = _repair_and_smooth_hand(res32[:, :21, :])
    res32[:, 21:42, :] = _repair_and_smooth_hand(res32[:, 21:42, :])
    return res32.astype(np.float32)


def _finger_extension_ratios(hand: np.ndarray) -> Dict[str, float]:
    """Return 0..1 extension score per finger from tip distance vs palm scale."""
    palm_scale = max(float(np.linalg.norm(hand[9] - hand[0])), 0.03)
    ratios: Dict[str, float] = {}
    for name, (cmc, mcp, pip, tip) in FINGER_CHAINS.items():
        reach = float(np.linalg.norm(hand[tip] - hand[mcp if name != "Thumb" else cmc]))
        ext = float(np.linalg.norm(hand[tip] - hand[0])) / palm_scale
        ratios[name] = float(np.clip((reach / palm_scale + ext * 0.5) / 1.45, 0.0, 1.0))
    return ratios


def _describe_sign(gloss: str, seq32: np.ndarray) -> Tuple[bool, str, str, List[str]]:
    """Derive hand count, dominant handshape label, and 3 concrete signing cues."""
    m0 = _active_mask(seq32[:, :21, :])
    m1 = _active_mask(seq32[:, 21:42, :])
    two_hand = bool(m0.mean() >= 0.45 and m1.mean() >= 0.45)

    primary_slice = slice(0, 21) if m0.sum() >= m1.sum() else slice(21, 42)
    primary = seq32[:, primary_slice, :]
    pmask = _active_mask(primary)
    active_frames = primary[pmask] if pmask.any() else seq32[:, :21, :]

    mid_frame = active_frames[len(active_frames) // 2]
    start_frame = active_frames[0]
    end_frame = active_frames[-1]

    ratios = _finger_extension_ratios(mid_frame)
    ext_names = [k for k, v in ratios.items() if k != "Thumb" and v >= 0.62]
    if len(ext_names) >= 4:
        handshape = "Open 5-Palm" if ratios["Thumb"] >= 0.55 else "Flat B-Palm"
    elif ext_names == ["Index"]:
        handshape = "Pointing Index (1 / G)"
    elif set(ext_names) == {"Index", "Middle"}:
        handshape = "Two-Finger (V / U / H)"
    elif len(ext_names) == 0:
        handshape = "Compact S-Fist"
    elif len(ext_names) == 3 and "Pinky" not in ext_names:
        handshape = "Three-Finger Open"
    else:
        handshape = f"{'/'.join(ext_names)} Extended" if ext_names else "Curved Handshape"

    # Wrist location zone from normalized image y (0 top -> 1 bottom)
    wy = float(start_frame[0, 1])
    height_zone = "head or forehead level" if wy < 0.45 else ("chin / upper chest height" if wy < 0.68 else "neutral mid-torso signing space")

    # Motion displacement vector from start to peak excursion
    wrists = active_frames[:, 0, :]
    disp = wrists - wrists[0:1]
    peak_idx = int(np.argmax(np.linalg.norm(disp, axis=-1)))
    dx = float(wrists[peak_idx, 0] - wrists[0, 0])
    dy = float(wrists[peak_idx, 1] - wrists[0, 1])
    total_path = float(np.linalg.norm(np.diff(wrists[:, :2], axis=0), axis=-1).sum()) if len(wrists) > 1 else 0.0
    end_return = float(np.linalg.norm(end_frame[0, :2] - start_frame[0, :2]))

    dirs: List[str] = []
    if abs(dy) > 0.045:
        dirs.append("upward" if dy < 0 else "downward")
    if abs(dx) > 0.045:
        dirs.append("laterally across signing space" if two_hand else "outward / side-to-side")
    if not dirs:
        motion_phrase = (
            "articulate the fingers in place with a light tap or pulse"
            if total_path < 0.14
            else "draw a smooth curved arc in front of your torso"
        )
    elif total_path > 0.25 and end_return < 0.06:
        motion_phrase = f"move {' and '.join(dirs)}, then cycle smoothly back toward start"
    else:
        motion_phrase = f"sweep steadily {' and '.join(dirs)} along a crisp path"

    # Palm facing estimate from index_mcp -> pinky_mcp cross wrist -> middle_mcp
    palm_cross = np.cross(mid_frame[5] - mid_frame[0], mid_frame[17] - mid_frame[0])
    palm_cue = "palm oriented inward/angled" if abs(palm_cross[2]) < 0.004 else "palm facing forward/side"

    hand_count_str = "both hands working together" if two_hand else "your dominant hand"
    article = "an" if handshape[0].lower() in "aeiou" else "a"
    steps = [
        f"Form {article} {handshape} with {hand_count_str} at {height_zone} ({palm_cue}).",
        f"Keeping fingers poised, {motion_phrase} (watch frames 8–24 along the golden arc).",
        f"Pause cleanly at the hold position on frames 26–32 to finish {gloss}.",
    ]
    return two_hand, handshape, motion_phrase, steps


def build_cache(out_path: Path, splits: Tuple[str, ...] = ("val", "test", "train")) -> Dict[str, Any]:
    names: List[str] = json.loads((PACKED_DIR / "class_names.json").read_text(encoding="utf-8"))
    num_classes = len(names)

    by_class: Dict[int, List[Tuple[np.ndarray, np.ndarray]]] = {i: [] for i in range(num_classes)}
    loaded_splits = []
    for sp in splits:
        x_p = PACKED_DIR / f"{sp}_x.npy"
        y_p = PACKED_DIR / f"{sp}_y.npy"
        if x_p.exists() and y_p.exists():
            loaded_splits.append((sp, np.load(x_p, mmap_mode="r"), np.load(y_p, mmap_mode="r")))

    # Pre-index first 10 candidate row indices per class across splits for O(1) pass
    split_indices: List[Tuple[str, np.ndarray, np.ndarray]] = loaded_splits
    class_best_score = np.full(num_classes, -1e9, dtype=np.float32)
    class_best_raw = np.zeros((num_classes, 32, 42, 3), dtype=np.float32)

    for sp_name, x_arr, y_arr in split_indices:
        # Sort labels once via argsort for fast group slicing
        y_np = np.asarray(y_arr, dtype=np.int64)
        order = np.argsort(y_np, kind="mergesort")
        y_sorted = y_np[order]
        cuts = np.flatnonzero(np.diff(y_sorted)) + 1
        starts = np.concatenate(([0], cuts))
        ends = np.concatenate((cuts, [len(y_sorted)]))

        sp_bonus = 0.25 if sp_name in ("test", "val") else 0.0
        for st, en in zip(starts, ends):
            cid = int(y_sorted[st])
            if class_best_score[cid] > 7.5 and sp_name == "train":
                continue
            sample_ids = order[st : min(en, st + 8)]
            for idx in sample_ids:
                seq = np.asarray(x_arr[int(idx)], dtype=np.float32)
                sc = _score_candidate(seq) + sp_bonus
                if sc > class_best_score[cid]:
                    class_best_score[cid] = sc
                    class_best_raw[cid] = seq

    frames_out = np.zeros((num_classes, 32, 42, 3), dtype=np.float16)
    two_hand_out = np.zeros(num_classes, dtype=bool)
    handshapes: List[str] = []
    steps_list: List[List[str]] = []

    for cid, gloss in enumerate(names):
        cleaned = _clean_exemplar(class_best_raw[cid])
        two_h, hs, _, steps = _describe_sign(gloss, cleaned)
        frames_out[cid] = cleaned.astype(np.float16)
        two_hand_out[cid] = two_h
        handshapes.append(hs)
        steps_list.append(steps)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        class_names=np.array(names, dtype=object),
        frames=frames_out,
        two_hand=two_hand_out,
        handshapes=np.array(handshapes, dtype=object),
        steps_json=np.array([json.dumps(s) for s in steps_list], dtype=object),
    )
    return {
        "glosses": num_classes,
        "two_hand_count": int(two_hand_out.sum()),
        "one_hand_count": int(num_classes - int(two_hand_out.sum())),
        "output_mb": round(out_path.stat().st_size / (1024 * 1024), 2),
        "path": str(out_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build 3D dictionary exemplar cache for 2,731 glosses.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    summary = build_cache(args.out)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
