#!/usr/bin/env python3
"""
Pack per-sample ASL Citizen .npy landmark caches into contiguous arrays under
landmark_cache_packed/ for ultra-fast in-RAM / memory-mapped training and evaluation.
"""

import json
import time
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPLITS_DIR = PROJECT_ROOT / "dataset" / "ASL_Citizen" / "splits"
RAW_CACHE_DIR = PROJECT_ROOT / "landmark_cache_full"
PACKED_DIR = PROJECT_ROOT / "landmark_cache_packed"
PROD_CLASSES = PROJECT_ROOT / "trained_models" / "production" / "class_names.json"


def pack_split(split_csv_name: str, cache_subdir: str, out_prefix: str, gloss_to_idx: dict) -> None:
    df = pd.read_csv(SPLITS_DIR / split_csv_name)
    n = len(df)
    src_dir = RAW_CACHE_DIR / cache_subdir

    t0 = time.perf_counter()
    x_arr = np.empty((n, 32, 42, 3), dtype=np.float32)
    y_arr = np.empty((n,), dtype=np.int64)
    signers: List[str] = df["Participant ID"].astype(str).tolist()
    unique_signers = sorted(set(signers))
    signer_to_idx = {s: i for i, s in enumerate(unique_signers)}
    s_arr = np.array([signer_to_idx[s] for s in signers], dtype=np.int32)

    empty_count = 0
    for i, gloss in enumerate(df["Gloss"]):
        p = src_dir / f"{i}.npy"
        arr = np.load(p).astype(np.float32)
        if arr.ndim == 2 and arr.shape[1] == 126:
            arr = arr.reshape(32, 42, 3)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        if np.all(arr == 0.0):
            empty_count += 1
        x_arr[i] = arr
        y_arr[i] = gloss_to_idx[gloss]

    np.save(PACKED_DIR / f"{out_prefix}_x.npy", x_arr)
    np.save(PACKED_DIR / f"{out_prefix}_y.npy", y_arr)
    np.save(PACKED_DIR / f"{out_prefix}_signer.npy", s_arr)
    dt = time.perf_counter() - t0
    print(
        f"Packed {out_prefix:5s}: shape={x_arr.shape}, classes={len(np.unique(y_arr))}, "
        f"signers={len(unique_signers)}, all_zero={empty_count} in {dt:.2f}s"
    )


def main() -> None:
    PACKED_DIR.mkdir(parents=True, exist_ok=True)
    test_df = pd.read_csv(SPLITS_DIR / "test.csv")
    class_names = sorted(test_df["Gloss"].unique().tolist())
    assert len(class_names) == 2731, f"Expected 2731 classes, got {len(class_names)}"

    if PROD_CLASSES.exists():
        prod = json.loads(PROD_CLASSES.read_text(encoding="utf-8"))
        assert prod == class_names, "Class names ordering mismatch against production class_names.json!"

    (PACKED_DIR / "class_names.json").write_text(json.dumps(class_names, indent=2), encoding="utf-8")
    gloss_to_idx = {g: i for i, g in enumerate(class_names)}

    pack_split("train.csv", "train", "train", gloss_to_idx)
    pack_split("val.csv", "validation", "val", gloss_to_idx)
    pack_split("test.csv", "test", "test", gloss_to_idx)
    print(f"Done. Saved contiguous caches under {PACKED_DIR}")


if __name__ == "__main__":
    main()
