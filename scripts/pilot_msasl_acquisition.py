#!/usr/bin/env python3
"""
SignBridge MS-ASL Pilot Video Acquisition & Preprocessing Pipeline

Performs:
1. Directory setup under dataset/MS_ASL/ (raw, clips, landmarks, metadata, logs).
2. Random selection of 20 diverse MS-ASL training samples across distinct classes and signers.
3. Segment extraction and download via yt-dlp & ffmpeg.
4. MediaPipe Hands extraction using the existing SignBridge preprocessing implementation.
5. Verification of (32, 126) tensor contract and forward pass through production GRU architectures.
6. Manifest generation:
   - dataset/MS_ASL/metadata/download_manifest.csv
   - dataset/MS_ASL/metadata/preprocessing_manifest.csv
"""

import argparse
import json
import logging
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import yt_dlp

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from signbridge.preprocessing.landmarks import extract_hand_sequence
from signbridge.preprocessing.normalize import normalize_sequence
from signbridge.inference.model import GRUClassifier, BiGRUAttentionPoolingClassifier


def setup_logger(log_file: Path) -> logging.Logger:
    logger = logging.getLogger("MSASL_Pilot")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh_fmt = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
    fh.setFormatter(fh_fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch_fmt = logging.Formatter("%(message)s")
    ch.setFormatter(ch_fmt)
    logger.addHandler(ch)

    return logger


def clean_url(url: str) -> str:
    """Normalize YouTube URL."""
    url = url.strip()
    if not url.startswith("http://") and not url.startswith("https://"):
        url = "https://" + url
    return url


def select_pilot_samples(
    matching_entries: List[Dict[str, Any]],
    label_to_asl: Dict[int, str],
    num_samples: int = 20,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Select num_samples with distinct classes and distinct signers."""
    random.seed(seed)
    shuffled = matching_entries.copy()
    random.shuffle(shuffled)

    selected = []
    seen_classes = set()
    seen_signers = set()

    for e in shuffled:
        lbl = e["label"]
        sid = e["signer_id"]
        if lbl not in seen_classes and sid not in seen_signers:
            entry_copy = dict(e)
            entry_copy["asl_citizen_class"] = label_to_asl[lbl]
            selected.append(entry_copy)
            seen_classes.add(lbl)
            seen_signers.add(sid)
            if len(selected) == num_samples:
                break

    # If could not find 20 completely disjoint signers, relax signer constraint
    if len(selected) < num_samples:
        for e in shuffled:
            lbl = e["label"]
            if lbl not in seen_classes:
                entry_copy = dict(e)
                entry_copy["asl_citizen_class"] = label_to_asl[lbl]
                selected.append(entry_copy)
                seen_classes.add(lbl)
                if len(selected) == num_samples:
                    break

    return selected


def download_and_clip(
    url: str,
    start_time: float,
    end_time: float,
    clip_output_path: Path,
    raw_dir: Path,
    logger: logging.Logger,
    timeout_sec: int = 45,
) -> Tuple[str, str, Optional[str]]:
    """
    Download and trim clip.
    Returns (download_status, clip_status, failure_reason).
    """
    clean_video_url = clean_url(url)
    clip_output_path.parent.mkdir(parents=True, exist_ok=True)

    # Strategy 1: Direct section download with yt-dlp
    def section_range(info_dict, ydl):
        return [{"start_time": max(0.0, start_time), "end_time": max(start_time + 0.5, end_time)}]

    temp_template = str(clip_output_path.parent / f"temp_{clip_output_path.stem}.%(ext)s")

    ydl_opts = {
        "format": "bestvideo[ext=mp4][height<=720]+bestaudio[ext=m4a]/best[ext=mp4][height<=720]/best",
        "outtmpl": temp_template,
        "download_ranges": section_range,
        "force_keyframes_at_cuts": True,
        "quiet": True,
        "no_warnings": True,
        "socket_timeout": 20,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([clean_video_url])

        # Find the produced file
        matching_files = list(clip_output_path.parent.glob(f"temp_{clip_output_path.stem}.*"))
        if matching_files and matching_files[0].exists() and matching_files[0].stat().st_size > 1024:
            temp_file = matching_files[0]
            # Ensure proper container via fast ffmpeg remux
            cmd = [
                "ffmpeg", "-y", "-i", str(temp_file),
                "-c:v", "libx264", "-preset", "ultrafast", "-crf", "22",
                "-c:a", "aac",
                str(clip_output_path)
            ]
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout_sec)
            temp_file.unlink(missing_ok=True)
            if res.returncode == 0 and clip_output_path.exists() and clip_output_path.stat().st_size > 1024:
                return "success", "success", None
    except yt_dlp.utils.DownloadError as e:
        err_msg = str(e).strip()
        logger.warning(f"yt-dlp DownloadError on {clean_video_url}: {err_msg[:120]}")
        if "Private video" in err_msg or "unavailable" in err_msg or "deleted" in err_msg or "removed" in err_msg:
            return "unavailable", "skipped", err_msg[:200]
        # Otherwise fallback to Strategy 2 below
    except Exception as e:
        logger.warning(f"Strategy 1 failed: {e}. Trying fallback.")

    # Strategy 2: Fallback - download raw video with yt-dlp then cut with ffmpeg
    try:
        raw_template = str(raw_dir / "%(id)s.%(ext)s")
        raw_opts = {
            "format": "bestvideo[ext=mp4][height<=720]+bestaudio[ext=m4a]/best[ext=mp4][height<=720]/best",
            "outtmpl": raw_template,
            "quiet": True,
            "no_warnings": True,
            "socket_timeout": 25,
        }
        with yt_dlp.YoutubeDL(raw_opts) as ydl:
            info = ydl.extract_info(clean_video_url, download=True)
            video_id = info.get("id")
            ext = info.get("ext", "mp4")

        raw_video_path = raw_dir / f"{video_id}.{ext}"
        if not raw_video_path.exists():
            # Search matching video_id in raw_dir
            matches = list(raw_dir.glob(f"{video_id}.*"))
            if matches:
                raw_video_path = matches[0]

        if raw_video_path.exists() and raw_video_path.stat().st_size > 1024:
            duration = max(0.5, end_time - start_time)
            cut_cmd = [
                "ffmpeg", "-y",
                "-ss", str(start_time),
                "-t", str(duration),
                "-i", str(raw_video_path),
                "-c:v", "libx264", "-preset", "ultrafast", "-crf", "22",
                "-c:a", "aac",
                str(clip_output_path)
            ]
            res = subprocess.run(cut_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout_sec)
            if res.returncode == 0 and clip_output_path.exists() and clip_output_path.stat().st_size > 1024:
                return "success", "success", None
            else:
                return "success", "failed", f"ffmpeg clip failed: {res.stderr.decode()[:200]}"
    except yt_dlp.utils.DownloadError as e:
        err_msg = str(e).strip()
        if "Private video" in err_msg or "unavailable" in err_msg or "deleted" in err_msg or "removed" in err_msg:
            return "unavailable", "skipped", err_msg[:200]
        return "download_error", "failed", err_msg[:200]
    except Exception as e:
        return "download_error", "failed", str(e)[:200]

    return "download_error", "failed", "Unknown download/clip error"


def run_pilot():
    parser = argparse.ArgumentParser(description="Run MS-ASL Video Acquisition & Preprocessing Pilot")
    parser.add_argument("--num-samples", type=int, default=20, help="Number of pilot samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    ms_asl_root = PROJECT_ROOT / "dataset" / "MS_ASL"
    raw_dir = ms_asl_root / "raw"
    clips_dir = ms_asl_root / "clips"
    landmarks_dir = ms_asl_root / "landmarks"
    metadata_dir = ms_asl_root / "metadata"
    logs_dir = ms_asl_root / "logs"

    for d in [raw_dir, clips_dir, landmarks_dir, metadata_dir, logs_dir]:
        d.mkdir(parents=True, exist_ok=True)

    log_file = logs_dir / "pilot_pipeline.log"
    logger = setup_logger(log_file)

    logger.info("==================================================================")
    logger.info("SignBridge MS-ASL Pilot Video Acquisition & Preprocessing (20 Clips)")
    logger.info("==================================================================")
    logger.info(f"Target directory: {ms_asl_root}")

    # Load matching metadata
    matching_json_path = PROJECT_ROOT / "msasl_matching_train.json"
    overlap_csv_path = PROJECT_ROOT / "msasl_signbridge_overlap.csv"

    if not matching_json_path.exists() or not overlap_csv_path.exists():
        logger.error("Required overlap analysis files not found. Run scripts/analyze_msasl_overlap.py first.")
        sys.exit(1)

    with open(matching_json_path, "r", encoding="utf-8") as f:
        matching_train = json.load(f)

    overlap_df = pd.read_csv(overlap_csv_path)
    label_to_asl = dict(zip(overlap_df["msasl_label"], overlap_df["asl_citizen_class"]))

    logger.info(f"Total available matching train entries: {len(matching_train)}")
    pilot_samples = select_pilot_samples(
        matching_train,
        label_to_asl,
        num_samples=args.num_samples,
        seed=args.seed,
    )
    logger.info(f"Selected {len(pilot_samples)} pilot entries across distinct classes and signers.\n")

    # Save pilot candidate metadata
    with open(metadata_dir / "pilot_candidates.json", "w", encoding="utf-8") as f:
        json.dump(pilot_samples, f, indent=2)

    download_records = []
    preprocessing_records = []

    # Model for compatibility check
    gru_model = GRUClassifier(126, 256, 2, 2731)
    bigru_model = BiGRUAttentionPoolingClassifier(126, 256, 2, 2731)
    gru_model.eval()
    bigru_model.eval()

    logger.info("--- Phase 1: Video Downloading & Clipping ---")
    for idx, sample in enumerate(pilot_samples, start=1):
        ms_cls = sample.get("clean_text", sample.get("text", ""))
        asl_cls = sample["asl_citizen_class"]
        lbl = sample["label"]
        sid = sample["signer_id"]
        url = sample["url"]
        st = float(sample.get("start_time", 0.0))
        et = float(sample.get("end_time", st + 2.0))

        clip_filename = f"clip_{idx:02d}_{asl_cls}_label{lbl}_signer{sid}.mp4"
        clip_path = clips_dir / clip_filename

        logger.info(f"[{idx:02d}/{len(pilot_samples):02d}] Downloading: '{asl_cls}' (MS-ASL: '{ms_cls}', label {lbl}, signer {sid})")
        logger.info(f"     URL: {url} [{st:.2f}s -> {et:.2f}s]")

        dl_status, clip_status, fail_reason = download_and_clip(
            url=url,
            start_time=st,
            end_time=et,
            clip_output_path=clip_path,
            raw_dir=raw_dir,
            logger=logger,
        )

        logger.info(f"     Status: Download={dl_status}, Clip={clip_status}" + (f" (Reason: {fail_reason})" if fail_reason else ""))

        download_records.append({
            "msasl_class": ms_cls,
            "asl_citizen_class": asl_cls,
            "msasl_label": lbl,
            "signer_id": sid,
            "youtube_url": url,
            "start_time": st,
            "end_time": et,
            "download_status": dl_status,
            "clip_status": clip_status,
            "clip_file": clip_filename if clip_status == "success" else "",
            "failure_reason": fail_reason or "",
        })

    # Save download manifest
    download_df = pd.DataFrame(download_records)
    dl_manifest_path = metadata_dir / "download_manifest.csv"
    download_df.to_csv(dl_manifest_path, index=False)
    logger.info(f"\nDownload manifest saved to: {dl_manifest_path}")

    logger.info("\n--- Phase 2: SignBridge MediaPipe Preprocessing & Tensor Validation ---")
    for idx, dl_rec in enumerate(download_records, start=1):
        ms_cls = dl_rec["msasl_class"]
        asl_cls = dl_rec["asl_citizen_class"]
        lbl = dl_rec["msasl_label"]
        sid = dl_rec["signer_id"]
        clip_file = dl_rec["clip_file"]

        if dl_rec["clip_status"] != "success" or not clip_file:
            preprocessing_records.append({
                "msasl_class": ms_cls,
                "asl_citizen_class": asl_cls,
                "msasl_label": lbl,
                "signer_id": sid,
                "clip_file": "",
                "landmark_file": "",
                "preprocessing_status": "skipped",
                "detected_hand_frames": 0,
                "final_landmark_shape": "",
                "is_pipeline_compatible": False,
                "failure_reason": f"Download/clipping skipped: {dl_rec['failure_reason']}",
            })
            continue

        clip_path = clips_dir / clip_file
        landmark_filename = clip_file.replace(".mp4", "_landmarks.npy")
        landmark_path = landmarks_dir / landmark_filename

        try:
            # 1. Extract raw landmarks via existing SignBridge implementation
            raw_seq = extract_hand_sequence(clip_path, num_frames=32)

            # Count frames with non-zero hand detections
            detected_frames = sum(1 for f in raw_seq if not np.allclose(f, 0))

            # 2. Normalize via verified SignBridge normalize_sequence pipeline
            norm_seq = normalize_sequence(raw_seq)

            # 3. Reshape to GRU contract (32, 126)
            features = norm_seq.reshape(32, 126).astype(np.float32)

            # 4. Save processed landmarks to MS_ASL/landmarks/
            np.save(landmark_path, features)

            # 5. Verify tensor contract & model compatibility
            tensor_input = torch.from_numpy(features).unsqueeze(0) # [1, 32, 126]
            with torch.no_grad():
                logits1 = gru_model(tensor_input)
                logits2 = bigru_model(tensor_input)

            compatible = (
                features.shape == (32, 126)
                and logits1.shape == (1, 2731)
                and logits2.shape == (1, 2731)
            )

            prep_status = "success" if detected_frames > 0 else "no_hands_detected"

            logger.info(f"[{idx:02d}] Preprocessed '{clip_file}' -> Hand frames: {detected_frames}/32, Shape: {features.shape}, Pipeline Compatible: {compatible}")

            preprocessing_records.append({
                "msasl_class": ms_cls,
                "asl_citizen_class": asl_cls,
                "msasl_label": lbl,
                "signer_id": sid,
                "clip_file": clip_file,
                "landmark_file": landmark_filename,
                "preprocessing_status": prep_status,
                "detected_hand_frames": detected_frames,
                "final_landmark_shape": str(features.shape),
                "is_pipeline_compatible": compatible,
                "failure_reason": "" if prep_status == "success" else "Low/zero hand detection in video frames",
            })

        except Exception as e:
            logger.error(f"[{idx:02d}] Preprocessing failed for '{clip_file}': {e}")
            preprocessing_records.append({
                "msasl_class": ms_cls,
                "asl_citizen_class": asl_cls,
                "msasl_label": lbl,
                "signer_id": sid,
                "clip_file": clip_file,
                "landmark_file": "",
                "preprocessing_status": "error",
                "detected_hand_frames": 0,
                "final_landmark_shape": "",
                "is_pipeline_compatible": False,
                "failure_reason": str(e)[:200],
            })

    # Save preprocessing manifest
    prep_df = pd.DataFrame(preprocessing_records)
    prep_manifest_path = metadata_dir / "preprocessing_manifest.csv"
    prep_df.to_csv(prep_manifest_path, index=False)
    logger.info(f"\nPreprocessing manifest saved to: {prep_manifest_path}")

    # Summary Statistics
    total_samples = len(pilot_samples)
    successful_dl = sum(1 for r in download_records if r["download_status"] == "success")
    unavailable_dl = sum(1 for r in download_records if r["download_status"] == "unavailable")
    error_dl = sum(1 for r in download_records if r["download_status"] == "download_error")
    clipped = sum(1 for r in download_records if r["clip_status"] == "success")

    successful_prep = sum(1 for r in preprocessing_records if r["preprocessing_status"] == "success")
    detection_failures = sum(1 for r in preprocessing_records if r["preprocessing_status"] == "no_hands_detected")
    prep_errors = sum(1 for r in preprocessing_records if r["preprocessing_status"] == "error")
    compatible_count = sum(1 for r in preprocessing_records if r["is_pipeline_compatible"])

    logger.info("\n==================================================================")
    logger.info("MS-ASL Pilot Evaluation Summary")
    logger.info("==================================================================")
    logger.info(f"1. Videos targeted:                {total_samples}")
    logger.info(f"2. Videos successfully downloaded: {successful_dl} ({successful_dl/total_samples*100:.1f}%)")
    logger.info(f"3. Videos unavailable / deleted:   {unavailable_dl} ({unavailable_dl/total_samples*100:.1f}%)")
    logger.info(f"4. Download errors:                {error_dl} ({error_dl/total_samples*100:.1f}%)")
    logger.info(f"5. Videos successfully clipped:    {clipped} ({clipped/total_samples*100:.1f}%)")
    logger.info(f"6. Videos successfully processed:  {successful_prep} ({successful_prep/total_samples*100:.1f}%)")
    logger.info(f"7. MediaPipe detection failures:   {detection_failures}")
    logger.info(f"8. Pipeline compatible tensors:    {compatible_count} / {clipped}")
    logger.info(f"9. All successful tensors shape:   (32, 126) float32")
    logger.info("==================================================================\n")


if __name__ == "__main__":
    run_pilot()
