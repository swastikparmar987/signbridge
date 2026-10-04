#!/usr/bin/env python3
"""
SignBridge MS-ASL Full-Scale Video Acquisition & Preprocessing Pipeline (Robust Edition)

Scales acquisition across MS-ASL TRAIN entries with enhanced resilience:
- Resumes seamlessly and strictly preserves existing successful samples and tensors.
- Re-evaluates recoverable failures (bot challenges, rate limits, transient network errors).
- URL-level caching: skips confirmed deleted/private/unavailable videos once.
- Efficient multi-clip cutting: downloads raw video once for multi-clip YouTube videos.
- Exponential backoff retries and configurable request delays to avoid API rate limits.
- Supports browser-cookie authentication via local environment or CLI options.
- Enforces strict (32, 126) float32 MediaPipe landmark tensor validation.
- Atomic manifest writing for crash resilience.
"""

import argparse
import csv
import json
import logging
import os
import re
import signal
import subprocess
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yt_dlp

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from signbridge.preprocessing.landmarks import extract_hand_sequence
from signbridge.preprocessing.normalize import normalize_sequence
from signbridge.inference.model import GRUClassifier, BiGRUAttentionPoolingClassifier

# Global flag for graceful shutdown
SHUTDOWN_REQUESTED = False


def sigint_handler(signum, frame):
    global SHUTDOWN_REQUESTED
    SHUTDOWN_REQUESTED = True
    print("\n[SIGINT / SIGTERM] Received shutdown signal. Finishing active items and saving manifest state...")


signal.signal(signal.SIGINT, sigint_handler)
signal.signal(signal.SIGTERM, sigint_handler)


def setup_logger(log_file: Path) -> logging.Logger:
    logger = logging.getLogger("MSASL_Scale")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
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
    url = url.strip()
    if not url.startswith("http://") and not url.startswith("https://"):
        url = "https://" + url
    return url


def extract_video_id(url: str) -> str:
    """Extract YouTube video ID or sanitized key from URL."""
    match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11}).*", url)
    if match:
        return match.group(1)
    return re.sub(r"[^A-Za-z0-9_-]", "_", url)[-20:]


MANIFEST_COLUMNS = [
    "sample_id",
    "msasl_label",
    "msasl_class",
    "asl_citizen_class",
    "signer_id",
    "url",
    "start_time",
    "end_time",
    "download_status",
    "clip_status",
    "preprocessing_status",
    "detected_frames",
    "tensor_shape",
    "tensor_dtype",
    "landmark_path",
    "is_pipeline_compatible",
    "failure_reason",
]


def classify_youtube_error(err_msg: str) -> Tuple[str, str]:
    """
    Classify yt-dlp / YouTube errors into (download_status, failure_reason).
    Returns status in ('unavailable', 'bot_challenge', 'transient_error', 'download_error').
    """
    if not err_msg or not isinstance(err_msg, str):
        return "download_error", "Unknown download error"

    err_lower = err_msg.lower()

    # 1. Permanent unavailable / private / deleted
    unavailable_keywords = [
        "private video",
        "video unavailable",
        "this video is unavailable",
        "this video is not available",
        "video deleted",
        "account associated with this video has been terminated",
        "video has been removed",
        "has been removed",
        "no longer available",
        "removed by the uploader",
    ]
    if any(k in err_lower for k in unavailable_keywords):
        return "unavailable", f"Unavailable: {err_msg[:200]}"

    # 2. Bot challenge / sign-in requirement
    bot_keywords = [
        "sign in to confirm you're not a bot",
        "sign in to confirm you’re not a bot",
        "please sign in",
        "use --cookies",
        "cookies-from-browser",
        "automated queries",
        "bot challenge",
    ]
    if any(k in err_lower for k in bot_keywords):
        return "bot_challenge", f"Bot challenge / sign-in required: {err_msg[:200]}"

    # 3. Transient network / rate limit errors
    transient_keywords = [
        "http error 429",
        "too many requests",
        "http error 403",
        "forbidden",
        "timeout",
        "timed out",
        "connection reset",
        "remotedisconnected",
        "temporary failure",
        "service unavailable",
        "http error 503",
        "http error 502",
        "unable to download video data",
    ]
    if any(k in err_lower for k in transient_keywords):
        return "transient_error", f"Transient error: {err_msg[:200]}"

    # 4. Fallback download error
    return "download_error", f"Download error: {err_msg[:200]}"


def get_cookie_options(
    cookies_file: Optional[str] = None,
    cookies_from_browser: Optional[str] = None,
) -> dict:
    """Safely build yt-dlp cookie options from CLI parameters or environment variables."""
    opts = {}
    env_c_file = os.environ.get("SIGNBRIDGE_YOUTUBE_COOKIES_FILE") or os.environ.get("YOUTUBE_COOKIES_FILE")
    env_c_browser = os.environ.get("SIGNBRIDGE_COOKIES_FROM_BROWSER") or os.environ.get("YOUTUBE_COOKIES_FROM_BROWSER")

    active_file = cookies_file or env_c_file
    active_browser = cookies_from_browser or env_c_browser

    if active_file and Path(active_file).exists():
        opts["cookiefile"] = str(Path(active_file).resolve())
    elif active_browser:
        opts["cookiesfrombrowser"] = (active_browser,)

    return opts


def build_ydl_opts(
    temp_template: Optional[str] = None,
    download_ranges: Optional[Any] = None,
    cookies_file: Optional[str] = None,
    cookies_from_browser: Optional[str] = None,
    socket_timeout: int = 25,
) -> dict:
    """Build yt-dlp options dictionary with custom headers and client settings."""
    opts = {
        "format": "bestvideo[ext=mp4][height<=720]+bestaudio[ext=m4a]/best[ext=mp4][height<=720]/best",
        "quiet": True,
        "no_warnings": True,
        "socket_timeout": socket_timeout,
        "retries": 3,
        "fragment_retries": 3,
        "extractor_args": {
            "youtube": {
                "player_client": ["android", "web"],
            }
        },
        "http_headers": {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9",
        },
    }
    if temp_template:
        opts["outtmpl"] = temp_template
    if download_ranges:
        opts["download_ranges"] = download_ranges
        opts["force_keyframes_at_cuts"] = True

    opts.update(get_cookie_options(cookies_file, cookies_from_browser))
    return opts


class ManifestManager:
    """Manages atomic, thread-safe, and crash-resilient manifest reading/writing."""

    def __init__(self, dl_path: Path, prep_path: Path, entry_lookup: Dict[Tuple, str], logger: logging.Logger):
        self.dl_path = dl_path
        self.prep_path = prep_path
        self.entry_lookup = entry_lookup
        self.logger = logger
        self.records: Dict[str, Dict[str, Any]] = {}
        self.unavailable_urls: Dict[str, str] = {}
        self._load_existing()

    def _get_key(self, row: Dict[str, Any]) -> Tuple:
        u = str(row.get("url", row.get("youtube_url", ""))).strip()
        if not u.startswith("http"):
            u = "https://" + u
        lbl = int(row.get("msasl_label", 0))
        st = round(float(row.get("start_time", 0.0)), 2)
        et = round(float(row.get("end_time", 0.0)), 2)
        return (u, lbl, st, et)

    def _load_existing(self):
        # 1. Load download manifest if exists
        if self.dl_path.exists():
            try:
                df_dl = pd.read_csv(self.dl_path)
                for _, row in df_dl.iterrows():
                    d = row.to_dict()
                    sid = str(d.get("sample_id", "")).strip()
                    if not sid or sid == "nan":
                        sid = self.entry_lookup.get(self._get_key(d), "")
                        d["sample_id"] = sid
                    if sid:
                        dl_st = str(d.get("download_status", ""))
                        f_reason = str(d.get("failure_reason", ""))
                        # Reclassify existing error statuses with refined error parser
                        if dl_st in ["download_error", "bot_challenge", "transient_error", "unavailable"] and f_reason and f_reason != "nan":
                            new_st, new_reason = classify_youtube_error(f_reason)
                            d["download_status"] = new_st
                            d["failure_reason"] = new_reason

                        self.records[sid] = d
                        u = d.get("url") or d.get("youtube_url")
                        if d.get("download_status") == "unavailable" and pd.notna(u):
                            self.unavailable_urls[clean_url(str(u))] = str(d.get("failure_reason", "Unavailable"))
            except Exception as e:
                self.logger.warning(f"Error reading existing download manifest: {e}")

        # 2. Merge preprocessing manifest if exists
        if self.prep_path.exists():
            try:
                df_pr = pd.read_csv(self.prep_path)
                for _, row in df_pr.iterrows():
                    d = row.to_dict()
                    sid = str(d.get("sample_id", "")).strip()
                    if not sid or sid == "nan":
                        sid = self.entry_lookup.get(self._get_key(d), "")
                        d["sample_id"] = sid
                    if sid:
                        if sid in self.records:
                            self.records[sid].update({k: v for k, v in d.items() if pd.notna(v)})
                        else:
                            self.records[sid] = d
            except Exception as e:
                self.logger.warning(f"Error reading existing preprocessing manifest: {e}")

        self.logger.info(f"Loaded {len(self.records)} existing records and {len(self.unavailable_urls)} known unavailable URLs.")

    def is_completed(self, sample_id: str, landmarks_dir: Path) -> bool:
        """
        Check if a sample is permanently completed and should be skipped.
        Only returns True for:
        1. Genuinely unavailable/private/deleted videos.
        2. Successfully preprocessed samples with valid landmark file on disk.
        """
        if sample_id not in self.records:
            return False
        rec = self.records[sample_id]
        if rec.get("download_status") == "unavailable":
            return True
        if rec.get("preprocessing_status") == "success":
            l_path = landmarks_dir / Path(str(rec.get("landmark_path", ""))).name
            if l_path.exists() and l_path.stat().st_size > 1024:
                return True
        return False

    def update_record(self, sample_id: str, data: Dict[str, Any]):
        if sample_id in self.records:
            self.records[sample_id].update(data)
        else:
            self.records[sample_id] = data

        if data.get("download_status") == "unavailable" and "url" in data:
            self.unavailable_urls[clean_url(str(data["url"]))] = str(data.get("failure_reason", "Unavailable"))

    def flush(self):
        """Atomic write to manifests with all required minimum columns."""
        if not self.records:
            return

        df = pd.DataFrame(list(self.records.values()))
        if "url" not in df.columns and "youtube_url" in df.columns:
            df["url"] = df["youtube_url"]
        elif "youtube_url" in df.columns:
            df["url"] = df["url"].fillna(df["youtube_url"])

        for col in MANIFEST_COLUMNS:
            if col not in df.columns:
                df[col] = ""

        df = df[MANIFEST_COLUMNS]

        # 1. Download manifest
        tmp_dl = self.dl_path.with_suffix(".tmp")
        df.to_csv(tmp_dl, index=False)
        tmp_dl.replace(self.dl_path)

        # 2. Preprocessing manifest
        tmp_pr = self.prep_path.with_suffix(".tmp")
        df.to_csv(tmp_pr, index=False)
        tmp_pr.replace(self.prep_path)


def download_raw_video(
    url: str,
    raw_dir: Path,
    cookies_file: Optional[str] = None,
    cookies_from_browser: Optional[str] = None,
    max_retries: int = 3,
    request_delay: float = 0.5,
) -> Tuple[Optional[Path], str, Optional[str]]:
    """
    Download full raw video for multi-clip reuse with retries.
    Returns (raw_path, download_status, failure_reason).
    """
    clean_video_url = clean_url(url)
    raw_template = str(raw_dir / "%(id)s.%(ext)s")

    vid = extract_video_id(clean_video_url)
    existing_matches = list(raw_dir.glob(f"{vid}.*"))
    if existing_matches and existing_matches[0].stat().st_size > 1024:
        return existing_matches[0], "success", None

    if request_delay > 0:
        time.sleep(request_delay)

    last_status = "download_error"
    last_reason = "Unknown raw download error"

    for attempt in range(1, max_retries + 1):
        try:
            ydl_opts = build_ydl_opts(
                temp_template=raw_template,
                cookies_file=cookies_file,
                cookies_from_browser=cookies_from_browser,
            )
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(clean_video_url, download=True)
                video_id = info.get("id")
                ext = info.get("ext", "mp4")

            raw_path = raw_dir / f"{video_id}.{ext}"
            if raw_path.exists() and raw_path.stat().st_size > 1024:
                return raw_path, "success", None
            matches = list(raw_dir.glob(f"{video_id}.*"))
            if matches and matches[0].stat().st_size > 1024:
                return matches[0], "success", None

        except yt_dlp.utils.DownloadError as e:
            err_msg = str(e).strip()
            status, reason = classify_youtube_error(err_msg)
            last_status, last_reason = status, reason
            if status == "unavailable":
                return None, "unavailable", reason
            if attempt < max_retries and status in ["transient_error", "bot_challenge"]:
                backoff = 2 ** attempt
                time.sleep(backoff)
                continue
            return None, status, reason
        except Exception as e:
            err_msg = str(e).strip()
            status, reason = classify_youtube_error(err_msg)
            last_status, last_reason = status, reason
            return None, status, reason

    return None, last_status, last_reason


def cut_clip_from_raw(raw_path: Path, start_time: float, end_time: float, clip_output_path: Path) -> Tuple[bool, Optional[str]]:
    """Cut clip from local raw video using ffmpeg."""
    duration = max(0.5, end_time - start_time)
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start_time),
        "-t", str(duration),
        "-i", str(raw_path),
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "22",
        "-c:a", "aac",
        str(clip_output_path)
    ]
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=40)
        if res.returncode == 0 and clip_output_path.exists() and clip_output_path.stat().st_size > 1024:
            return True, None
        return False, f"ffmpeg error: {res.stderr.decode()[:200]}"
    except Exception as e:
        return False, str(e)[:200]


def download_single_clip(
    url: str,
    start_time: float,
    end_time: float,
    clip_output_path: Path,
    raw_dir: Path,
    timeout_sec: int = 40,
    cookies_file: Optional[str] = None,
    cookies_from_browser: Optional[str] = None,
    max_retries: int = 3,
    request_delay: float = 0.5,
) -> Tuple[str, str, Optional[str]]:
    """
    Download and trim a single clip with retries and error classification.
    Returns (download_status, clip_status, failure_reason).
    """
    clean_video_url = clean_url(url)
    clip_output_path.parent.mkdir(parents=True, exist_ok=True)

    if request_delay > 0:
        time.sleep(request_delay)

    def section_range(info_dict, ydl):
        return [{"start_time": max(0.0, start_time), "end_time": max(start_time + 0.5, end_time)}]

    temp_template = str(clip_output_path.parent / f"temp_{clip_output_path.stem}.%(ext)s")

    last_status = "download_error"
    last_reason = "Unknown download error"

    for attempt in range(1, max_retries + 1):
        try:
            ydl_opts = build_ydl_opts(
                temp_template=temp_template,
                download_ranges=section_range,
                cookies_file=cookies_file,
                cookies_from_browser=cookies_from_browser,
            )
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([clean_video_url])

            matching_files = list(clip_output_path.parent.glob(f"temp_{clip_output_path.stem}.*"))
            if matching_files and matching_files[0].exists() and matching_files[0].stat().st_size > 1024:
                temp_file = matching_files[0]
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
            status, reason = classify_youtube_error(err_msg)
            last_status, last_reason = status, reason
            if status == "unavailable":
                return "unavailable", "skipped", reason
            if attempt < max_retries and status in ["transient_error", "bot_challenge"]:
                backoff = 2 ** attempt
                time.sleep(backoff)
                continue
            break
        except Exception as e:
            err_msg = str(e).strip()
            status, reason = classify_youtube_error(err_msg)
            last_status, last_reason = status, reason
            break

    # Strategy 2: Fallback via raw video download
    raw_path, raw_status, raw_reason = download_raw_video(
        url=url,
        raw_dir=raw_dir,
        cookies_file=cookies_file,
        cookies_from_browser=cookies_from_browser,
        max_retries=max_retries,
        request_delay=request_delay,
    )
    if raw_path and raw_path.exists():
        cut_ok, cut_err = cut_clip_from_raw(raw_path, start_time, end_time, clip_output_path)
        if cut_ok:
            return "success", "success", None
        else:
            return "success", "failed", cut_err or "ffmpeg clip cut failed"

    return raw_status or last_status, "failed", raw_reason or last_reason


def preprocess_clip(
    clip_path: Path,
    landmark_path: Path,
    gru_model: nn.Module,
    bigru_model: nn.Module,
) -> Tuple[str, int, str, str, bool, Optional[str]]:
    """
    Run SignBridge MediaPipe extraction & normalization.
    Returns (status, detected_frames, shape_str, dtype_str, is_compatible, failure_reason).
    """
    try:
        raw_seq = extract_hand_sequence(clip_path, num_frames=32)
        detected_frames = sum(1 for f in raw_seq if not np.allclose(f, 0))

        norm_seq = normalize_sequence(raw_seq)

        features = norm_seq.reshape(32, 126).astype(np.float32)

        if np.isnan(features).any() or np.isinf(features).any():
            return "invalid_tensor", detected_frames, str(features.shape), str(features.dtype), False, "Tensor contains NaN or Inf"

        tmp_landmark_path = landmark_path.with_suffix(".tmp.npy")
        np.save(tmp_landmark_path, features)
        tmp_landmark_path.replace(landmark_path)

        tensor_input = torch.from_numpy(features).unsqueeze(0)
        with torch.no_grad():
            out1 = gru_model(tensor_input)
            out2 = bigru_model(tensor_input)

        is_compatible = (
            features.shape == (32, 126)
            and out1.shape == (1, 2731)
            and out2.shape == (1, 2731)
        )

        status = "success" if detected_frames > 0 else "no_hands_detected"
        reason = "" if status == "success" else "Zero hand landmarks detected in video"

        return status, detected_frames, str(features.shape), str(features.dtype), is_compatible, reason

    except Exception as e:
        return "error", 0, "", "", False, str(e)[:200]


def process_url_group(
    url: str,
    entries: List[Tuple[int, Dict[str, Any]]],
    clips_dir: Path,
    raw_dir: Path,
    landmarks_dir: Path,
    manifest: ManifestManager,
    gru_model: nn.Module,
    bigru_model: nn.Module,
    logger: logging.Logger,
    cookies_file: Optional[str] = None,
    cookies_from_browser: Optional[str] = None,
    max_retries: int = 3,
    request_delay: float = 0.5,
) -> List[Dict[str, Any]]:
    """Process all clips belonging to a single YouTube URL."""
    clean_v_url = clean_url(url)
    results = []

    if clean_v_url in manifest.unavailable_urls:
        cached_reason = manifest.unavailable_urls[clean_v_url]
        for e_idx, sample in entries:
            sample_id = f"msasl_train_{e_idx:05d}"
            rec = {
                "sample_id": sample_id,
                "msasl_label": sample["label"],
                "msasl_class": sample.get("clean_text", sample.get("text", "")),
                "asl_citizen_class": sample["asl_citizen_class"],
                "signer_id": sample["signer_id"],
                "url": url,
                "start_time": sample.get("start_time", 0.0),
                "end_time": sample.get("end_time", 0.0),
                "download_status": "unavailable",
                "clip_status": "skipped",
                "clip_file": "",
                "preprocessing_status": "skipped",
                "detected_frames": 0,
                "tensor_shape": "",
                "tensor_dtype": "",
                "landmark_path": "",
                "is_pipeline_compatible": False,
                "failure_reason": cached_reason,
            }
            manifest.update_record(sample_id, rec)
            results.append(rec)
        return results

    uncompleted = []
    for e_idx, sample in entries:
        sample_id = f"msasl_train_{e_idx:05d}"
        if manifest.is_completed(sample_id, landmarks_dir):
            results.append(manifest.records[sample_id])
        else:
            uncompleted.append((e_idx, sample))

    if not uncompleted:
        return results

    raw_video_path: Optional[Path] = None
    is_multi_clip = len(uncompleted) > 1

    if is_multi_clip:
        vid = extract_video_id(url)
        matches = list(raw_dir.glob(f"{vid}.*"))
        if matches and matches[0].stat().st_size > 1024:
            raw_video_path = matches[0]
        else:
            raw_video_path, raw_status, raw_err = download_raw_video(
                url=url,
                raw_dir=raw_dir,
                cookies_file=cookies_file,
                cookies_from_browser=cookies_from_browser,
                max_retries=max_retries,
                request_delay=request_delay,
            )
            if not raw_video_path:
                dl_status = raw_status or "download_error"
                for e_idx, sample in uncompleted:
                    sample_id = f"msasl_train_{e_idx:05d}"
                    rec = {
                        "sample_id": sample_id,
                        "msasl_label": sample["label"],
                        "msasl_class": sample.get("clean_text", sample.get("text", "")),
                        "asl_citizen_class": sample["asl_citizen_class"],
                        "signer_id": sample["signer_id"],
                        "url": url,
                        "start_time": sample.get("start_time", 0.0),
                        "end_time": sample.get("end_time", 0.0),
                        "download_status": dl_status,
                        "clip_status": "skipped" if dl_status == "unavailable" else "failed",
                        "clip_file": "",
                        "preprocessing_status": "skipped",
                        "detected_frames": 0,
                        "tensor_shape": "",
                        "tensor_dtype": "",
                        "landmark_path": "",
                        "is_pipeline_compatible": False,
                        "failure_reason": raw_err or "Raw download failed",
                    }
                    manifest.update_record(sample_id, rec)
                    results.append(rec)
                return results

    for e_idx, sample in uncompleted:
        if SHUTDOWN_REQUESTED:
            break

        sample_id = f"msasl_train_{e_idx:05d}"
        ms_cls = sample.get("clean_text", sample.get("text", ""))
        asl_cls = sample["asl_citizen_class"]
        lbl = sample["label"]
        sid = sample["signer_id"]
        st = float(sample.get("start_time", 0.0))
        et = float(sample.get("end_time", st + 2.0))

        clip_filename = f"{sample_id}_{asl_cls}_label{lbl}_signer{sid}.mp4"
        clip_path = clips_dir / clip_filename
        landmark_filename = clip_filename.replace(".mp4", "_landmarks.npy")
        landmark_path = landmarks_dir / landmark_filename

        dl_status = "success"
        clip_status = "success"
        fail_reason = ""

        if not clip_path.exists() or clip_path.stat().st_size <= 1024:
            if is_multi_clip and raw_video_path and raw_video_path.exists():
                success, cut_err = cut_clip_from_raw(raw_video_path, st, et, clip_path)
                if not success:
                    dl_status = "success"
                    clip_status = "failed"
                    fail_reason = cut_err or "Cut failed"
            else:
                dl_status, clip_status, fail_reason = download_single_clip(
                    url=url,
                    start_time=st,
                    end_time=et,
                    clip_output_path=clip_path,
                    raw_dir=raw_dir,
                    cookies_file=cookies_file,
                    cookies_from_browser=cookies_from_browser,
                    max_retries=max_retries,
                    request_delay=request_delay,
                )

        if clip_status != "success":
            rec = {
                "sample_id": sample_id,
                "msasl_label": lbl,
                "msasl_class": ms_cls,
                "asl_citizen_class": asl_cls,
                "signer_id": sid,
                "url": url,
                "start_time": st,
                "end_time": et,
                "download_status": dl_status,
                "clip_status": clip_status,
                "clip_file": "",
                "preprocessing_status": "skipped",
                "detected_frames": 0,
                "tensor_shape": "",
                "tensor_dtype": "",
                "landmark_path": "",
                "is_pipeline_compatible": False,
                "failure_reason": fail_reason or "Clip extraction failed",
            }
            manifest.update_record(sample_id, rec)
            results.append(rec)
            continue

        prep_status, det_frames, shape_str, dtype_str, is_compat, p_reason = preprocess_clip(
            clip_path=clip_path,
            landmark_path=landmark_path,
            gru_model=gru_model,
            bigru_model=bigru_model,
        )

        rec = {
            "sample_id": sample_id,
            "msasl_label": lbl,
            "msasl_class": ms_cls,
            "asl_citizen_class": asl_cls,
            "signer_id": sid,
            "url": url,
            "start_time": st,
            "end_time": et,
            "download_status": "success",
            "clip_status": "success",
            "clip_file": clip_filename,
            "preprocessing_status": prep_status,
            "detected_frames": det_frames,
            "tensor_shape": shape_str,
            "tensor_dtype": dtype_str,
            "landmark_path": str(landmark_path.relative_to(PROJECT_ROOT)),
            "is_pipeline_compatible": is_compat,
            "failure_reason": p_reason or "",
        }
        manifest.update_record(sample_id, rec)
        results.append(rec)

    return results


def run_scaling():
    parser = argparse.ArgumentParser(description="Scale MS-ASL Acquisition & Preprocessing")
    parser.add_argument("--max-samples", type=int, default=None, help="Maximum number of samples to process")
    parser.add_argument("--workers", type=int, default=2, help="Number of concurrent worker threads (default: 2)")
    parser.add_argument("--report-interval", type=int, default=20, help="Log progress every N samples")
    parser.add_argument("--cookies-file", type=str, default=None, help="Path to Netscape cookies text file")
    parser.add_argument("--cookies-from-browser", type=str, default=None, help="Browser name for yt-dlp cookies (e.g. chrome, firefox)")
    parser.add_argument("--delay", type=float, default=0.5, help="Pause between YouTube requests in seconds (default: 0.5)")
    parser.add_argument("--max-retries", type=int, default=3, help="Max retries for recoverable YouTube errors (default: 3)")
    parser.add_argument("--retry-failed-only", action="store_true", help="Filter target queue to only previously failed/unsuccessful samples")
    parser.add_argument("--sample-ids", type=str, default=None, help="Comma-separated list or file path containing explicit sample_ids to process")
    args = parser.parse_args()

    ms_asl_root = PROJECT_ROOT / "dataset" / "MS_ASL"
    raw_dir = ms_asl_root / "raw"
    clips_dir = ms_asl_root / "clips"
    landmarks_dir = ms_asl_root / "landmarks"
    metadata_dir = ms_asl_root / "metadata"
    logs_dir = ms_asl_root / "logs"

    for d in [raw_dir, clips_dir, landmarks_dir, metadata_dir, logs_dir]:
        d.mkdir(parents=True, exist_ok=True)

    log_file = logs_dir / "scale_acquisition.log"
    logger = setup_logger(log_file)

    logger.info("==================================================================")
    logger.info("SignBridge MS-ASL Robust Video Acquisition & Preprocessing")
    logger.info("==================================================================")
    logger.info(f"Root: {ms_asl_root} | Workers: {args.workers} | Delay: {args.delay}s | Retries: {args.max_retries}")

    matching_json_path = PROJECT_ROOT / "msasl_matching_train.json"
    overlap_csv_path = PROJECT_ROOT / "msasl_signbridge_overlap.csv"

    if not matching_json_path.exists() or not overlap_csv_path.exists():
        logger.error("Missing required metadata. Run scripts/analyze_msasl_overlap.py first.")
        sys.exit(1)

    with open(matching_json_path, "r", encoding="utf-8") as f:
        all_entries = json.load(f)

    overlap_df = pd.read_csv(overlap_csv_path)
    label_to_asl = dict(zip(overlap_df["msasl_label"], overlap_df["asl_citizen_class"]))

    total_target = len(all_entries)
    logger.info(f"Loaded {total_target} total matching train entries.")

    def make_entry_key(url, label, st, et):
        u = str(url).strip()
        if not u.startswith("http"):
            u = "https://" + u
        return (u, int(label), round(float(st), 2), round(float(et), 2))

    entry_lookup = {}
    for idx, e in enumerate(all_entries):
        k = make_entry_key(e["url"], e["label"], e.get("start_time", 0.0), e.get("end_time", 0.0))
        entry_lookup[k] = f"msasl_train_{idx:05d}"
        e["asl_citizen_class"] = label_to_asl[e["label"]]

    dl_manifest_path = metadata_dir / "download_manifest.csv"
    prep_manifest_path = metadata_dir / "preprocessing_manifest.csv"
    manifest = ManifestManager(dl_manifest_path, prep_manifest_path, entry_lookup, logger)

    # Models for validation
    gru_model = GRUClassifier(126, 256, 2, 2731)
    bigru_model = BiGRUAttentionPoolingClassifier(126, 256, 2, 2731)
    gru_model.eval()
    bigru_model.eval()

    # Parse targeted sample IDs if specified
    explicit_sample_ids: Optional[Set[str]] = None
    if args.sample_ids:
        if Path(args.sample_ids).exists():
            with open(args.sample_ids, "r", encoding="utf-8") as f:
                explicit_sample_ids = set(line.strip() for line in f if line.strip())
        else:
            explicit_sample_ids = set(s.strip() for s in args.sample_ids.split(",") if s.strip())
        logger.info(f"Targeting {len(explicit_sample_ids)} explicit sample IDs for processing.")

    # Filter target candidate entries
    candidate_entries = []
    for idx, e in enumerate(all_entries):
        sid = f"msasl_train_{idx:05d}"
        if explicit_sample_ids and sid not in explicit_sample_ids:
            continue

        if manifest.is_completed(sid, landmarks_dir):
            continue

        if args.retry_failed_only:
            rec = manifest.records.get(sid, {})
            if rec.get("download_status") not in ["download_error", "bot_challenge", "transient_error"]:
                continue

        candidate_entries.append((idx, e))

    if args.max_samples and len(candidate_entries) > args.max_samples:
        candidate_entries = candidate_entries[:args.max_samples]

    logger.info(f"Queued {len(candidate_entries)} uncompleted samples for processing/retry.")

    url_to_entries = defaultdict(list)
    for idx, e in candidate_entries:
        url_to_entries[e["url"].strip()].append((idx, e))

    logger.info(f"Clustered queued samples into {len(url_to_entries)} unique YouTube URLs.")

    already_completed = sum(1 for idx in range(total_target) if manifest.is_completed(f"msasl_train_{idx:05d}", landmarks_dir))
    logger.info(f"Already completed & validated in manifests: {already_completed} samples.")

    processed_count = 0
    target_count = len(candidate_entries)
    last_reported = 0

    start_time = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {}
        for url, entries in url_to_entries.items():
            fut = executor.submit(
                process_url_group,
                url=url,
                entries=entries,
                clips_dir=clips_dir,
                raw_dir=raw_dir,
                landmarks_dir=landmarks_dir,
                manifest=manifest,
                gru_model=gru_model,
                bigru_model=bigru_model,
                logger=logger,
                cookies_file=args.cookies_file,
                cookies_from_browser=args.cookies_from_browser,
                max_retries=args.max_retries,
                request_delay=args.delay,
            )
            futures[fut] = (url, len(entries))

        logger.info(f"Dispatched {len(futures)} URL batches to {args.workers} worker(s).\n")

        for fut in as_completed(futures):
            if SHUTDOWN_REQUESTED:
                logger.info("Shutdown requested. Cancelling pending tasks...")
                for f in futures:
                    f.cancel()
                break

            url, num_clips = futures[fut]
            try:
                res_records = fut.result()
                processed_count += len(res_records)
            except Exception as e:
                logger.error(f"Error processing URL {url}: {e}")

            manifest.flush()

            if processed_count - last_reported >= args.report_interval or processed_count >= target_count:
                last_reported = processed_count
                records = list(manifest.records.values())
                dl_ok = sum(1 for r in records if r.get("download_status") == "success")
                unavail = sum(1 for r in records if r.get("download_status") == "unavailable")
                bot_err = sum(1 for r in records if r.get("download_status") == "bot_challenge")
                transient_err = sum(1 for r in records if r.get("download_status") == "transient_error")
                dl_err = sum(1 for r in records if r.get("download_status") == "download_error")
                clipped = sum(1 for r in records if r.get("clip_status") == "success")
                prep_ok = sum(1 for r in records if r.get("preprocessing_status") == "success")
                valid_tensors = sum(1 for r in records if r.get("is_pipeline_compatible") is True)

                logger.info(
                    f"Progress: Attempted queued={processed_count}/{target_count} | Total Records={len(records)} | "
                    f"DL Success: {dl_ok} | Unavailable: {unavail} | Bot Blocked: {bot_err} | "
                    f"Transient: {transient_err} | Errors: {dl_err} | Valid Tensors: {valid_tensors}"
                )

    manifest.flush()
    elapsed = time.time() - start_time

    # Generate Detailed Report
    records = list(manifest.records.values())
    total_records = len(records)
    dl_success = sum(1 for r in records if r.get("download_status") == "success")
    unavail = sum(1 for r in records if r.get("download_status") == "unavailable")
    bot_challenges = sum(1 for r in records if r.get("download_status") == "bot_challenge")
    transient_errors = sum(1 for r in records if r.get("download_status") == "transient_error")
    dl_errors = sum(1 for r in records if r.get("download_status") == "download_error")
    clipped = sum(1 for r in records if r.get("clip_status") == "success")
    prep_success = sum(1 for r in records if r.get("preprocessing_status") == "success")
    media_pipe_fails = sum(1 for r in records if r.get("preprocessing_status") == "no_hands_detected")
    valid_tensors = sum(1 for r in records if r.get("is_pipeline_compatible") is True)

    successful_records = [r for r in records if r.get("is_pipeline_compatible") is True]
    unique_signers = len(set(r["signer_id"] for r in successful_records))
    classes_represented = len(set(r["asl_citizen_class"] for r in successful_records))

    failure_reasons = Counter(r.get("failure_reason", "") for r in records if r.get("failure_reason"))

    samples_per_class = Counter(r["asl_citizen_class"] for r in successful_records)
    cls_ge_5 = sum(1 for c, cnt in samples_per_class.items() if cnt >= 5)
    cls_ge_10 = sum(1 for c, cnt in samples_per_class.items() if cnt >= 10)
    cls_ge_20 = sum(1 for c, cnt in samples_per_class.items() if cnt >= 20)

    det_frames_list = [int(r.get("detected_frames", 0)) for r in successful_records]
    mean_det_frames = np.mean(det_frames_list) if det_frames_list else 0.0
    median_det_frames = np.median(det_frames_list) if det_frames_list else 0.0

    logger.info("\n" + "=" * 66)
    logger.info("SignBridge MS-ASL Robust Acquisition & Preprocessing Final Report")
    logger.info("=" * 66)
    logger.info(f"1. Total manifest records tracked:           {total_records}")
    logger.info(f"2. Samples processed in this run:             {processed_count}")
    logger.info(f"3. Successfully downloaded & clipped:         {dl_success} ({dl_success/total_records*100:.1f}%)")
    logger.info(f"4. Confirmed unavailable / private / deleted: {unavail} ({unavail/total_records*100:.1f}%)")
    logger.info(f"5. Bot challenges / sign-in blocked:          {bot_challenges} ({bot_challenges/total_records*100:.1f}%)")
    logger.info(f"6. Transient network / rate limit errors:     {transient_errors} ({transient_errors/total_records*100:.1f}%)")
    logger.info(f"7. Other download errors:                     {dl_errors} ({dl_errors/total_records*100:.1f}%)")
    logger.info(f"8. Successfully preprocessed:                 {prep_success}")
    logger.info(f"9. MediaPipe failures (0 hands detected):     {media_pipe_fails}")
    logger.info(f"10. Valid (32, 126) float32 tensors:          {valid_tensors} ({valid_tensors/total_records*100:.1f}%)")
    logger.info(f"11. Unique MS-ASL signers acquired:           {unique_signers}")
    logger.info(f"12. ASL Citizen classes represented:          {classes_represented} / 646")
    logger.info(f"13. Detected-frame statistics:                mean={mean_det_frames:.1f}, median={median_det_frames:.1f} / 32")
    logger.info(f"14. Classes with >= 5 successful samples:     {cls_ge_5} / {classes_represented}")
    logger.info(f"15. Classes with >= 10 successful samples:    {cls_ge_10} / {classes_represented}")
    logger.info(f"16. Classes with >= 20 successful samples:    {cls_ge_20} / {classes_represented}")
    logger.info(f"17. Total elapsed time:                       {elapsed:.1f}s")
    logger.info("=" * 66)

    logger.info("\nTop Failure Reasons:")
    for reason, cnt in failure_reasons.most_common(5):
        clean_r = str(reason).replace("\n", " ").strip()[:100]
        if clean_r and clean_r != "nan":
            logger.info(f"  - {clean_r}: {cnt} samples")


if __name__ == "__main__":
    run_scaling()
