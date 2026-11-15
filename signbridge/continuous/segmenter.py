from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from signbridge.preprocessing.landmarks import stabilize_hand_sequence


@dataclass
class CandidateWindow:
    start_idx: int
    end_idx: int
    t0_ms: float
    t1_ms: float
    trigger: str  # "dip_boundary", "pause_boundary", "live_tail", "manual_finalize"
    sequence_32: np.ndarray  # (32, 42, 3) float32 stabilized
    motion_score: float
    stroke_id: int = 0


def resample_sequence_aware(seq: np.ndarray, target_len: int = 32) -> np.ndarray:
    """
    Resample an arbitrary-length (W, 42, 3) hand sequence into target_len frames
    without linearly dragging missing (all-zero) hands through the origin (0, 0, 0).
    """
    seq = np.asarray(seq, dtype=np.float32)
    n = len(seq)
    if n == 0:
        return np.zeros((target_len, 42, 3), dtype=np.float32)
    if n == 1:
        return np.repeat(seq, target_len, axis=0)
    if n == target_len:
        return seq.copy()

    pos = np.linspace(0.0, float(n - 1), target_len, dtype=np.float32)
    lo = np.floor(pos).astype(np.int64)
    hi = np.minimum(lo + 1, n - 1)
    alpha = (pos - lo)[:, None, None]

    out = np.zeros((target_len, 42, 3), dtype=np.float32)
    for h_start, h_end in ((0, 21), (21, 42)):
        s_lo = seq[lo, h_start:h_end]
        s_hi = seq[hi, h_start:h_end]
        act_lo = np.abs(s_lo).sum(axis=(1, 2)) > 1e-4
        act_hi = np.abs(s_hi).sum(axis=(1, 2)) > 1e-4

        both = (act_lo & act_hi)[:, None, None]
        only_lo = (act_lo & ~act_hi)[:, None, None]
        only_hi = (~act_lo & act_hi)[:, None, None]

        interp = (1.0 - alpha) * s_lo + alpha * s_hi
        out[:, h_start:h_end] = np.where(
            both,
            interp,
            np.where(only_lo, s_lo, np.where(only_hi, s_hi, 0.0)),
        )
    return out


def resample_natural_cadence(
    seq: np.ndarray,
    target_len: int = 32,
    max_upscale: float = 1.45,
) -> np.ndarray:
    """
    Preserve natural gesture velocity when packing short strokes into target_len frames.

    Stretching a 6-frame sign to 32 frames divides per-frame speed by ~5x and pushes
    in-vocabulary signs into the OOD slow-drift regime. Cap upsampling to max_upscale
    and zero-pad symmetrically around the centered stroke instead.
    """
    arr = np.asarray(seq, dtype=np.float32)
    n = len(arr)
    if n == 0:
        return np.zeros((target_len, 42, 3), dtype=np.float32)
    if n >= target_len:
        return resample_sequence_aware(arr, target_len=target_len)

    max_scaled = int(round(n * max_upscale))
    scaled_n = min(target_len, max(n, max_scaled))
    core = (
        resample_sequence_aware(arr, target_len=scaled_n)
        if scaled_n != n
        else arr.copy()
    )
    if scaled_n >= target_len:
        return core

    pre = (target_len - scaled_n) // 2
    out = np.zeros((target_len, 42, 3), dtype=np.float32)
    out[pre : pre + scaled_n] = core
    return out


class KinematicStrokeSegmenter:
    """
    Real-time continuous signing stroke segmenter.

    Tracks palm velocity + local finger articulation speed over a rolling buffer,
    detecting:
      1) Coarticulation velocity dips (sign-to-sign boundaries during continuous flow)
      2) Rest/pause transitions (utterance or inter-word pauses)
      3) Multi-scale overlapping tail windows for immediate low-latency streaming feedback
    """

    def __init__(
        self,
        max_buffer_frames: int = 240,
        min_stroke_frames: int = 5,
        min_stroke_ms: float = 220.0,
        max_stroke_frames: int = 44,
        step_frames: int = 2,
        step_ms: float = 120.0,
        motion_start_thresh: float = 0.0094,
        motion_rest_thresh: float = 0.0058,
        pause_finalize_ms: float = 1050.0,
        ema_alpha: float = 0.35,
    ):
        self.max_buffer_frames = max_buffer_frames
        self.min_stroke_frames = min_stroke_frames
        self.min_stroke_ms = min_stroke_ms
        self.max_stroke_frames = max_stroke_frames
        self.step_frames = step_frames
        self.step_ms = step_ms
        self.motion_start_thresh = motion_start_thresh
        self.motion_rest_thresh = motion_rest_thresh
        self.pause_finalize_ms = pause_finalize_ms
        self.ema_alpha = ema_alpha
        self.reset()

    def reset(self) -> None:
        self.frames: List[np.ndarray] = []
        self.timestamps_ms: List[float] = []
        self.raw_speeds: List[float] = []
        self.ema_speeds: List[float] = []
        self.hands_counts: List[int] = []

        self.in_stroke: bool = False
        self.stroke_id: int = 0
        self.stroke_start_idx: int = 0
        self.last_active_idx: int = 0
        self.stroke_peak_speed: float = 0.0
        self.frames_since_tick: int = 0
        self.last_live_tick_ms: float = -1e9
        self.rest_start_ms: Optional[float] = None
        self.emitted_pause_boundary: bool = False
        self.last_boundary_idx: int = -1
        self.total_frames: int = 0

    def _frame_metrics(
        self, cur: np.ndarray, prev: Optional[np.ndarray], rate_scale: float = 1.0
    ) -> Tuple[int, float]:
        hands = 0
        palm_disp = 0.0
        local_disp = 0.0
        for h_s, h_e in ((0, 21), (21, 42)):
            c_hand = cur[h_s:h_e]
            c_active = np.abs(c_hand).sum() > 1e-4
            if not c_active:
                continue
            hands += 1
            if prev is not None and np.abs(prev[h_s:h_e]).sum() > 1e-4:
                p_hand = prev[h_s:h_e]
                scale = max(float(np.linalg.norm(c_hand[9, :2] - c_hand[0, :2])), 0.045)
                wrist_vel = float(np.linalg.norm(c_hand[0] - p_hand[0])) / scale * 0.12
                c_loc = c_hand - c_hand[0:1]
                p_loc = p_hand - p_hand[0:1]
                # Finger tips & PIP joints capture subtle handshape transitions
                art_idx = [4, 8, 12, 16, 20]
                finger_vel = float(np.linalg.norm(c_loc[art_idx] - p_loc[art_idx], axis=-1).mean()) / scale * 0.12
                palm_disp = max(palm_disp, wrist_vel * rate_scale)
                local_disp = max(local_disp, finger_vel * rate_scale)
        if hands == 0:
            return 0, 0.0
        speed = 0.50 * palm_disp + 0.50 * local_disp
        return hands, speed

    def _make_window(self, s_idx: int, e_idx: int, trigger: str) -> Optional[CandidateWindow]:
        s_idx = max(0, s_idx)
        e_idx = min(len(self.frames), e_idx)
        span = e_idx - s_idx
        is_close_boundary = trigger in ("pause_boundary", "manual_finalize")
        min_span = min(self.min_stroke_frames, 4) if is_close_boundary else max(self.min_stroke_frames, 6)
        min_ms = 95.0 if is_close_boundary else self.min_stroke_ms
        if span < min_span:
            return None
        elapsed_ms = float(self.timestamps_ms[e_idx - 1]) - float(self.timestamps_ms[s_idx])
        if elapsed_ms < min_ms and span < 12:
            return None
        seg = np.stack(self.frames[s_idx:e_idx], axis=0)
        res32 = stabilize_hand_sequence(resample_natural_cadence(seg, target_len=32))
        speeds = self.raw_speeds[s_idx:e_idx]
        motion_score = float(np.percentile(speeds, 85)) if speeds else 0.0
        return CandidateWindow(
            start_idx=self.total_frames - (len(self.frames) - s_idx),
            end_idx=self.total_frames - (len(self.frames) - e_idx),
            t0_ms=float(self.timestamps_ms[s_idx]),
            t1_ms=float(self.timestamps_ms[e_idx - 1]),
            trigger=trigger,
            sequence_32=res32,
            motion_score=motion_score,
            stroke_id=self.stroke_id,
        )

    def push_frame(
        self, frame: np.ndarray, ts_ms: Optional[float] = None
    ) -> Tuple[List[CandidateWindow], bool]:
        """
        Push a single (42, 3) frame into the stream segmenter.
        Returns (candidate_windows_to_evaluate, pause_finalized).
        """
        f = np.asarray(frame, dtype=np.float32).reshape(42, 3)
        if ts_ms is None:
            ts_ms = (self.timestamps_ms[-1] + 33.333) if self.timestamps_ms else 0.0

        prev = self.frames[-1] if self.frames else None
        dt_ms = max(float(ts_ms) - float(self.timestamps_ms[-1]), 8.0) if self.timestamps_ms else 33.333
        rate_scale = float(np.clip(33.333 / dt_ms, 0.45, 2.0))
        hands, speed = self._frame_metrics(f, prev, rate_scale=rate_scale)
        ema_prev = self.ema_speeds[-1] if self.ema_speeds else speed
        eff_alpha = float(np.clip(1.0 - (1.0 - self.ema_alpha) ** (dt_ms / 33.333), 0.18, 0.80))
        ema = eff_alpha * speed + (1.0 - eff_alpha) * ema_prev

        self.frames.append(f)
        self.timestamps_ms.append(float(ts_ms))
        self.raw_speeds.append(speed)
        self.ema_speeds.append(ema)
        self.hands_counts.append(hands)
        self.total_frames += 1

        if len(self.frames) > self.max_buffer_frames:
            trim = len(self.frames) - self.max_buffer_frames
            self.frames = self.frames[trim:]
            self.timestamps_ms = self.timestamps_ms[trim:]
            self.raw_speeds = self.raw_speeds[trim:]
            self.ema_speeds = self.ema_speeds[trim:]
            self.hands_counts = self.hands_counts[trim:]
            self.stroke_start_idx = max(0, self.stroke_start_idx - trim)
            self.last_active_idx = max(0, self.last_active_idx - trim)
            self.last_boundary_idx = max(-1, self.last_boundary_idx - trim)

        cur_len = len(self.frames)
        cur_idx = cur_len - 1
        windows: List[CandidateWindow] = []
        pause_finalized = False

        prev_raw = self.raw_speeds[-2] if len(self.raw_speeds) >= 2 else speed
        if hands > 0 and ema >= self.motion_start_thresh:
            self.rest_start_ms = None
            self.emitted_pause_boundary = False
            self.last_active_idx = cur_idx
            if not self.in_stroke:
                self.in_stroke = True
                self.stroke_id += 1
                self.stroke_start_idx = max(0, cur_idx - 2)
                self.stroke_peak_speed = ema
                self.frames_since_tick = 1
                self.last_live_tick_ms = -1e9
            else:
                self.stroke_peak_speed = max(self.stroke_peak_speed, ema)
                self.frames_since_tick += 1
        elif (
            hands > 0
            and self.in_stroke
            and speed >= self.motion_rest_thresh
            and (self.rest_start_ms is None or prev_raw >= self.motion_rest_thresh)
        ):
            # Require consecutive active frames to cancel an active rest timer against webcam tremor
            self.rest_start_ms = None
            self.last_active_idx = cur_idx
            self.frames_since_tick += 1
        else:
            if self.rest_start_ms is None:
                self.rest_start_ms = float(ts_ms)

        # Detect coarticulation dip boundary inside active continuous signing
        stroke_span = cur_len - self.stroke_start_idx
        stroke_elapsed_ms = (
            float(ts_ms) - float(self.timestamps_ms[self.stroke_start_idx])
            if self.stroke_start_idx < cur_len
            else 0.0
        )
        since_boundary_ms = (
            float(ts_ms) - float(self.timestamps_ms[self.last_boundary_idx])
            if 0 <= self.last_boundary_idx < cur_len
            else 1e9
        )
        if self.in_stroke and self.rest_start_ms is None and stroke_span >= 11 and stroke_elapsed_ms >= 580.0 and cur_len >= 4:
            s_m2, s_m1, s_cur = self.ema_speeds[-3], self.ema_speeds[-2], self.ema_speeds[-1]
            is_local_min = (s_m1 < s_m2) and (s_m1 <= s_cur)
            deep_dip = s_m1 < max(self.motion_start_thresh * 0.90, 0.30 * self.stroke_peak_speed)
            enough_sep = (cur_idx - 1 - self.last_boundary_idx) >= 9 and since_boundary_ms >= 500.0
            if is_local_min and deep_dip and enough_sep:
                w = self._make_window(self.stroke_start_idx, cur_idx, "dip_boundary")
                if w is not None:
                    windows.append(w)
                self.last_boundary_idx = cur_idx - 1
                self.stroke_id += 1
                self.stroke_start_idx = max(0, cur_idx - 1)
                self.stroke_peak_speed = max(ema, self.motion_start_thresh)
                self.frames_since_tick = 0

        # Detect transition into rest/pause; bridge brief internal micro-holds inside short strokes
        if self.rest_start_ms is not None:
            rest_elapsed = float(ts_ms) - self.rest_start_ms
            active_span = self.last_active_idx - self.stroke_start_idx + 1
            active_ms = (
                float(self.timestamps_ms[self.last_active_idx]) - float(self.timestamps_ms[self.stroke_start_idx])
                if 0 <= self.stroke_start_idx <= self.last_active_idx < cur_len
                else 0.0
            )
            mature_pause = (
                (hands == 0 and rest_elapsed >= 90.0 and active_span >= 4)
                or (rest_elapsed >= 120.0 and (active_span >= 12 or active_ms >= 260.0))
                or (rest_elapsed >= 240.0)
            )
            if self.in_stroke and mature_pause and not self.emitted_pause_boundary:
                end_cut = min(cur_len, max(self.stroke_start_idx + 1, self.last_active_idx + 1))
                start_cut = max(self.stroke_start_idx, end_cut - self.max_stroke_frames)
                w = self._make_window(start_cut, end_cut, "pause_boundary")
                if w is not None:
                    windows.append(w)
                self.emitted_pause_boundary = True
                self.in_stroke = False
                self.stroke_peak_speed = 0.0
                self.frames_since_tick = 0
            if rest_elapsed >= self.pause_finalize_ms:
                pause_finalized = True

        # Periodic multi-scale streaming windows while actively moving (FPS-adaptive)
        stroke_w = min(cur_len - self.stroke_start_idx, self.max_stroke_frames)
        if (
            self.in_stroke
            and self.rest_start_ms is None
            and stroke_w >= self.min_stroke_frames
            and stroke_elapsed_ms >= self.min_stroke_ms
            and self.frames_since_tick >= self.step_frames
            and (float(ts_ms) - self.last_live_tick_ms) >= self.step_ms
        ):
            self.frames_since_tick = 0
            self.last_live_tick_ms = float(ts_ms)
            avg_dt = max(12.0, stroke_elapsed_ms / max(1, stroke_w - 1))
            w_mid = int(round(720.0 / avg_dt))
            w_long = int(round(1050.0 / avg_dt))
            candidates = sorted(
                {w for w in (stroke_w, w_mid, w_long) if max(self.min_stroke_frames, 7) <= w <= min(stroke_w, cur_len)}
                | ({stroke_w} if stroke_w <= cur_len else set())
            )
            seen_bounds = set()
            for span in candidates:
                s_idx = max(self.stroke_start_idx, cur_len - span)
                if (s_idx, cur_len) in seen_bounds:
                    continue
                seen_bounds.add((s_idx, cur_len))
                w = self._make_window(s_idx, cur_len, "live_tail")
                if w is not None:
                    windows.append(w)
            if cur_len - self.stroke_start_idx > self.max_stroke_frames:
                # Forced valley split on ultra-long unbroken signing streams
                search_lo = self.stroke_start_idx + max(8, self.min_stroke_frames)
                search_hi = max(search_lo + 1, cur_len - max(6, self.min_stroke_frames))
                if search_hi > search_lo and search_hi <= len(self.ema_speeds):
                    valley_rel = int(np.argmin(self.ema_speeds[search_lo:search_hi]))
                    cut_idx = search_lo + valley_rel
                else:
                    cut_idx = cur_len - max(8, self.min_stroke_frames)
                w_split = self._make_window(self.stroke_start_idx, cut_idx + 1, "dip_boundary")
                if w_split is not None:
                    windows.append(w_split)
                self.last_boundary_idx = cut_idx
                self.stroke_id += 1
                self.stroke_start_idx = max(0, cut_idx)
                self.stroke_peak_speed = max(self.ema_speeds[self.stroke_start_idx:], default=ema)

        return windows, pause_finalized

    def flush_pending(self) -> List[CandidateWindow]:
        """Force flush any unclosed active tail window (e.g., at end-of-stream)."""
        if not self.in_stroke or not self.frames:
            return []
        cur_len = len(self.frames)
        start_cut = max(self.stroke_start_idx, cur_len - self.max_stroke_frames)
        w = self._make_window(start_cut, cur_len, "manual_finalize")
        self.in_stroke = False
        self.frames_since_tick = 0
        return [w] if w is not None else []
