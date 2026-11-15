import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from signbridge.continuous.decoder import ContinuousLatticeDecoder, EmittedToken
from signbridge.continuous.segmenter import CandidateWindow, KinematicStrokeSegmenter


@dataclass
class ContinuousSession:
    session_id: str
    segmenter: KinematicStrokeSegmenter = field(default_factory=KinematicStrokeSegmenter)
    decoder: ContinuousLatticeDecoder = field(default_factory=ContinuousLatticeDecoder)
    last_active_ts: float = field(default_factory=time.time)


class ContinuousCSLRSessionManager:
    """
    Thread-safe manager for continuous webcam CSLR streaming sessions.
    """

    def __init__(self, session_ttl_sec: float = 900.0):
        self._sessions: Dict[str, ContinuousSession] = {}
        self._lock = threading.Lock()
        self._ttl_sec = session_ttl_sec

    def _prune_stale(self, now: float) -> None:
        stale = [sid for sid, sess in self._sessions.items() if now - sess.last_active_ts > self._ttl_sec]
        for sid in stale:
            del self._sessions[sid]

    def get_or_create(self, session_id: str) -> ContinuousSession:
        now = time.time()
        with self._lock:
            self._prune_stale(now)
            sess = self._sessions.get(session_id)
            if sess is None:
                sess = ContinuousSession(session_id=session_id)
                self._sessions[session_id] = sess
            sess.last_active_ts = now
            return sess

    def reset(self, session_id: str) -> None:
        with self._lock:
            sess = self._sessions.get(session_id)
            if sess is not None:
                sess.segmenter.reset()
                sess.decoder.reset_utterance()
                sess.last_active_ts = time.time()

    def _evaluate_windows(
        self,
        sess: ContinuousSession,
        windows: List[CandidateWindow],
        predict_fn: Callable[[np.ndarray, int], Dict[str, Any]],
        top_k: int = 5,
    ) -> List[EmittedToken]:
        # Group tail windows from the same push tick or score boundaries directly;
        # when multiple multi-scale live_tail windows fire together, pick the strongest valid score.
        newly_committed: List[EmittedToken] = []
        boundaries = [w for w in windows if w.trigger != "live_tail"]
        tails = [w for w in windows if w.trigger == "live_tail"]

        for w in boundaries:
            pred = predict_fn(w.sequence_32, top_k)
            tok = sess.decoder.observe_window(w.t0_ms, w.t1_ms, w.trigger, pred, stroke_id=w.stroke_id)
            if tok is not None:
                newly_committed.append(tok)

        if tails:
            best_w = None
            best_pred = None
            best_score = -1.0
            for w in tails:
                pred = predict_fn(w.sequence_32, top_k)
                conf = float(pred.get("confidence", 0.0)) if not pred.get("is_no_sign", False) else 0.0
                # Slight preference for longer well-supported stroke windows when confidence ties
                score = conf * (0.95 + 0.05 * min(1.0, (w.end_idx - w.start_idx) / 28.0))
                if score > best_score:
                    best_score = score
                    best_w = w
                    best_pred = pred
            if best_w is not None and best_pred is not None:
                tok = sess.decoder.observe_window(
                    best_w.t0_ms, best_w.t1_ms, best_w.trigger, best_pred, stroke_id=best_w.stroke_id
                )
                if tok is not None:
                    newly_committed.append(tok)

        return newly_committed

    def push_frames(
        self,
        session_id: str,
        frames: np.ndarray,
        predict_fn: Callable[[np.ndarray, int], Dict[str, Any]],
        timestamps_ms: Optional[List[float]] = None,
        auto_finalize_on_pause: bool = True,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        sess = self.get_or_create(session_id)
        arr = np.asarray(frames, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr[None, :, :]

        all_committed: List[EmittedToken] = []
        paused_triggered = False

        with self._lock:
            for i in range(len(arr)):
                ts = float(timestamps_ms[i]) if timestamps_ms is not None and i < len(timestamps_ms) else None
                windows, pause_finalized = sess.segmenter.push_frame(arr[i], ts)
                if windows:
                    committed = self._evaluate_windows(sess, windows, predict_fn, top_k=top_k)
                    all_committed.extend(committed)
                if pause_finalized and sess.decoder.tokens:
                    paused_triggered = True

            finalized_utterance = None
            if paused_triggered and auto_finalize_on_pause and sess.decoder.tokens:
                finalized_utterance = {
                    "glosses": sess.decoder.get_glosses(),
                    "hypotheses": sess.decoder.get_hypotheses(),
                    "tokens": [
                        {
                            "gloss": t.gloss,
                            "confidence": round(t.confidence, 4),
                            "t0_ms": round(t.t0_ms, 1),
                            "t1_ms": round(t.t1_ms, 1),
                            "trigger": t.trigger,
                            "top_k": t.top_k,
                        }
                        for t in sess.decoder.tokens
                    ],
                }
                sess.decoder.reset_utterance()

            return {
                "session_id": session_id,
                "newly_committed": [
                    {
                        "gloss": t.gloss,
                        "confidence": round(t.confidence, 4),
                        "t0_ms": round(t.t0_ms, 1),
                        "t1_ms": round(t.t1_ms, 1),
                        "trigger": t.trigger,
                        "top_k": t.top_k,
                    }
                    for t in all_committed
                ],
                "active_glosses": sess.decoder.get_glosses(),
                "active_hypotheses": sess.decoder.get_hypotheses(),
                "live_candidate": sess.decoder.live_candidate,
                "in_stroke": sess.segmenter.in_stroke,
                "stroke_frames": (
                    max(0, len(sess.segmenter.frames) - sess.segmenter.stroke_start_idx)
                    if sess.segmenter.in_stroke
                    else 0
                ),
                "pause_finalized": finalized_utterance is not None,
                "finalized_utterance": finalized_utterance,
            }

    def finalize(
        self,
        session_id: str,
        predict_fn: Optional[Callable[[np.ndarray, int], Dict[str, Any]]] = None,
        clear_after: bool = True,
    ) -> Dict[str, Any]:
        sess = self.get_or_create(session_id)
        with self._lock:
            if predict_fn is not None:
                pending = sess.segmenter.flush_pending()
                if pending:
                    self._evaluate_windows(sess, pending, predict_fn, top_k=5)
            out = {
                "session_id": session_id,
                "glosses": sess.decoder.get_glosses(),
                "hypotheses": sess.decoder.get_hypotheses(),
                "tokens": [
                    {
                        "gloss": t.gloss,
                        "confidence": round(t.confidence, 4),
                        "t0_ms": round(t.t0_ms, 1),
                        "t1_ms": round(t.t1_ms, 1),
                        "trigger": t.trigger,
                        "top_k": t.top_k,
                    }
                    for t in sess.decoder.tokens
                ],
            }
            if clear_after:
                sess.decoder.reset_utterance()
            return out
