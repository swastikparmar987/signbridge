from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class EmittedToken:
    gloss: str
    confidence: float
    t0_ms: float
    t1_ms: float
    trigger: str
    top_k: List[Dict[str, Any]] = field(default_factory=list)
    ood_prob: float = 1.0
    stroke_id: int = 0


class ContinuousLatticeDecoder:
    """
    Temporal hypothesis decoder and deduplicator for continuous ASL streaming.

    Combines multi-scale tail consensus voting with boundary-triggered stroke
    commits, suppresses OOD / transition frames, prevents accidental rapid double-fires
    on held signs, and retains Top-K per-slot alternatives for contextual NLP.
    """

    def __init__(
        self,
        min_conf: float = 0.18,
        boundary_commit_conf: float = 0.18,
        high_conf_instant: float = 0.58,
        consensus_hits: int = 2,
        min_duplicate_gap_ms: float = 850.0,
        min_inter_repeat_rest_ms: float = 240.0,
        max_utterance_tokens: int = 40,
    ):
        self.min_conf = min_conf
        self.boundary_commit_conf = boundary_commit_conf
        self.high_conf_instant = high_conf_instant
        self.consensus_hits = consensus_hits
        self.min_duplicate_gap_ms = min_duplicate_gap_ms
        self.min_inter_repeat_rest_ms = min_inter_repeat_rest_ms
        self.max_utterance_tokens = max_utterance_tokens
        self.reset_utterance()

    def reset_utterance(self) -> None:
        self.tokens: List[EmittedToken] = []
        self.live_candidate: Optional[Dict[str, Any]] = None
        self._recent_votes: List[Tuple[str, float, float, float, List[Dict[str, Any]]]] = []
        self._best_stroke_vote: Optional[Tuple[int, str, float, float, float, List[Dict[str, Any]], float]] = None
        self._paused_since_last_token: bool = False
        self._last_commit_t1_ms: float = -1e9
        self._last_boundary_commit_t0_ms: float = -1e9

    def note_pause_gap(self) -> None:
        self.live_candidate = None
        if self.tokens:
            self._paused_since_last_token = True

    def _valid_candidates(self, top_k: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out = []
        for item in top_k:
            g = str(item.get("gloss", "")).strip()
            if not g or g.upper() in ("NO_SIGN", "REST", "UNKNOWN", "<BLANK>"):
                continue
            conf = float(item.get("confidence", 0.0))
            if conf > 1.0:
                conf /= 100.0
            out.append({"gloss": g, "confidence": conf})
        return out

    def _is_same_word_continuation(self, prev: EmittedToken, t0_ms: float, t1_ms: float, stroke_id: int) -> bool:
        if stroke_id <= prev.stroke_id:
            return True
        if self._paused_since_last_token and (t1_ms - prev.t1_ms) >= 350.0:
            return False
        if (t0_ms - prev.t1_ms) < self.min_inter_repeat_rest_ms and (t1_ms - prev.t1_ms) < 1400.0:
            return True
        return False

    def _can_emit(self, gloss: str, t0_ms: float, t1_ms: float, trigger: str, conf: float, stroke_id: int) -> bool:
        if len(self.tokens) >= self.max_utterance_tokens:
            return False
        if not self.tokens:
            return True
        prev = self.tokens[-1]
        if prev.gloss == gloss:
            if self._is_same_word_continuation(prev, t0_ms, t1_ms, stroke_id):
                return False
            if self._paused_since_last_token and stroke_id > prev.stroke_id:
                return conf >= self.min_conf
            is_boundary = trigger in ("dip_boundary", "pause_boundary", "manual_finalize")
            if is_boundary and (t1_ms - prev.t1_ms >= self.min_duplicate_gap_ms * 0.65):
                return conf >= self.boundary_commit_conf
            return (t1_ms - prev.t1_ms) >= self.min_duplicate_gap_ms and conf >= self.high_conf_instant
        return True

    def _commit_token(
        self,
        gloss: str,
        conf: float,
        t0_ms: float,
        t1_ms: float,
        trigger: str,
        top_k: List[Dict[str, Any]],
        ood_prob: float,
        stroke_id: int,
    ) -> EmittedToken:
        emitted = EmittedToken(
            gloss=gloss,
            confidence=conf,
            t0_ms=t0_ms,
            t1_ms=t1_ms,
            trigger=trigger,
            top_k=top_k[:3],
            ood_prob=ood_prob,
            stroke_id=stroke_id,
        )
        self.tokens.append(emitted)
        self._last_commit_t1_ms = t1_ms
        self._paused_since_last_token = trigger == "pause_boundary"
        self._recent_votes.clear()
        self._best_stroke_vote = None
        return emitted

    def _try_fallback_stroke_commit(
        self, stroke_id: int, t1_ms: float, trigger: str, min_fallback_conf: float
    ) -> Optional[EmittedToken]:
        vote = self._best_stroke_vote
        if vote is None or vote[0] != stroke_id or vote[2] < min_fallback_conf:
            return None
        b_sid, b_gloss, b_conf, b_t0, _b_t1, b_topk, b_ood = vote
        self._best_stroke_vote = None
        if not self._can_emit(b_gloss, b_t0, t1_ms, trigger, b_conf, b_sid):
            return None
        return self._commit_token(b_gloss, b_conf, b_t0, t1_ms, trigger, b_topk, b_ood, b_sid)

    def _make_live_candidate(
        self, gloss: str, conf: float, t0_ms: float, t1_ms: float, trigger: str, top_k: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        return {
            "gloss": gloss,
            "confidence": round(conf, 4),
            "t0_ms": round(t0_ms, 1),
            "t1_ms": round(t1_ms, 1),
            "trigger": trigger,
            "top_k": top_k[:3],
        }

    def observe_window(
        self,
        t0_ms: float,
        t1_ms: float,
        trigger: str,
        pred: Dict[str, Any],
        stroke_id: int = 0,
    ) -> Optional[EmittedToken]:
        is_no_sign = bool(pred.get("is_no_sign", False))
        ood_prob = float(pred.get("ood_prob", 1.0))
        top_k = self._valid_candidates(pred.get("top_k", []))
        if not top_k and not is_no_sign:
            g0 = str(pred.get("gloss") or pred.get("predicted_gloss") or "").strip()
            c0 = float(pred.get("confidence", 0.0))
            if c0 > 1.0:
                c0 /= 100.0
            if g0 and g0.upper() != "NO_SIGN":
                top_k = [{"gloss": g0, "confidence": c0}]

        is_boundary = trigger in ("dip_boundary", "pause_boundary", "manual_finalize")
        if is_no_sign or not top_k:
            self.live_candidate = None
            if is_boundary:
                emitted = self._try_fallback_stroke_commit(
                    stroke_id, t1_ms, trigger, max(self.boundary_commit_conf, 0.24)
                )
                if emitted is not None:
                    return emitted
            if trigger == "pause_boundary":
                self._best_stroke_vote = None
                self.note_pause_gap()
            return None

        top1 = top_k[0]
        gloss = top1["gloss"]
        conf = float(top1["confidence"])
        if conf < self.min_conf:
            # Expose non-OOD early candidate >= 4% (100x above 2731-class random) for live visual feedback
            self.live_candidate = (
                self._make_live_candidate(gloss, conf, t0_ms, t1_ms, trigger, top_k) if conf >= 0.04 else None
            )
            if is_boundary:
                emitted = self._try_fallback_stroke_commit(
                    stroke_id, t1_ms, trigger, max(self.boundary_commit_conf, 0.22)
                )
                if emitted is not None:
                    return emitted
            if trigger == "pause_boundary":
                self._best_stroke_vote = None
                self.note_pause_gap()
            return None

        stroke_already_committed = any(t.stroke_id == stroke_id for t in self.tokens)
        if not stroke_already_committed and (
            self._best_stroke_vote is None
            or self._best_stroke_vote[0] != stroke_id
            or conf >= self._best_stroke_vote[2]
        ):
            self._best_stroke_vote = (stroke_id, gloss, conf, t0_ms, t1_ms, top_k[:3], ood_prob)

        self.live_candidate = self._make_live_candidate(gloss, conf, t0_ms, t1_ms, trigger, top_k)

        # Update confidence/bounds when same gloss continues within same sign stroke
        if self.tokens and self.tokens[-1].gloss == gloss and self._is_same_word_continuation(self.tokens[-1], t0_ms, t1_ms, stroke_id):
            if conf > self.tokens[-1].confidence:
                self.tokens[-1].confidence = conf
                self.tokens[-1].top_k = top_k[:3]
            if stroke_id == self.tokens[-1].stroke_id:
                self.tokens[-1].t1_ms = max(self.tokens[-1].t1_ms, t1_ms)
            if trigger == "pause_boundary":
                self._paused_since_last_token = True
            return None

        emitted: Optional[EmittedToken] = None
        if is_boundary:
            # Refine an earlier tentative live_tail commit on the exact same stroke if boundary disagrees
            if (
                self.tokens
                and self.tokens[-1].stroke_id == stroke_id
                and self.tokens[-1].trigger == "live_tail"
                and self.tokens[-1].gloss != gloss
            ):
                if conf > self.tokens[-1].confidence * 1.15 and conf >= self.boundary_commit_conf:
                    self.tokens[-1].gloss = gloss
                    self.tokens[-1].confidence = conf
                    self.tokens[-1].t1_ms = max(self.tokens[-1].t1_ms, t1_ms)
                    self.tokens[-1].trigger = trigger
                    self.tokens[-1].top_k = top_k[:3]
                if trigger == "pause_boundary":
                    self._paused_since_last_token = True
                return None

            if conf >= self.boundary_commit_conf and self._can_emit(gloss, t0_ms, t1_ms, trigger, conf, stroke_id):
                emitted = self._commit_token(gloss, conf, t0_ms, t1_ms, trigger, top_k, ood_prob, stroke_id)
        else:
            # Streaming consensus across mature tail windows (>=460ms) or instant high-conf peak (>=260ms)
            self._recent_votes.append((gloss, conf, t0_ms, t1_ms, top_k[:3]))
            if len(self._recent_votes) > 8:
                self._recent_votes = self._recent_votes[-8:]
            recent_matches = [v for v in self._recent_votes[-4:] if v[0] == gloss]
            avg_conf = sum(v[1] for v in recent_matches) / max(1, len(recent_matches))
            span_ms = t1_ms - t0_ms
            mature_tail = span_ms >= 500.0
            should_commit = (
                (mature_tail and len(recent_matches) >= self.consensus_hits and avg_conf >= self.min_conf)
                or (span_ms >= 260.0 and conf >= self.high_conf_instant)
            )
            if should_commit:
                if (
                    self.tokens
                    and self.tokens[-1].stroke_id == stroke_id
                    and self.tokens[-1].trigger == "live_tail"
                    and self.tokens[-1].gloss != gloss
                    and conf > self.tokens[-1].confidence
                ):
                    self.tokens[-1].gloss = gloss
                    self.tokens[-1].confidence = max(conf, avg_conf)
                    self.tokens[-1].t1_ms = max(self.tokens[-1].t1_ms, t1_ms)
                    self.tokens[-1].top_k = top_k[:3]
                    self.tokens[-1].ood_prob = ood_prob
                    self._last_commit_t1_ms = self.tokens[-1].t1_ms
                    return None
                if self._can_emit(gloss, t0_ms, t1_ms, trigger, conf, stroke_id):
                    commit_t0 = recent_matches[0][2] if recent_matches else t0_ms
                    emitted = self._commit_token(
                        gloss, max(conf, avg_conf), commit_t0, t1_ms, trigger, top_k, ood_prob, stroke_id
                    )

        if emitted is None and trigger == "pause_boundary":
            self._best_stroke_vote = None
            self.note_pause_gap()
        return emitted

    def get_glosses(self) -> List[str]:
        return [t.gloss for t in self.tokens]

    def get_hypotheses(self) -> List[List[Dict[str, Any]]]:
        return [t.top_k[:3] for t in self.tokens]
