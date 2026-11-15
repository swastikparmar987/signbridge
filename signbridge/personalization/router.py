from typing import Dict, Any, List, Optional
import numpy as np


class PersonalizationRouter:
    def __init__(self, manager):
        self.manager = manager

    def get_thresholds(self, user_id: Optional[str] = None) -> Dict[str, float]:
        uid = user_id or self.manager.config.user_id
        state = self.manager.get_model_state(uid)
        config = self.manager.config
        saved = state.get("thresholds", {})
        # Migrate legacy 0.90 threshold cleanly down to calibrated config threshold
        saved_ov = saved.get("override_similarity_threshold", config.override_similarity_threshold)
        override_thr = (
            config.override_similarity_threshold
            if saved_ov >= 0.85 and config.override_similarity_threshold < 0.85
            else saved_ov
        )
        saved_rr = saved.get("rerank_similarity_threshold", config.rerank_similarity_threshold)
        rerank_thr = (
            config.rerank_similarity_threshold
            if saved_rr >= 0.72 and config.rerank_similarity_threshold < 0.72
            else saved_rr
        )
        return {
            "rerank_similarity_threshold": float(rerank_thr),
            "rerank_margin_threshold": float(saved.get("rerank_margin_threshold", config.rerank_margin_threshold)),
            "override_similarity_threshold": float(override_thr),
        }

    def route_prediction(
        self,
        global_predictions: List[Dict[str, Any]],
        query_embedding: Optional[np.ndarray],
        personalized_prototypes: Dict[str, np.ndarray],
        user_id: Optional[str] = None,
        sample_banks: Optional[Dict[str, np.ndarray]] = None,
    ) -> Dict[str, Any]:
        if not personalized_prototypes or query_embedding is None:
            return self._no_personalization(global_predictions)

        q = np.asarray(query_embedding, dtype=np.float32)
        if q.ndim != 1 or q.size == 0:
            return self._no_personalization(global_predictions)
        q_norm = float(np.linalg.norm(q))
        if not np.isfinite(q_norm) or q_norm <= 1e-8:
            return self._no_personalization(global_predictions)
        q = q / q_norm

        matching_protos = {
            g: emb for g, emb in personalized_prototypes.items()
            if getattr(emb, "shape", None) == q.shape
        }
        if not matching_protos:
            return self._no_personalization(global_predictions)

        thresholds = self.get_thresholds(user_id)
        proto_glosses = list(matching_protos.keys())
        scored: List[tuple[str, float]] = []
        for g in proto_glosses:
            p_emb = matching_protos[g]
            p_norm = float(np.linalg.norm(p_emb))
            centroid_sim = float((p_emb / max(1e-8, p_norm)) @ q)
            if sample_banks and g in sample_banks and sample_banks[g].shape[1:] == q.shape:
                max_sample_sim = float(np.max(sample_banks[g] @ q))
                blended = 0.65 * centroid_sim + 0.35 * max_sample_sim
            else:
                blended = centroid_sim
            scored.append((g, blended))

        scored.sort(key=lambda item: item[1], reverse=True)
        ranked = [(g, float(s)) for g, s in scored]

        best_gloss, best_sim = ranked[0]
        second_sim = float(ranked[1][1]) if len(ranked) > 1 else 0.0
        margin = (best_sim - second_sim) if len(ranked) > 1 else best_sim

        global_top1 = global_predictions[0] if global_predictions else None
        global_top5_glosses = {str(p.get("gloss", "")).upper() for p in global_predictions}
        global_confidence = float(global_top1["confidence"]) if global_top1 else 0.0
        global_margin = (
            float(global_predictions[0]["confidence"]) - float(global_predictions[1]["confidence"])
            if len(global_predictions) >= 2
            else 0.0
        )

        uid = user_id or self.manager.config.user_id
        sign_detail = self.manager.get_personalized_sign_detail(best_gloss, uid)
        proto_sample_count = (
            sign_detail.get("prototype", {}).get("sample_count", 0) if sign_detail.get("prototype") else 0
        )

        # Require stricter threshold when trained with only 1-2 samples vs >=3 samples
        override_threshold = thresholds["override_similarity_threshold"]
        if proto_sample_count < 3:
            override_threshold = max(override_threshold, 0.76)
        rerank_threshold = thresholds["rerank_similarity_threshold"]
        rerank_margin = thresholds["rerank_margin_threshold"]
        enough_multi_margin = len(ranked) == 1 or margin >= rerank_margin

        # Case D: strong custom-sign match (predicts user-trained signs even if outside 2,731 vocab)
        if proto_sample_count >= 1 and best_sim >= override_threshold and enough_multi_margin:
            return {
                "personalized": True,
                "matched": True,
                "mode": "override",
                "override_gloss": best_gloss,
                "predicted_gloss": best_gloss,
                "confidence": round(min(99.5, best_sim * 100.0), 2),
                "global_prediction": global_top1,
                "global_confidence": global_confidence,
                "global_margin": global_margin,
                "personalized_similarity": round(best_sim, 4),
                "personalized_margin": round(margin, 4),
                "top_personalized": ranked[:5],
                "threshold": override_threshold,
                "reason": f"Custom sign '{best_gloss}' matched (sim={best_sim:.3f} >= {override_threshold:.2f})",
            }

        # Case C: rerank when user sign is in global Top-K with strong similarity
        if (
            proto_sample_count >= 1
            and best_gloss.upper() in global_top5_glosses
            and best_sim >= rerank_threshold
            and enough_multi_margin
        ):
            return {
                "personalized": True,
                "matched": True,
                "mode": "rerank",
                "override_gloss": best_gloss,
                "predicted_gloss": best_gloss,
                "confidence": round(min(99.5, best_sim * 100.0), 2),
                "global_prediction": global_top1,
                "global_confidence": global_confidence,
                "global_margin": global_margin,
                "personalized_similarity": round(best_sim, 4),
                "personalized_margin": round(margin, 4),
                "top_personalized": ranked[:5],
                "threshold": rerank_threshold,
                "reason": f"Reranked custom sign '{best_gloss}' within Top-5 (sim={best_sim:.3f})",
            }

        # Case B: defer to global model
        return {
            "personalized": True,
            "matched": False,
            "mode": "defer",
            "override_gloss": None,
            "predicted_gloss": global_top1["gloss"] if global_top1 else best_gloss,
            "confidence": global_confidence,
            "global_prediction": global_top1,
            "global_confidence": global_confidence,
            "global_margin": global_margin,
            "personalized_similarity": round(best_sim, 4),
            "personalized_margin": round(margin, 4),
            "top_personalized": ranked[:5],
            "reason": f"Custom similarity {best_sim:.3f} below threshold {override_threshold:.2f}. Deferring.",
        }

    def _no_personalization(self, global_predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        if global_predictions:
            return {
                "personalized": False,
                "matched": False,
                "mode": "global",
                "override_gloss": None,
                "predicted_gloss": global_predictions[0]["gloss"],
                "confidence": global_predictions[0]["confidence"],
                "global_prediction": global_predictions[0],
                "global_confidence": global_predictions[0]["confidence"],
                "global_margin": (
                    float(global_predictions[0]["confidence"]) - float(global_predictions[1]["confidence"])
                ) if len(global_predictions) >= 2 else 0.0,
                "personalized_similarity": None,
                "personalized_margin": None,
                "top_personalized": [],
                "reason": "Personalization not enabled or no personalized signs.",
            }
        return {
            "personalized": False,
            "matched": False,
            "mode": "global",
            "override_gloss": None,
            "predicted_gloss": None,
            "confidence": 0.0,
            "global_prediction": None,
            "global_confidence": 0.0,
            "global_margin": 0.0,
            "personalized_similarity": None,
            "personalized_margin": None,
            "top_personalized": [],
            "reason": "No predictions available.",
        }
