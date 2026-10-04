from typing import Dict, Any, List, Optional
import numpy as np


class PersonalizationRouter:
    def __init__(self, manager):
        self.manager = manager

    def get_thresholds(self) -> Dict[str, float]:
        state = self.manager.get_model_state()
        config = self.manager.config
        return {
            "rerank_similarity_threshold": state.get("thresholds", {}).get(
                "rerank_similarity_threshold", config.rerank_similarity_threshold
            ),
            "rerank_margin_threshold": state.get("thresholds", {}).get(
                "rerank_margin_threshold", config.rerank_margin_threshold
            ),
            "override_similarity_threshold": state.get("thresholds", {}).get(
                "override_similarity_threshold", config.override_similarity_threshold
            ),
        }

    def route_prediction(
        self,
        global_predictions: List[Dict[str, Any]],
        query_embedding: Optional[np.ndarray],
        personalized_prototypes: Dict[str, np.ndarray],
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not personalized_prototypes or query_embedding is None:
            return self._no_personalization(global_predictions)

        thresholds = self.get_thresholds()
        proto_glosses = list(personalized_prototypes.keys())
        proto_embs = np.stack([personalized_prototypes[g] for g in proto_glosses])
        similarities = proto_embs @ query_embedding

        ranked_indices = np.argsort(similarities)[::-1]
        ranked = [(proto_glosses[i], float(similarities[i])) for i in ranked_indices]

        best_gloss, best_sim = ranked[0]
        second_sim = float(ranked[1][1]) if len(ranked) > 1 else 0.0
        margin = best_sim - second_sim

        global_top1 = global_predictions[0] if global_predictions else None
        global_top_k_glosses = set(p["gloss"] for p in global_predictions)
        global_top5_glosses = global_top_k_glosses
        global_confidence = global_top1["confidence"] if global_top1 else 0.0
        global_margin = (
            (global_predictions[0]["confidence"] - global_predictions[1]["confidence"])
            if len(global_predictions) >= 2
            else 0.0
        )

        state = self.manager.get_model_state(self.manager.config.user_id)
        config = self.manager.config

        override_threshold = thresholds["override_similarity_threshold"]
        rerank_threshold = thresholds["rerank_similarity_threshold"]
        rerank_margin = thresholds["rerank_margin_threshold"]
        min_proto_samples = config.reraank_min_prototype_samples

        uid = user_id or self.manager.config.user_id
        proto = self.manager._load_prototype(best_gloss, uid)
        # Access prototype via manager's stored prototype count
        sign_detail = self.manager.get_personalized_sign_detail(best_gloss, uid)
        proto_sample_count = sign_detail.get("prototype", {}).get("sample_count", 0) if sign_detail.get("prototype") else 0

        # Case D: strong override
        if (
            best_sim >= override_threshold
            and proto_sample_count >= min_proto_samples
        ):
            return {
                "personalized": True,
                "mode": "override",
                "predicted_gloss": best_gloss,
                "confidence": best_sim * 100.0,
                "global_prediction": global_top1,
                "global_confidence": global_confidence,
                "global_margin": global_margin,
                "personalized_similarity": best_sim,
                "personalized_margin": margin,
                "top_personalized": ranked[:5],
                "threshold": override_threshold,
                "reason": f"Personalized similarity {best_sim:.3f} >= {override_threshold}",
            }

        # Case C: rerank — personalized class is in global Top-K and similarity is strong
        if (
            best_gloss in global_top5_glosses
            and best_sim >= rerank_threshold
            and margin >= rerank_margin
            and proto_sample_count >= min_proto_samples
        ):
            return {
                "personalized": True,
                "mode": "rerank",
                "predicted_gloss": best_gloss,
                "confidence": best_sim * 100.0,
                "global_prediction": global_top1,
                "global_confidence": global_confidence,
                "global_margin": global_margin,
                "personalized_similarity": best_sim,
                "personalized_margin": margin,
                "top_personalized": ranked[:5],
                "threshold": rerank_threshold,
                "reason": f"Reranked within global Top-5 (sim={best_sim:.3f}, margin={margin:.3f})",
            }

        # Case B: weak personalized signal — defer to global
        return {
            "personalized": True,
            "mode": "defer",
            "predicted_gloss": global_top1["gloss"] if global_top1 else best_gloss,
            "confidence": global_confidence,
            "global_prediction": global_top1,
            "global_confidence": global_confidence,
            "global_margin": global_margin,
            "personalized_similarity": best_sim,
            "personalized_margin": margin,
            "top_personalized": ranked[:5],
            "reason": f"Personalized similarity {best_sim:.3f} below threshold. Deferring to global model.",
        }

    def _no_personalization(self, global_predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        if global_predictions:
            return {
                "personalized": False,
                "mode": "global",
                "predicted_gloss": global_predictions[0]["gloss"],
                "confidence": global_predictions[0]["confidence"],
                "global_prediction": global_predictions[0],
                "global_confidence": global_predictions[0]["confidence"],
                "global_margin": (
                    global_predictions[0]["confidence"] - global_predictions[1]["confidence"]
                ) if len(global_predictions) >= 2 else 0.0,
                "personalized_similarity": None,
                "personalized_margin": None,
                "top_personalized": [],
                "reason": "Personalization not enabled or no personalized signs.",
            }
        return {
            "personalized": False,
            "mode": "global",
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
