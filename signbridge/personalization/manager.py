from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import json
import numpy as np
from datetime import datetime, timezone
import torch
import shutil

from signbridge.inference.predictor import SignBridgePredictor
from signbridge.preprocessing.normalize import normalize_sequence
from signbridge.personalization.models import (
    PersonalSignSample,
    PersonalSignPrototype,
    PersonalizationConfig,
)
from signbridge.personalization.quality import SampleQualityChecker


class PersonalizationManager:
    def __init__(
        self,
        data_dir: Path | str = None,
        predictor: SignBridgePredictor = None,
        config: PersonalizationConfig = None,
    ):
        self.data_dir = Path(data_dir) if data_dir else Path("data/personalization")
        self.predictor = predictor
        self.config = config or PersonalizationConfig()
        self.default_user = self.config.user_id

        self._ensure_dirs()

    def _ensure_dirs(self):
        self.data_dir.mkdir(parents=True, exist_ok=True)
        for user in [self.default_user]:
            user_dir = self._user_dir(user)
            (user_dir / "signs").mkdir(parents=True, exist_ok=True)
            (user_dir / "prototypes").mkdir(parents=True, exist_ok=True)

    def _user_dir(self, user_id: str) -> Path:
        return self.data_dir / user_id

    def _sign_dir(self, user_id: str, gloss: str) -> Path:
        return self._user_dir(user_id) / "signs" / gloss

    def _prototype_dir(self, user_id: str, gloss: str) -> Path:
        return self._user_dir(user_id) / "prototypes" / gloss

    def _metadata_path(self, user_id: str) -> Path:
        return self._user_dir(user_id) / "metadata.json"

    def _model_state_path(self, user_id: str) -> Path:
        return self._user_dir(user_id) / "model_state.json"

    def get_profile(self, user_id: str = None) -> Dict[str, Any]:
        uid = user_id or self.default_user
        meta_path = self._metadata_path(uid)
        if meta_path.exists():
            with open(meta_path) as f:
                return json.load(f)
        return {
            "user_id": uid,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "personalized_signs": [],
            "total_samples": 0,
        }

    def get_model_state(self, user_id: str = None) -> Dict[str, Any]:
        uid = user_id or self.default_user
        state_path = self._model_state_path(uid)
        if state_path.exists():
            with open(state_path) as f:
                return json.load(f)
        return {
            "version": "1.0",
            "enabled": self.config.enabled,
            "thresholds": {
                "rerank_similarity_threshold": self.config.rerank_similarity_threshold,
                "rerank_margin_threshold": self.config.rerank_margin_threshold,
                "override_similarity_threshold": self.config.override_similarity_threshold,
            },
            "total_signs_personalized": 0,
            "total_samples": 0,
            "last_updated": datetime.now(timezone.utc).isoformat(),
        }

    def list_personalized_signs(self, user_id: str = None) -> List[Dict[str, Any]]:
        uid = user_id or self.default_user
        signs_path = self._user_dir(uid) / "signs"
        if not signs_path.exists():
            return []
        result = []
        for gloss_dir in sorted(signs_path.iterdir()):
            if not gloss_dir.is_dir():
                continue
            prototype_path = self._prototype_dir(uid, gloss_dir.name) / "prototype.json"
            sample_files = list((gloss_dir / "samples").glob("*.json"))
            active_count = 0
            total_count = 0
            for sf in sample_files:
                with open(sf) as f:
                    s = json.load(f)
                total_count += 1
                if s["is_active"]:
                    active_count += 1
            proto = None
            if prototype_path.exists():
                with open(prototype_path) as f:
                    proto = json.load(f)
            result.append({
                "gloss": gloss_dir.name,
                "total_samples": total_count,
                "active_samples": active_count,
                "has_prototype": proto is not None,
                "prototype_sample_count": proto["sample_count"] if proto else 0,
                "last_updated": proto["updated_at"] if proto else None,
            })
        return result

    def get_personalized_sign_detail(self, gloss: str, user_id: str = None) -> Dict[str, Any]:
        uid = user_id or self.default_user
        sign_dir = self._sign_dir(uid, gloss)
        samples_dir = sign_dir / "samples"
        embeddings_dir = sign_dir / "embeddings"
        proto_path = self._prototype_dir(uid, gloss) / "prototype.json"

        if not sign_dir.exists():
            return {"gloss": gloss, "samples": [], "prototype": None, "exists": False}

        samples = []
        if samples_dir.exists():
            for sf in sorted(samples_dir.glob("*.json")):
                with open(sf) as f:
                    s = json.load(f)
                    s.pop("sequence", None)
                    s.pop("embedding", None)
                    samples.append(s)

        proto = None
        if proto_path.exists():
            with open(proto_path) as f:
                proto = json.load(f)

        return {
            "gloss": gloss,
            "exists": True,
            "samples": samples,
            "sample_count": len(samples),
            "active_sample_count": sum(1 for s in samples if s["is_active"]),
            "prototype": proto,
        }

    def save_sample(
        self,
        gloss: str,
        sequence: np.ndarray,
        embedding: np.ndarray,
        quality: Dict[str, Any],
        user_id: str = None,
    ) -> Dict[str, Any]:
        uid = user_id or self.default_user
        embedding = np.asarray(embedding, dtype=np.float32)
        norm = np.linalg.norm(embedding)
        if embedding.ndim != 1 or norm == 0 or not np.isfinite(norm):
            raise ValueError("Personalized embedding must be a finite, non-zero vector.")
        embedding = embedding / norm
        sign_dir = self._sign_dir(uid, gloss)
        samples_dir = sign_dir / "samples"
        embeddings_dir = sign_dir / "embeddings"
        samples_dir.mkdir(parents=True, exist_ok=True)
        embeddings_dir.mkdir(parents=True, exist_ok=True)

        # Do not store repeated windows from one continuous gesture.
        for existing in self._load_active_samples(gloss, uid):
            if existing.embedding.shape == embedding.shape and float(np.dot(existing.embedding, embedding)) >= 0.995:
                return {
                    "gloss": gloss,
                    "sample_id": existing.sample_id,
                    "quality_score": existing.quality_score,
                    "quality_feedback": "Duplicate gesture window ignored.",
                    "is_active": True,
                    "duplicate": True,
                    "total_samples": len(list(samples_dir.glob("*.json"))),
                }

        sample = PersonalSignSample(
            gloss=gloss,
            sequence=sequence,
            embedding=embedding,
            quality_score=quality["score"],
            quality_feedback=quality["feedback"],
            is_active=True if quality["valid"] else False,
            created_at=datetime.now(timezone.utc).isoformat(),
        )

        sample_meta = {
            "gloss": sample.gloss,
            "quality_score": sample.quality_score,
            "quality_feedback": sample.quality_feedback,
            "is_active": sample.is_active,
            "created_at": sample.created_at,
            "sample_id": sample.sample_id,
        }
        with open(samples_dir / f"{sample.sample_id}.json", "w") as f:
            json.dump(sample_meta, f, indent=2)

        np.save(embeddings_dir / f"{sample.sample_id}.npy", embedding)
        np.save(sign_dir / f"{sample.sample_id}.npy", sequence)

        self._recompute_prototype(gloss, uid)
        self._update_metadata(uid, gloss)

        return {
            "gloss": gloss,
            "sample_id": sample.sample_id,
            "quality_score": sample.quality_score,
            "quality_feedback": sample.quality_feedback,
            "is_active": sample.is_active,
            "total_samples": len(list(samples_dir.glob("*.json"))),
        }

    def _load_active_samples(self, gloss: str, user_id: str) -> List[PersonalSignSample]:
        sign_dir = self._sign_dir(user_id, gloss)
        samples_dir = sign_dir / "samples"
        embeddings_dir = sign_dir / "embeddings"
        if not samples_dir.exists():
            return []
        samples = []
        for sf in sorted(samples_dir.glob("*.json")):
            with open(sf) as f:
                meta = json.load(f)
            if not meta["is_active"]:
                continue
            emb_path = embeddings_dir / f"{meta['sample_id']}.npy"
            seq_path = sign_dir / f"{meta['sample_id']}.npy"
            if emb_path.exists() and seq_path.exists():
                emb = np.load(emb_path)
                seq = np.load(seq_path)
                samples.append(PersonalSignSample(
                    gloss=gloss,
                    sequence=seq,
                    embedding=emb,
                    quality_score=meta["quality_score"],
                    quality_feedback=meta["quality_feedback"],
                    is_active=meta["is_active"],
                    created_at=meta["created_at"],
                    sample_id=meta["sample_id"],
                ))
        return samples

    def _recompute_prototype(self, gloss: str, user_id: str) -> Optional[PersonalSignPrototype]:
        active_samples = self._load_active_samples(gloss, user_id)
        if not active_samples:
            return None

        embeddings = np.stack([s.embedding for s in active_samples])
        if embeddings.shape[0] < self.config.reraank_min_prototype_samples:
            return None

        prototype_embedding = embeddings.mean(axis=0)
        prototype_embedding = prototype_embedding / np.linalg.norm(prototype_embedding)

        sims = []
        for i, emb in enumerate(embeddings):
            for j, other in enumerate(embeddings):
                if i != j:
                    sims.append(float(np.dot(emb, other)))
        avg_sim = float(np.mean(sims)) if sims else 0.0

        proto = PersonalSignPrototype(
            gloss=gloss,
            embedding=prototype_embedding,
            sample_count=len(active_samples),
            average_similarity_to_others=round(avg_sim, 4),
            updated_at=datetime.now(timezone.utc).isoformat(),
        )

        proto_dir = self._prototype_dir(user_id, gloss)
        proto_dir.mkdir(parents=True, exist_ok=True)
        with open(proto_dir / "prototype.json", "w") as f:
            json.dump(proto.to_dict(), f, indent=2)

        return proto

    def _update_metadata(self, user_id: str, gloss: str):
        uid = user_id or self.default_user
        meta = self.get_profile(uid)
        if gloss not in meta["personalized_signs"]:
            meta["personalized_signs"].append(gloss)
        meta["total_samples"] = sum(
            len(list((self._sign_dir(uid, g) / "samples").glob("*.json")))
            for g in meta["personalized_signs"]
        )
        meta["updated_at"] = datetime.now(timezone.utc).isoformat()
        with open(self._metadata_path(uid), "w") as f:
            json.dump(meta, f, indent=2)

    def _load_prototype(self, gloss: str, user_id: str) -> Optional[np.ndarray]:
        proto_path = self._prototype_dir(user_id, gloss) / "prototype.json"
        if not proto_path.exists():
            return None
        with open(proto_path) as f:
            proto = json.load(f)
        return np.array(proto["embedding"], dtype=np.float32)

    def get_all_prototypes(self, user_id: str = None) -> Dict[str, np.ndarray]:
        uid = user_id or self.default_user
        signs = self.list_personalized_signs(uid)
        prototypes = {}
        for sign in signs:
            emb = self._load_prototype(sign["gloss"], uid)
            if emb is not None:
                prototypes[sign["gloss"]] = emb
        return prototypes

    def delete_sample(self, gloss: str, sample_id: str, user_id: str = None) -> Dict[str, Any]:
        uid = user_id or self.default_user
        sign_dir = self._sign_dir(uid, gloss)
        samples_dir = sign_dir / "samples"
        embeddings_dir = sign_dir / "embeddings"

        meta_path = samples_dir / f"{sample_id}.json"
        if meta_path.exists():
            meta_path.unlink()
        embedding_path = embeddings_dir / f"{sample_id}.npy"
        if embedding_path.exists():
            embedding_path.unlink()

        f3 = sign_dir / f"{sample_id}.npy"
        if f3.exists():
            f3.unlink()

        self._recompute_prototype(gloss, uid)
        self._update_metadata(uid, gloss)

        return {"gloss": gloss, "sample_id": sample_id, "deleted": True}

    def forget_sign(self, gloss: str, user_id: str = None) -> Dict[str, Any]:
        uid = user_id or self.default_user
        sign_dir = self._sign_dir(uid, gloss)
        prototype_dir = self._prototype_dir(uid, gloss)
        if sign_dir.exists():
            shutil.rmtree(sign_dir)
        if prototype_dir.exists():
            shutil.rmtree(prototype_dir)
        meta = self.get_profile(uid)
        meta["personalized_signs"] = [g for g in meta.get("personalized_signs", []) if g != gloss]
        meta["total_samples"] = sum(
            len(list((self._sign_dir(uid, g) / "samples").glob("*.json")))
            for g in meta["personalized_signs"]
            if self._sign_dir(uid, g).exists()
        )
        meta["updated_at"] = datetime.now(timezone.utc).isoformat()
        self._user_dir(uid).mkdir(parents=True, exist_ok=True)
        with open(self._metadata_path(uid), "w") as f:
            json.dump(meta, f, indent=2)
        return {"gloss": gloss, "deleted": True}

    def delete_all_personalization(self, user_id: str = None) -> Dict[str, Any]:
        uid = user_id or self.default_user
        user_dir = self._user_dir(uid)
        if user_dir.exists():
            shutil.rmtree(user_dir)
        (user_dir / "signs").mkdir(parents=True, exist_ok=True)
        (user_dir / "prototypes").mkdir(parents=True, exist_ok=True)
        return {"user_id": uid, "deleted": True, "total_signs_removed": 0}

    def retrain_prototype(self, gloss: str, user_id: str = None) -> Dict[str, Any]:
        uid = user_id or self.default_user
        proto = self._recompute_prototype(gloss, uid)
        if proto is None:
            return {"gloss": gloss, "status": "no_active_samples"}
        return {
            "gloss": gloss,
            "status": "recomputed",
            "sample_count": proto.sample_count,
            "average_similarity_to_others": proto.average_similarity_to_others,
            "updated_at": proto.updated_at,
        }

    def set_config(self, config: PersonalizationConfig) -> None:
        self.config = config
        uid = self.default_user
        state = self.get_model_state(uid)
        state["enabled"] = config.enabled
        state["thresholds"] = {
            "rerank_similarity_threshold": config.rerank_similarity_threshold,
            "rerank_margin_threshold": config.rerank_margin_threshold,
            "override_similarity_threshold": config.override_similarity_threshold,
        }
        state["last_updated"] = datetime.now(timezone.utc).isoformat()
        with open(self._model_state_path(uid), "w") as f:
            json.dump(state, f, indent=2)

    def get_config(self) -> Dict[str, Any]:
        state = self.get_model_state()
        return state.get("thresholds", {})

    def check_sample_quality(self, sequence: np.ndarray) -> Dict[str, Any]:
        return SampleQualityChecker.check(sequence)

    def extract_embedding(self, sequence: np.ndarray) -> np.ndarray:
        if self.predictor is None:
            raise RuntimeError("Predictor not set. Cannot extract embedding.")
        tensor_input = self.predictor._prepare_tensor(sequence, is_normalized=False)
        with torch.no_grad():
            embedding = self.predictor.model.extract_embedding(tensor_input)
        return embedding[0].cpu().numpy()
