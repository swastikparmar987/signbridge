from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
import numpy as np
from datetime import datetime, timezone


@dataclass
class PersonalSignSample:
    gloss: str
    sequence: np.ndarray
    embedding: np.ndarray
    quality_score: float
    quality_feedback: str
    is_active: bool
    created_at: str
    sample_id: str = ""

    def __post_init__(self):
        if not self.sample_id:
            self.sample_id = f"{self.gloss}_{int(datetime.now().timestamp() * 1000000)}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "gloss": self.gloss,
            "sequence": self.sequence.tolist(),
            "embedding": self.embedding.tolist(),
            "quality_score": self.quality_score,
            "quality_feedback": self.quality_feedback,
            "is_active": self.is_active,
            "created_at": self.created_at,
            "sample_id": self.sample_id,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PersonalSignSample":
        return cls(
            gloss=d["gloss"],
            sequence=np.array(d["sequence"], dtype=np.float32),
            embedding=np.array(d["embedding"], dtype=np.float32),
            quality_score=d["quality_score"],
            quality_feedback=d["quality_feedback"],
            is_active=d["is_active"],
            created_at=d["created_at"],
            sample_id=d.get("sample_id", ""),
        )


@dataclass
class PersonalSignPrototype:
    gloss: str
    embedding: np.ndarray
    sample_count: int
    average_similarity_to_others: float
    updated_at: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "gloss": self.gloss,
            "embedding": self.embedding.tolist(),
            "sample_count": self.sample_count,
            "average_similarity_to_others": self.average_similarity_to_others,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PersonalSignPrototype":
        return cls(
            gloss=d["gloss"],
            embedding=np.array(d["embedding"], dtype=np.float32),
            sample_count=d["sample_count"],
            average_similarity_to_others=d["average_similarity_to_others"],
            updated_at=d["updated_at"],
        )


@dataclass
class PersonalizationConfig:
    enabled: bool = False
    user_id: str = "default"
    min_samples_required: int = 3
    target_samples_per_sign: int = 10
    max_samples_per_sign: int = 20
    override_similarity_threshold: float = 0.90
    rerank_similarity_threshold: float = 0.75
    rerank_margin_threshold: float = 0.05
    reraank_min_prototype_samples: int = 3
