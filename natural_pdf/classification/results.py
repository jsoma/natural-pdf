# natural_pdf/classification/results.py
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class CategoryScore:
    """Represents a category and its confidence score from classification."""

    label: str
    score: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {"category": self.label, "score": self.score}


@dataclass(init=False, repr=False)
class ClassificationResult(Mapping):
    """Results from a classification operation."""

    category: Optional[str]  # Can be None if scores are empty
    score: float
    scores: List[CategoryScore]
    model_id: str
    timestamp: datetime
    using: str  # 'text' or 'vision'
    parameters: Optional[Dict[str, Any]] = None

    def __init__(
        self,
        scores: List[CategoryScore],
        model_id: str,
        using: str,
        parameters: Optional[Dict[str, Any]] = None,
        timestamp: Optional[datetime] = None,
    ):
        if scores:
            sorted_scores = sorted(scores, key=lambda s: s.score, reverse=True)
            self.category = sorted_scores[0].label
            self.score = sorted_scores[0].score
            self.scores = sorted_scores
        else:
            self.category = None
            self.score = 0.0
            self.scores = []

        self.model_id = model_id
        self.using = using
        self.parameters = parameters or {}
        self.timestamp = timestamp or datetime.now()
        # Cache for to_dict(). Safe because instances are treated as immutable
        # after construction — do not mutate attributes after __init__.
        self._dict_cache: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert the classification result to a dictionary for serialization."""
        if self._dict_cache is None:
            self._dict_cache = {
                "category": self.category,
                "score": self.score,
                "scores": [s.to_dict() for s in self.scores],
                "model_id": self.model_id,
                "using": self.using,
                "parameters": self.parameters,
                "timestamp": self.timestamp.isoformat(),
            }
        return self._dict_cache

    def __repr__(self) -> str:
        return f"<ClassificationResult category='{self.category}' score={self.score:.3f} model='{self.model_id}'>"

    def __iter__(self):
        """Iterate over mapping keys."""
        return iter(self.to_dict())

    def __getitem__(self, key):
        """Dictionary-style access to attributes."""
        try:
            return self.to_dict()[key]
        except KeyError as exc:
            raise KeyError(key) from exc

    def __len__(self):
        return len(self.to_dict())
