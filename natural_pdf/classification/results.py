# natural_pdf/classification/results.py
import logging
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


@dataclass
class ClassificationResult:
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
        scores: List[CategoryScore],  # Now the primary source
        model_id: str,
        using: str,
        parameters: Optional[Dict[str, Any]] = None,
        timestamp: Optional[datetime] = None,
    ):
        # Determine top category and score from the scores list
        if scores:
            # Sort scores descending by score to find the top one
            sorted_scores = sorted(scores, key=lambda s: s.score, reverse=True)
            self.category = sorted_scores[0].label
            self.score = sorted_scores[0].score
            self.scores = sorted_scores  # Store the sorted list
        else:
            # Handle empty scores list
            self.category = None
            self.score = 0.0
            self.scores = []  # Store empty list

        self.model_id = model_id
        self.using = using
        self.parameters = parameters or {}
        self.timestamp = timestamp or datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the classification result to a dictionary for serialization.

        Returns:
            Dictionary representation of the classification result
        """
        return {
            "category": self.category,
            "score": self.score,
            "scores": [s.to_dict() for s in self.scores],
            "model_id": self.model_id,
            "using": self.using,
            "parameters": self.parameters,
            "timestamp": self.timestamp.isoformat(),
        }

    @property
    def top_category(self) -> str:
        """Returns the category with the highest score."""
        return self.category

    @property
    def top_confidence(self) -> float:
        """Returns the confidence score of the top category."""
        return self.score

    def __repr__(self) -> str:
        return f"<ClassificationResult category='{self.category}' score={self.score:.3f} model='{self.model_id}'>"
