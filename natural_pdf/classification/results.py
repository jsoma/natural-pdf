# natural_pdf/classification/results.py
from typing import List, Optional, Dict, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class CategoryScore:
    """Represents the score for a single category."""
    label: str
    confidence: float # Score between 0.0 and 1.0

    def __init__(self, label: str, confidence: float):
        # Basic validation
        if not isinstance(label, str) or not label:
             logger.warning(f"Initializing CategoryScore with invalid label: {label}")
             # Fallback or raise? For now, allow but log.
             # raise ValueError("Category label must be a non-empty string.")
        if not isinstance(confidence, (float, int)) or not (0.0 <= confidence <= 1.0):
             logger.warning(f"Initializing CategoryScore with invalid confidence: {confidence} for label '{label}'. Clamping to [0, 1].")
             confidence = max(0.0, min(1.0, float(confidence)))
             # raise ValueError("Category confidence must be a float between 0.0 and 1.0.")
        
        self.label = str(label)
        self.confidence = float(confidence)

    def __repr__(self):
        return f"<CategoryScore label='{self.label}' confidence={self.confidence:.3f}>"

class ClassificationResult:
    """Holds the structured results of a classification task."""
    model_id: str
    using: str # Renamed from engine_type ('text' or 'vision')
    timestamp: datetime
    parameters: Dict[str, Any] # e.g., {'categories': [...], 'min_confidence': 0.1}
    scores: List[CategoryScore] # List of scores above threshold, sorted by confidence

    def __init__(self, model_id: str, using: str, timestamp: datetime, parameters: Dict[str, Any], scores: List[CategoryScore]):
        if not isinstance(scores, list) or not all(isinstance(s, CategoryScore) for s in scores):
             raise TypeError("Scores must be a list of CategoryScore objects.")
             
        self.model_id = str(model_id)
        self.using = str(using) # Renamed from engine_type
        self.timestamp = timestamp
        self.parameters = parameters if parameters is not None else {}
        # Ensure scores are sorted descending by confidence
        self.scores = sorted(scores, key=lambda s: s.confidence, reverse=True)

    @property
    def top_category(self) -> Optional[str]:
        """Returns the label of the category with the highest confidence."""
        return self.scores[0].label if self.scores else None

    @property
    def top_confidence(self) -> Optional[float]:
        """Returns the confidence score of the top category."""
        return self.scores[0].confidence if self.scores else None

    def __repr__(self):
        top_cat = f" top='{self.top_category}' ({self.top_confidence:.2f})" if self.scores else ""
        num_scores = len(self.scores)
        return f"<ClassificationResult model='{self.model_id}' using='{self.using}' scores={num_scores}{top_cat}>" 