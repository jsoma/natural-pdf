from __future__ import annotations

from typing import Any, Dict, Optional

from .results import ClassificationResult


class ClassificationResultAccessorMixin:
    """Adds convenience accessors for the latest classification analysis."""

    _classification_analysis_key: str = "classification"

    def _get_classification_analysis(self, analysis_key: Optional[str] = None) -> Any:
        """Return the stored analysis entry for the requested classification key."""
        analyses = getattr(self, "analyses", None)
        if analyses is None:
            return None
        key = analysis_key or self._classification_analysis_key
        if not key:
            return None
        return analyses.get(key)

    @property
    def category(self) -> Optional[str]:
        """Top category label for the last classification run."""
        result = self._get_classification_analysis()
        if result is None:
            return None
        if isinstance(result, ClassificationResult):
            return result.category
        if isinstance(result, dict):
            return result.get("category") or result.get("label")
        return getattr(result, "category", None)

    @property
    def category_confidence(self) -> Optional[float]:
        """Confidence score associated with ``category``."""
        result = self._get_classification_analysis()
        if result is None:
            return None
        if isinstance(result, ClassificationResult):
            return result.score
        if isinstance(result, dict):
            val = result.get("score")
            if val is None:
                val = result.get("confidence")
            if val is not None:
                try:
                    return float(val)
                except (TypeError, ValueError):
                    return None
            return None
        return getattr(result, "score", None)

    @property
    def classification_results(self) -> Optional[Dict[str, Any]]:
        """Full classification payload converted into a dictionary."""
        result = self._get_classification_analysis()
        if result is None:
            return None
        if isinstance(result, ClassificationResult):
            return result.to_dict()
        if isinstance(result, dict):
            return dict(result)
        to_dict_fn = getattr(result, "to_dict", None)
        if callable(to_dict_fn):
            data = to_dict_fn()
            if isinstance(data, dict):
                return data
        return None
