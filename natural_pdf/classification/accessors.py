from __future__ import annotations

from collections.abc import Mapping
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

        getter = getattr(analyses, "get", None)
        if callable(getter):
            return getter(key)

        if isinstance(analyses, Mapping):
            return analyses.get(key)

        return None

    @staticmethod
    def _extract_category(result: Any) -> Optional[str]:
        if result is None:
            return None

        category = getattr(result, "category", None)
        if category is not None:
            return category

        if isinstance(result, Mapping):
            category = result.get("category")
            if category is not None:
                return category
            label = result.get("label")
            if label is not None:
                return label

        return None

    @staticmethod
    def _extract_confidence(result: Any) -> Optional[float]:
        if result is None:
            return None

        for attr in ("score", "confidence", "category_confidence"):
            value = getattr(result, attr, None)
            if value is not None:
                try:
                    return float(value)
                except (TypeError, ValueError):
                    continue

        if isinstance(result, Mapping):
            for key in ("score", "confidence", "category_confidence"):
                value = result.get(key)
                if value is not None:
                    try:
                        return float(value)
                    except (TypeError, ValueError):
                        continue

        return None

    @staticmethod
    def _result_to_dict(result: Any) -> Optional[Dict[str, Any]]:
        if result is None:
            return None

        if isinstance(result, ClassificationResult):
            return result.to_dict()

        if isinstance(result, Mapping):
            return dict(result)

        to_dict = getattr(result, "to_dict", None)
        if callable(to_dict):
            data = to_dict()
            if isinstance(data, Mapping):
                return dict(data)
            if isinstance(data, dict):
                return data

        return None

    @property
    def category(self) -> Optional[str]:
        """Top category label for the last classification run."""

        result = self._get_classification_analysis()
        return self._extract_category(result)

    @property
    def category_confidence(self) -> Optional[float]:
        """Confidence score associated with ``category``."""

        result = self._get_classification_analysis()
        return self._extract_confidence(result)

    @property
    def classification_results(self) -> Optional[Dict[str, Any]]:
        """Full classification payload converted into a dictionary."""

        result = self._get_classification_analysis()
        return self._result_to_dict(result)
