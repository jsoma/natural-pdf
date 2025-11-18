"""Classification helper functions shared across hosts."""

from __future__ import annotations

from typing import Any, List, Optional

from natural_pdf.services.base import resolve_service


def classify(
    self,
    labels: List[str],
    model: Optional[str] = None,
    using: Optional[str] = None,
    min_confidence: float = 0.0,
    analysis_key: str = "classification",
    multi_label: bool = False,
    **kwargs: Any,
):
    """Delegate classification to the classification service and return the result."""

    service = resolve_service(self, "classification")
    return service.classify(
        self,
        labels=labels,
        model=model,
        using=using,
        min_confidence=min_confidence,
        analysis_key=analysis_key,
        multi_label=multi_label,
        **kwargs,
    )


__all__ = ["classify"]
