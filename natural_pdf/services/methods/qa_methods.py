"""QA helper functions that preserve the canonical service signature."""

from __future__ import annotations

from typing import Any, Optional

from natural_pdf.core.qa_mixin import QuestionInput
from natural_pdf.services.base import resolve_service


def ask(
    self,
    question: QuestionInput,
    min_confidence: float = 0.1,
    model: Optional[str] = None,
    debug: bool = False,
    **kwargs: Any,
) -> Any:
    """Delegate QA execution to the shared QA service."""

    service = resolve_service(self, "qa")
    return service.ask(
        self,
        question=question,
        min_confidence=min_confidence,
        model=model,
        debug=debug,
        **kwargs,
    )


__all__ = ["ask"]
