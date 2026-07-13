"""Per-call OCR execution state for overlapping aggregate scopes."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Iterator


@dataclass(slots=True)
class OCRExecutionState:
    """Artifacts created by the currently executing public OCR operation."""

    protected_artifact_ids: set[int] = field(default_factory=set)


_CURRENT_OCR_EXECUTION: ContextVar[OCRExecutionState | None] = ContextVar(
    "natural_pdf_current_ocr_execution", default=None
)


@contextmanager
def ocr_execution_session() -> Iterator[OCRExecutionState]:
    """Create or reuse one nested OCR operation state in this task/thread."""

    state = _CURRENT_OCR_EXECUTION.get()
    if state is not None:
        yield state
        return

    state = OCRExecutionState()
    token = _CURRENT_OCR_EXECUTION.set(state)
    try:
        yield state
    finally:
        _CURRENT_OCR_EXECUTION.reset(token)


def register_ocr_artifacts(*artifacts: Any) -> None:
    """Protect newly created artifacts from later overlapping cleanup."""

    state = _CURRENT_OCR_EXECUTION.get()
    if state is None:
        return
    for artifact in artifacts:
        if artifact is not None:
            state.protected_artifact_ids.add(id(artifact))


def protected_ocr_artifact_ids() -> frozenset[int]:
    """Return identities protected by the current public OCR operation."""

    state = _CURRENT_OCR_EXECUTION.get()
    if state is None:
        return frozenset()
    return frozenset(state.protected_artifact_ids)


__all__ = [
    "OCRExecutionState",
    "ocr_execution_session",
    "protected_ocr_artifact_ids",
    "register_ocr_artifacts",
]
