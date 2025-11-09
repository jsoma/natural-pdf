"""OCR registry helpers."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Optional

from .base import register_engine

__all__ = ["register_ocr_engine"]


def register_ocr_engine(
    name: str,
    factory: Callable[..., Any],
    *,
    replace: bool = True,
    metadata: Optional[dict[str, Any]] = None,
    capabilities: Sequence[str] = ("ocr", "ocr.apply", "ocr.extract"),
) -> None:
    for capability in capabilities:
        register_engine(
            capability,
            name,
            factory,
            replace=replace,
            metadata=metadata,
        )
