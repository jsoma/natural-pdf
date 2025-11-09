"""Deskew registry helper."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Optional

from .base import register_engine

__all__ = ["register_deskew_engine"]


def register_deskew_engine(
    name: str,
    factory: Callable[..., Any],
    *,
    replace: bool = True,
    metadata: Optional[dict[str, Any]] = None,
    capabilities: Sequence[str] = ("deskew", "deskew.detect", "deskew.apply"),
) -> None:
    for capability in capabilities:
        register_engine(
            capability,
            name,
            factory,
            replace=replace,
            metadata=metadata,
        )
