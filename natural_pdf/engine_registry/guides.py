"""Guide detection registry helper."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Optional

from .base import register_engine

__all__ = ["register_guides_engine"]


def register_guides_engine(
    name: str,
    factory: Callable[..., Any],
    *,
    replace: bool = True,
    metadata: Optional[dict[str, Any]] = None,
) -> None:
    register_engine(
        "guides.detect",
        name,
        factory,
        replace=replace,
        metadata=metadata,
    )
