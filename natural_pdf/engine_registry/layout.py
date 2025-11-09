"""Layout detector registration helper."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Optional

from .base import register_engine

__all__ = ["register_layout_engine"]


def register_layout_engine(
    name: str,
    factory: Callable[..., Any],
    *,
    replace: bool = True,
    metadata: Optional[dict[str, Any]] = None,
) -> None:
    register_engine(
        "layout",
        name,
        factory,
        replace=replace,
        metadata=metadata,
    )
