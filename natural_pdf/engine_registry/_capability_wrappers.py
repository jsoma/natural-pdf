"""Per-capability wrappers around :func:`natural_pdf.engine_registry.base.register_engine`.

Each wrapper binds one capability string so callers register engines without
spelling the capability themselves. These are deliberate explicit ``def``
wrappers (not ``functools.partial``) so signatures and docstrings show up in
IDEs and ``help()``.

``ocr`` and ``tables`` have real registration logic of their own and live in
their own modules (``ocr.py``, ``tables.py``).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Optional

from .base import register_engine

__all__ = [
    "register_checkbox_engine",
    "register_classification_engine",
    "register_deskew_engine",
    "register_guides_engine",
    "register_layout_engine",
    "register_selector_engine",
]


def register_checkbox_engine(
    name: str,
    factory: Callable[..., Any],
    *,
    replace: bool = True,
    metadata: Optional[dict[str, Any]] = None,
) -> None:
    """Register a checkbox detection engine under *name*."""
    register_engine("checkbox", name, factory, replace=replace, metadata=metadata)


def register_classification_engine(
    name: str,
    factory: Callable[..., Any],
    *,
    replace: bool = True,
    metadata: Optional[dict[str, Any]] = None,
) -> None:
    """Register a classification engine under *name*."""
    register_engine("classification", name, factory, replace=replace, metadata=metadata)


def register_deskew_engine(
    name: str,
    factory: Callable[..., Any],
    *,
    replace: bool = True,
    metadata: Optional[dict[str, Any]] = None,
    capabilities: Sequence[str] = ("deskew", "deskew.detect", "deskew.apply"),
) -> None:
    """Register a deskew engine under *name* for each capability in *capabilities*."""
    for capability in capabilities:
        register_engine(capability, name, factory, replace=replace, metadata=metadata)


def register_guides_engine(
    name: str,
    factory: Callable[..., Any],
    *,
    replace: bool = True,
    metadata: Optional[dict[str, Any]] = None,
) -> None:
    """Register a guide detection engine under *name*."""
    register_engine("guides.detect", name, factory, replace=replace, metadata=metadata)


def register_layout_engine(
    name: str,
    factory: Callable[..., Any],
    *,
    replace: bool = True,
    metadata: Optional[dict[str, Any]] = None,
) -> None:
    """Register a layout detection engine under *name*."""
    register_engine("layout", name, factory, replace=replace, metadata=metadata)


def register_selector_engine(
    name: str,
    factory: Callable[..., Any],
    *,
    replace: bool = True,
    metadata: Optional[dict[str, Any]] = None,
) -> None:
    """Register a selector engine under *name*."""
    register_engine("selectors", name, factory, replace=replace, metadata=metadata)
