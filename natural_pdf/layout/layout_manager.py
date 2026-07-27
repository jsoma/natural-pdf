"""Registration helpers for layout engines.

All shared behavior (options-class lookup, alias handling, instance creation
with install hints, provider registration) lives in
:class:`natural_pdf.engine_registry.manager_builder.EngineManagerSpec`;
this module supplies the layout-specific data.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Type, cast

from natural_pdf.engine_registry.manager_builder import EngineManagerSpec

from .base import LayoutDetector
from .layout_options import (
    BaseLayoutOptions,
    DocLayoutOptions,
    PaddleLayoutOptions,
    SuryaLayoutOptions,
    TATRLayoutOptions,
    VLMLayoutOptions,
    YOLOLayoutOptions,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lazy import helpers
# ---------------------------------------------------------------------------


def _lazy_import_yolo_detector() -> Type[LayoutDetector]:
    from .yolo import YOLODocLayoutDetector

    return cast(Type[LayoutDetector], YOLODocLayoutDetector)


def _lazy_import_tatr_detector() -> Type[LayoutDetector]:
    from .tatr import TableTransformerDetector

    return cast(Type[LayoutDetector], TableTransformerDetector)


def _lazy_import_paddle_detector() -> Type[LayoutDetector]:
    from .paddle import PaddleLayoutDetector

    return cast(Type[LayoutDetector], PaddleLayoutDetector)


def _lazy_import_surya_detector() -> Type[LayoutDetector]:
    from .surya import SuryaLayoutDetector

    return cast(Type[LayoutDetector], SuryaLayoutDetector)


def _lazy_import_vlm_detector() -> Type[LayoutDetector]:
    from .vlm import VLMLayoutDetector

    return cast(Type[LayoutDetector], VLMLayoutDetector)


def _lazy_import_doclayout_detector() -> Type[LayoutDetector]:
    from .doclayout import DocLayoutDetector

    return cast(Type[LayoutDetector], DocLayoutDetector)


# ---------------------------------------------------------------------------
# Engine data (name -> lazy class factory + options class)
# ---------------------------------------------------------------------------

_ENGINE_DEFS: list[tuple[str, Any, type]] = [
    ("yolo", _lazy_import_yolo_detector, YOLOLayoutOptions),
    ("tatr", _lazy_import_tatr_detector, TATRLayoutOptions),
    ("paddle", _lazy_import_paddle_detector, PaddleLayoutOptions),
    ("surya", _lazy_import_surya_detector, SuryaLayoutOptions),
    ("vlm", _lazy_import_vlm_detector, VLMLayoutOptions),
    ("doclayout", _lazy_import_doclayout_detector, DocLayoutOptions),
]

# Deprecated alias: "gemini" -> "vlm"
_DEPRECATED_ALIASES: Dict[str, str] = {
    "gemini": "vlm",
}

_MANAGER = EngineManagerSpec(
    capability="layout",
    label="Layout",
    engine_defs=_ENGINE_DEFS,
    deprecated_aliases=_DEPRECATED_ALIASES,
    install_hints={
        "yolo": "pip install doclayout_yolo",
        "paddle": 'pip install "natural-pdf[paddle]"',
        "surya": 'pip install "surya-ocr<0.15"',
        "doclayout": "pip install transformers torch",
    },
    fallback_options_class=BaseLayoutOptions,
    logger=logger,
)


# ---------------------------------------------------------------------------
# Module-level API (kept for existing importers)
# ---------------------------------------------------------------------------


def engine_name_for_options(options: BaseLayoutOptions) -> Optional[str]:
    """Return the engine name whose options_class matches *options*, or None.

    Deprecated aliases are skipped so the canonical name is always returned.
    """
    return _MANAGER.engine_name_for_options(options)


def get_options_class_for_engine(name: str) -> Optional[type]:
    """Return the options class registered for *name*, or None."""
    return _MANAGER.get_options_class_for_engine(name)


def _create_engine_instance(engine_name: str) -> LayoutDetector:
    """Create a new layout engine instance. EngineProvider handles caching."""
    return cast(LayoutDetector, _MANAGER.create_engine_instance(engine_name))


def register_layout_engines(provider=None) -> None:
    """Register all built-in layout engines (and aliases) with the EngineProvider."""
    _MANAGER.register_engines(provider)


# Register at import time so engines are discoverable immediately.
_MANAGER.register_at_import()


__all__ = [
    "engine_name_for_options",
    "get_options_class_for_engine",
    "register_layout_engines",
]
