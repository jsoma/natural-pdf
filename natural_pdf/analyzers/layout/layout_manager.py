"""Registration helpers for layout engines."""

from __future__ import annotations

import inspect
import logging
import threading
from typing import Any, Dict, Type, cast

from natural_pdf.engine_provider import get_provider
from natural_pdf.engine_registry import register_builtin, register_layout_engine

from .base import LayoutDetector
from .layout_options import (
    DoclingLayoutOptions,
    GeminiLayoutOptions,
    LayoutOptions,
    PaddleLayoutOptions,
    SuryaLayoutOptions,
    TATRLayoutOptions,
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


def _lazy_import_docling_detector() -> Type[LayoutDetector]:
    from .docling import DoclingLayoutDetector

    return cast(Type[LayoutDetector], DoclingLayoutDetector)


def _lazy_import_gemini_detector() -> Type[LayoutDetector]:
    from .gemini import GeminiLayoutDetector

    return cast(Type[LayoutDetector], GeminiLayoutDetector)


# ---------------------------------------------------------------------------
# Registry and caching
# ---------------------------------------------------------------------------

ENGINE_REGISTRY: Dict[str, Dict[str, Any]] = {
    "yolo": {"class": _lazy_import_yolo_detector, "options_class": YOLOLayoutOptions},
    "tatr": {"class": _lazy_import_tatr_detector, "options_class": TATRLayoutOptions},
    "paddle": {"class": _lazy_import_paddle_detector, "options_class": PaddleLayoutOptions},
    "surya": {"class": _lazy_import_surya_detector, "options_class": SuryaLayoutOptions},
    "docling": {"class": _lazy_import_docling_detector, "options_class": DoclingLayoutOptions},
    "gemini": {"class": _lazy_import_gemini_detector, "options_class": GeminiLayoutOptions},
}

_detector_instances: Dict[str, LayoutDetector] = {}
_detector_lock = threading.RLock()


def _resolve_engine_class(engine_name: str) -> Type[LayoutDetector]:
    entry = ENGINE_REGISTRY[engine_name]["class"]
    if inspect.isclass(entry):
        return cast(Type[LayoutDetector], entry)
    return cast(Type[LayoutDetector], entry())


def _get_engine_instance(engine_name: str) -> LayoutDetector:
    engine_name = engine_name.lower()
    if engine_name not in ENGINE_REGISTRY:
        raise RuntimeError(
            f"Unknown layout engine '{engine_name}'. Available: {list(ENGINE_REGISTRY.keys())}"
        )

    with _detector_lock:
        if engine_name in _detector_instances:
            return _detector_instances[engine_name]

        logger.info("Creating layout engine instance: %s", engine_name)
        engine_class = _resolve_engine_class(engine_name)
        detector_instance = engine_class()

        try:
            available = detector_instance.is_available()
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Failed to check availability for %s", engine_name)
            raise RuntimeError(f"Layout engine '{engine_name}' availability check failed: {exc}")

        if not available:
            install_hint = (
                f"npdf install {engine_name}"
                if engine_name in {"yolo", "paddle", "surya", "docling"}
                else ""
            )
            raise RuntimeError(
                f"Layout engine '{engine_name}' is not available. {install_hint}".strip()
            )

        _detector_instances[engine_name] = detector_instance
        return detector_instance


# ---------------------------------------------------------------------------
# Provider registration
# ---------------------------------------------------------------------------


def register_layout_engines(provider=None) -> None:
    for engine_name in ENGINE_REGISTRY.keys():

        def factory(*, context=None, _engine_name=engine_name, **opts):
            return _get_engine_instance(_engine_name)

        register_builtin(provider, "layout", engine_name, factory)


try:  # Register at import time so engines are discoverable immediately.
    register_layout_engines()
except Exception:  # pragma: no cover - defensive
    logger.exception("Failed to register built-in layout engines")


__all__ = ["ENGINE_REGISTRY", "register_layout_engines"]
