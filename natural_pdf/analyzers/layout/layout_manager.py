"""Registration helpers for layout engines."""

from __future__ import annotations

import inspect
import logging
from typing import Any, Dict, Optional, Type, cast

from natural_pdf.engine_provider import get_provider
from natural_pdf.engine_registry import register_builtin, register_layout_engine

from .base import LayoutDetector
from .layout_options import (
    BaseLayoutOptions,
    LayoutOptions,
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


# ---------------------------------------------------------------------------
# Engine definitions (name -> lazy class factory + options class)
# ---------------------------------------------------------------------------

_ENGINE_DEFS: list[tuple[str, Any, type]] = [
    ("yolo", _lazy_import_yolo_detector, YOLOLayoutOptions),
    ("tatr", _lazy_import_tatr_detector, TATRLayoutOptions),
    ("paddle", _lazy_import_paddle_detector, PaddleLayoutOptions),
    ("surya", _lazy_import_surya_detector, SuryaLayoutOptions),
    ("vlm", _lazy_import_vlm_detector, VLMLayoutOptions),
]

# Deprecated alias: "gemini" -> "vlm"
_DEPRECATED_ALIASES: Dict[str, str] = {
    "gemini": "vlm",
}
_DEPRECATION_WARNED: set = set()


# ---------------------------------------------------------------------------
# Helpers for querying options_class from EngineProvider metadata
# ---------------------------------------------------------------------------


def engine_name_for_options(options: BaseLayoutOptions) -> Optional[str]:
    """Return the engine name whose options_class matches *options*, or None.

    Deprecated aliases are skipped so the canonical name is always returned.
    """
    provider = get_provider()
    layout_engines = provider.list("layout").get("layout", ())
    for name in layout_engines:
        if name in _DEPRECATED_ALIASES:
            continue
        meta = provider.get_metadata("layout", name)
        if meta and isinstance(options, meta.get("options_class", type(None))):
            return name
    return None


def get_options_class_for_engine(name: str) -> Optional[type]:
    """Return the options class registered for *name*, or None."""
    provider = get_provider()
    meta = provider.get_metadata("layout", name)
    if meta:
        return meta.get("options_class")
    return None


# ---------------------------------------------------------------------------
# Instance creation
# ---------------------------------------------------------------------------


def _resolve_engine_class(engine_name: str) -> Type[LayoutDetector]:
    """Resolve the detector class for an engine name by querying registered metadata."""
    provider = get_provider()
    meta = provider.get_metadata("layout", engine_name)
    if not meta or "class_factory" not in meta:
        raise RuntimeError(
            f"Unknown layout engine '{engine_name}'. "
            f"Available: {list(provider.list('layout').get('layout', ()))}"
        )
    entry = meta["class_factory"]
    if inspect.isclass(entry):
        return cast(Type[LayoutDetector], entry)
    return cast(Type[LayoutDetector], entry())


def _create_engine_instance(engine_name: str) -> LayoutDetector:
    """Create a new layout engine instance. EngineProvider handles caching."""
    engine_name = engine_name.lower()

    # Handle deprecated aliases
    if engine_name in _DEPRECATED_ALIASES:
        canonical = _DEPRECATED_ALIASES[engine_name]
        if engine_name not in _DEPRECATION_WARNED:
            logger.warning(
                "Layout engine name '%s' is deprecated; use '%s' instead.",
                engine_name,
                canonical,
            )
            _DEPRECATION_WARNED.add(engine_name)
        engine_name = canonical

    logger.info("Creating layout engine instance: %s", engine_name)
    engine_class = _resolve_engine_class(engine_name)
    detector_instance = engine_class()

    try:
        available = detector_instance.is_available()
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Failed to check availability for %s", engine_name)
        raise RuntimeError(f"Layout engine '{engine_name}' availability check failed: {exc}")

    if not available:
        install_map = {
            "yolo": "pip install doclayout_yolo",
            "paddle": 'pip install "natural-pdf[paddle]"',
            "surya": 'pip install "surya-ocr<0.15"',
        }
        install_hint = install_map.get(engine_name, "")
        raise RuntimeError(
            f"Layout engine '{engine_name}' is not available. {install_hint}".strip()
        )

    return detector_instance


# ---------------------------------------------------------------------------
# Provider registration
# ---------------------------------------------------------------------------


def register_layout_engines(provider=None) -> None:
    for engine_name, class_factory, options_class in _ENGINE_DEFS:

        def factory(*, context=None, _engine_name=engine_name, **opts):
            return _create_engine_instance(_engine_name)

        register_builtin(
            provider,
            "layout",
            engine_name,
            factory,
            metadata={"options_class": options_class, "class_factory": class_factory},
        )

    # Register deprecated aliases so they are discoverable via EngineProvider
    for alias, canonical in _DEPRECATED_ALIASES.items():

        def alias_factory(*, context=None, _alias=alias, **opts):
            return _create_engine_instance(_alias)

        # Alias shares the canonical engine's options class
        canonical_options = next(
            (oc for name, _, oc in _ENGINE_DEFS if name == canonical), BaseLayoutOptions
        )
        register_builtin(
            provider,
            "layout",
            alias,
            alias_factory,
            metadata={"options_class": canonical_options},
        )


try:  # Register at import time so engines are discoverable immediately.
    register_layout_engines()
except Exception:  # pragma: no cover - defensive
    logger.exception("Failed to register built-in layout engines")


__all__ = [
    "engine_name_for_options",
    "get_options_class_for_engine",
    "register_layout_engines",
]
