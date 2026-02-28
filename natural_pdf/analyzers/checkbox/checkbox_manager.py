"""Registration helpers for checkbox detection engines.

Mirrors natural_pdf/analyzers/layout/layout_manager.py:
- Lazy import factories for each engine
- EngineProvider registration at import time
- Helpers for options-class <-> engine-name mapping
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Type, cast

from natural_pdf.engine_provider import get_provider
from natural_pdf.engine_registry import register_builtin, register_checkbox_engine

from .base import CheckboxDetector
from .checkbox_options import (
    BaseCheckboxOptions,
    DefaultCheckboxOptions,
    OnnxCheckboxOptions,
    VectorCheckboxOptions,
    VLMCheckboxOptions,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lazy import helpers
# ---------------------------------------------------------------------------


def _lazy_import_vector_detector() -> Type[CheckboxDetector]:
    from .vector import VectorCheckboxDetector

    return cast(Type[CheckboxDetector], VectorCheckboxDetector)


def _lazy_import_onnx_detector() -> Type[CheckboxDetector]:
    from .onnx_engine import OnnxCheckboxDetector

    return cast(Type[CheckboxDetector], OnnxCheckboxDetector)


def _lazy_import_vlm_detector() -> Type[CheckboxDetector]:
    from .vlm_detector import VLMCheckboxDetector

    return cast(Type[CheckboxDetector], VLMCheckboxDetector)


def _lazy_import_default_detector() -> Type[CheckboxDetector]:
    from .default_detector import DefaultCheckboxDetector

    return cast(Type[CheckboxDetector], DefaultCheckboxDetector)


# ---------------------------------------------------------------------------
# Engine definitions (name -> lazy class factory + options class)
# ---------------------------------------------------------------------------

_ENGINE_DEFS: list[tuple[str, Any, type]] = [
    ("vector", _lazy_import_vector_detector, VectorCheckboxOptions),
    ("default", _lazy_import_default_detector, DefaultCheckboxOptions),
    ("onnx", _lazy_import_onnx_detector, OnnxCheckboxOptions),
    ("vlm", _lazy_import_vlm_detector, VLMCheckboxOptions),
]

# Backward-compat aliases
_DEPRECATED_ALIASES: Dict[str, str] = {}
_DEPRECATION_WARNED: set = set()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def engine_name_for_options(options: BaseCheckboxOptions) -> Optional[str]:
    """Return the engine name whose options_class matches *options*, or None."""
    provider = get_provider()
    checkbox_engines = provider.list("checkbox").get("checkbox", ())
    for name in checkbox_engines:
        if name in _DEPRECATED_ALIASES:
            continue
        meta = provider.get_metadata("checkbox", name)
        if meta and isinstance(options, meta.get("options_class", type(None))):
            return name
    return None


def get_options_class_for_engine(name: str) -> Optional[type]:
    """Return the options class registered for *name*, or None."""
    provider = get_provider()
    meta = provider.get_metadata("checkbox", name)
    if meta:
        return meta.get("options_class")
    return None


# ---------------------------------------------------------------------------
# Instance creation
# ---------------------------------------------------------------------------


def _create_engine_instance(engine_name: str) -> CheckboxDetector:
    """Create a new checkbox engine instance."""
    engine_name = engine_name.lower()

    # Handle deprecated aliases
    if engine_name in _DEPRECATED_ALIASES:
        canonical = _DEPRECATED_ALIASES[engine_name]
        if engine_name not in _DEPRECATION_WARNED:
            logger.warning(
                "Checkbox engine name '%s' is deprecated; use '%s' instead.",
                engine_name,
                canonical,
            )
            _DEPRECATION_WARNED.add(engine_name)
        engine_name = canonical

    provider = get_provider()
    meta = provider.get_metadata("checkbox", engine_name)
    if not meta or "class_factory" not in meta:
        available = list(provider.list("checkbox").get("checkbox", ()))
        raise RuntimeError(f"Unknown checkbox engine '{engine_name}'. Available: {available}")

    entry = meta["class_factory"]
    if isinstance(entry, type):
        engine_class = entry
    else:
        engine_class = entry()

    detector = engine_class()

    try:
        available = detector.is_available()
    except Exception as exc:
        raise RuntimeError(f"Checkbox engine '{engine_name}' availability check failed: {exc}")

    if not available:
        install_map = {
            "onnx": "pip install onnxruntime huggingface_hub",
            "vlm": "pip install openai",
        }
        hint = install_map.get(engine_name, "")
        raise RuntimeError(f"Checkbox engine '{engine_name}' is not available. {hint}".strip())

    return detector


# ---------------------------------------------------------------------------
# Provider registration
# ---------------------------------------------------------------------------


def register_checkbox_engines(provider=None) -> None:
    """Register all built-in checkbox engines with the EngineProvider."""
    for engine_name, class_factory, options_class in _ENGINE_DEFS:

        def factory(*, context=None, _engine_name=engine_name, **opts):
            return _create_engine_instance(_engine_name)

        register_builtin(
            provider,
            "checkbox",
            engine_name,
            factory,
            metadata={"options_class": options_class, "class_factory": class_factory},
        )

    # Register deprecated aliases
    for alias, canonical in _DEPRECATED_ALIASES.items():

        def alias_factory(*, context=None, _alias=alias, **opts):
            return _create_engine_instance(_alias)

        canonical_options = next(
            (oc for name, _, oc in _ENGINE_DEFS if name == canonical), BaseCheckboxOptions
        )
        register_builtin(
            provider,
            "checkbox",
            alias,
            alias_factory,
            metadata={"options_class": canonical_options},
        )


try:
    register_checkbox_engines()
except Exception:
    logger.exception("Failed to register built-in checkbox engines")


__all__ = [
    "engine_name_for_options",
    "get_options_class_for_engine",
    "register_checkbox_engines",
]
