"""Registration helpers for checkbox detection engines.

All shared behavior (options-class lookup, instance creation with install
hints, provider registration) lives in
:class:`natural_pdf.engine_registry.manager_builder.EngineManagerSpec`;
this module supplies the checkbox-specific data.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Type, cast

from natural_pdf.engine_registry.manager_builder import EngineManagerSpec

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
# Engine data (name -> lazy class factory + options class)
# ---------------------------------------------------------------------------

_ENGINE_DEFS: list[tuple[str, Any, type]] = [
    ("vector", _lazy_import_vector_detector, VectorCheckboxOptions),
    ("default", _lazy_import_default_detector, DefaultCheckboxOptions),
    ("onnx", _lazy_import_onnx_detector, OnnxCheckboxOptions),
    ("vlm", _lazy_import_vlm_detector, VLMCheckboxOptions),
]

_MANAGER = EngineManagerSpec(
    capability="checkbox",
    label="Checkbox",
    engine_defs=_ENGINE_DEFS,
    install_hints={
        "onnx": "pip install onnxruntime huggingface_hub",
        "vlm": "pip install openai",
    },
    fallback_options_class=BaseCheckboxOptions,
    logger=logger,
)


# ---------------------------------------------------------------------------
# Module-level API (kept for existing importers)
# ---------------------------------------------------------------------------


def engine_name_for_options(options: BaseCheckboxOptions) -> Optional[str]:
    """Return the engine name whose options_class matches *options*, or None."""
    return _MANAGER.engine_name_for_options(options)


def get_options_class_for_engine(name: str) -> Optional[type]:
    """Return the options class registered for *name*, or None."""
    return _MANAGER.get_options_class_for_engine(name)


def _create_engine_instance(engine_name: str) -> CheckboxDetector:
    """Create a new checkbox engine instance. EngineProvider handles caching."""
    return cast(CheckboxDetector, _MANAGER.create_engine_instance(engine_name))


def register_checkbox_engines(provider=None) -> None:
    """Register all built-in checkbox engines with the EngineProvider."""
    _MANAGER.register_engines(provider)


# Register at import time so engines are discoverable immediately.
_MANAGER.register_at_import()


__all__ = [
    "engine_name_for_options",
    "get_options_class_for_engine",
    "register_checkbox_engines",
]
