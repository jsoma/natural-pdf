"""Registry helpers for checkbox detection engines."""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, Mapping, Optional, Tuple, Type, Union

from .base import CheckboxDetector
from .checkbox_options import CheckboxOptions, RTDETRCheckboxOptions

logger = logging.getLogger(__name__)


def _lazy_import_rtdetr_detector() -> Type[CheckboxDetector]:
    from .rtdetr import RTDETRCheckboxDetector

    return RTDETRCheckboxDetector


@dataclass(frozen=True)
class EngineRegistration:
    detector_factory: Callable[[], Type[CheckboxDetector]]
    options_type: Type[CheckboxOptions]


ENGINE_REGISTRY: Dict[str, EngineRegistration] = {
    "rtdetr": EngineRegistration(
        detector_factory=_lazy_import_rtdetr_detector,
        options_type=RTDETRCheckboxOptions,
    ),
    "wendys": EngineRegistration(
        detector_factory=_lazy_import_rtdetr_detector,
        options_type=RTDETRCheckboxOptions,
    ),
}

DEFAULT_ENGINE = "rtdetr"
_DETECTOR_CACHE: Dict[str, CheckboxDetector] = {}


def register_checkbox_engine(
    name: str,
    detector_factory: Callable[[], Type[CheckboxDetector]],
    options_type: Type[CheckboxOptions],
) -> None:
    ENGINE_REGISTRY[name] = EngineRegistration(detector_factory, options_type)
    logger.info("Registered checkbox detection engine '%s'", name)


def list_available_engines() -> list[str]:
    available: list[str] = []
    for name in ENGINE_REGISTRY:
        try:
            detector_cls = ENGINE_REGISTRY[name].detector_factory()
            if detector_cls.is_available():
                available.append(name)
        except Exception:
            continue
    return available


def _create_options(engine: str, **kwargs) -> CheckboxOptions:
    if engine not in ENGINE_REGISTRY:
        raise ValueError(
            f"Unknown checkbox detection engine: {engine}. "
            f"Available: {list(ENGINE_REGISTRY.keys())}"
        )
    registration = ENGINE_REGISTRY[engine]
    return registration.options_type(**kwargs)


def _engine_from_options(options: CheckboxOptions) -> str:
    for engine_name, registration in ENGINE_REGISTRY.items():
        if isinstance(options, registration.options_type):
            return engine_name
    return DEFAULT_ENGINE


def prepare_checkbox_options(
    engine: Optional[str],
    options: Optional[Union[CheckboxOptions, Mapping[str, Any]]],
    *,
    overrides: Optional[Mapping[str, Any]] = None,
) -> Tuple[str, CheckboxOptions]:
    override_kwargs: Dict[str, Any] = {}
    if overrides:
        override_kwargs.update(dict(overrides))

    if isinstance(options, CheckboxOptions):
        engine_name = engine or _engine_from_options(options)
        final_options = replace(options, **override_kwargs) if override_kwargs else options
        return engine_name, final_options

    engine_name = engine or DEFAULT_ENGINE
    base_kwargs: Dict[str, Any] = {}
    if isinstance(options, Mapping):
        base_kwargs.update(dict(options))
    base_kwargs.update(override_kwargs)

    final_options = _create_options(engine_name, **base_kwargs)
    return engine_name, final_options


def get_detector(engine: str) -> CheckboxDetector:
    if engine not in ENGINE_REGISTRY:
        raise ValueError(
            f"Unknown checkbox detection engine: {engine}. "
            f"Available: {list(ENGINE_REGISTRY.keys())}"
        )

    if engine not in _DETECTOR_CACHE:
        registration = ENGINE_REGISTRY[engine]
        detector_cls = registration.detector_factory()
        if not detector_cls.is_available():
            raise RuntimeError(
                f"Checkbox detection engine '{engine}' is not available. "
                "Please install required dependencies."
            )
        _DETECTOR_CACHE[engine] = detector_cls()
        logger.info("Initialized checkbox detector: %s", engine)
    return _DETECTOR_CACHE[engine]
