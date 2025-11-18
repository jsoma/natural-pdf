"""OCR provider utilities wrapping EngineProvider registrations."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, TypedDict, Union, cast

from PIL import Image

from natural_pdf.engine_provider import get_provider
from natural_pdf.engine_registry import register_builtin, register_ocr_engine
from natural_pdf.utils.locks import pdf_render_lock

from .engine import OCREngine
from .engine_doctr import DoctrOCREngine
from .engine_easyocr import EasyOCREngine
from .engine_paddle import PaddleOCREngine
from .engine_surya import SuryaOCREngine
from .ocr_options import (
    BaseOCROptions,
    DoctrOCROptions,
    EasyOCROptions,
    PaddleOCROptions,
    SuryaOCROptions,
)

logger = logging.getLogger(__name__)


EngineProviderValue = Union[Callable[[], OCREngine], Type[OCREngine], OCREngine]


class EngineRegistryEntry(TypedDict):
    provider: EngineProviderValue
    options_class: Optional[Type[BaseOCROptions]]


@dataclass
class OCRRunResult:
    """Container for OCR execution output."""

    results: List[Dict[str, Any]]
    image_size: Tuple[int, int]


ENGINE_REGISTRY: Dict[str, EngineRegistryEntry] = {
    "easyocr": {"provider": EasyOCREngine, "options_class": EasyOCROptions},
    "paddle": {"provider": PaddleOCREngine, "options_class": PaddleOCROptions},
    "surya": {"provider": SuryaOCREngine, "options_class": SuryaOCROptions},
    "doctr": {"provider": DoctrOCREngine, "options_class": DoctrOCROptions},
}


def _instantiate_engine_provider(provider: EngineProviderValue) -> OCREngine:
    if isinstance(provider, OCREngine):
        return provider
    instance = provider()
    return cast(OCREngine, instance)


_engine_instances: Dict[str, OCREngine] = {}
_engine_locks: Dict[str, threading.Lock] = {}
_engine_inference_locks: Dict[str, threading.Lock] = {}


def _get_engine_instance(engine_name: str) -> OCREngine:
    engine_name = engine_name.lower()
    if engine_name not in ENGINE_REGISTRY:
        raise RuntimeError(
            f"Unknown OCR engine '{engine_name}'. Available: {list(ENGINE_REGISTRY.keys())}"
        )

    if engine_name in _engine_instances:
        return _engine_instances[engine_name]

    lock = _engine_locks.setdefault(engine_name, threading.Lock())
    with lock:
        if engine_name in _engine_instances:
            return _engine_instances[engine_name]

        registry_entry = ENGINE_REGISTRY[engine_name]
        engine_instance = _instantiate_engine_provider(registry_entry["provider"])
        if not engine_instance.is_available():
            raise RuntimeError(
                f"OCR engine '{engine_name}' is not available. "
                "Install optional dependencies with 'pip install natural-pdf[ocr]'."
            )
        _engine_instances[engine_name] = engine_instance
        return engine_instance


def _get_engine_inference_lock(engine_name: str) -> threading.Lock:
    return _engine_inference_locks.setdefault(engine_name, threading.Lock())


def register_ocr_engines(provider=None) -> None:
    for engine_name in ENGINE_REGISTRY.keys():

        def factory(*, _engine_name=engine_name, **_opts):
            return _get_engine_instance(_engine_name)

        for capability in ("ocr", "ocr.apply", "ocr.extract"):
            register_builtin(provider, capability, engine_name, factory)


def normalize_ocr_options(
    options: Optional[Union[BaseOCROptions, Dict[str, Any]]]
) -> Optional[BaseOCROptions]:
    if options is None or isinstance(options, BaseOCROptions):
        return options
    if isinstance(options, dict):
        return BaseOCROptions(extra_args=dict(options))
    raise TypeError(
        "OCR options must be a BaseOCROptions instance, subclass thereof, or a mapping."
    )


def infer_engine_from_options(options: Optional[BaseOCROptions]) -> Optional[str]:
    if options is None:
        return None
    for name, entry in ENGINE_REGISTRY.items():
        opt_cls = entry.get("options_class")
        if opt_cls is not None and isinstance(options, opt_cls):
            return name
    return None


def resolve_ocr_engine_name(
    *,
    context: Any,
    requested: Optional[str] = None,
    options: Optional[BaseOCROptions] = None,
    scope: str = "page",
    capability: str = "ocr.extract",
) -> str:
    provider = get_provider()
    available = tuple(provider.list(capability).get(capability, ()))
    if not available:
        available = tuple(provider.list("ocr").get("ocr", ()))
    if not available:
        raise RuntimeError("No OCR engines are registered.")

    candidates = (
        _normalize_engine_name(requested),
        _normalize_engine_name(infer_engine_from_options(options)),
        _normalize_engine_name(_context_option(context, "ocr", "ocr_engine", scope)),
        _normalize_engine_name(_global_ocr_option("engine")),
    )

    for candidate in candidates:
        if candidate and candidate in available:
            return candidate

    if requested:
        raise LookupError(
            f"OCR engine '{requested}' is not registered. Available engines: {available}"
        )

    return available[0]


def _normalize_engine_name(name: Optional[Any]) -> Optional[str]:
    if isinstance(name, str):
        stripped = name.strip().lower()
        return stripped or None
    return None


def _context_option(host: Any, capability: str, key: str, scope: str) -> Any:
    ctx = getattr(host, "_context", None)
    if ctx is not None and hasattr(ctx, "get_option"):
        value = ctx.get_option(capability, key, host=host, scope=scope)
        if value is not None:
            return value

    getter = getattr(host, "get_config", None)
    if callable(getter):
        sentinel = object()
        try:
            value = getter(key, sentinel, scope=scope)
        except TypeError:
            try:
                value = getter(key, sentinel)
            except TypeError:
                value = sentinel
        if value is not sentinel:
            return value

    cfg = getattr(host, "_config", None)
    if isinstance(cfg, dict):
        return cfg.get(key)
    return None


def _global_ocr_option(attr: str) -> Any:
    try:
        import natural_pdf
    except Exception:  # pragma: no cover
        return None

    options = getattr(natural_pdf, "options", None)
    if options is None:
        return None
    section = getattr(options, "ocr", None)
    if section is None:
        return None
    return getattr(section, attr, None)


def _resolve_with_fallback(
    *,
    context: Any,
    scope: str,
    explicit: Optional[Any],
    capability: str,
    config_key: str,
    global_key: str,
) -> Optional[Any]:
    if explicit is not None:
        return explicit

    cfg_value = _context_option(context, capability=capability, key=config_key, scope=scope)
    if cfg_value is not None:
        return cfg_value

    global_value = _global_ocr_option(global_key)
    if isinstance(global_value, list):
        return list(global_value)
    return global_value


def resolve_ocr_languages(
    context: Any,
    languages: Optional[List[str]] = None,
    *,
    scope: str = "page",
) -> Optional[List[str]]:
    resolved = _resolve_with_fallback(
        context=context,
        scope=scope,
        explicit=languages,
        capability="ocr",
        config_key="ocr_languages",
        global_key="languages",
    )
    if resolved is None:
        return None
    if isinstance(resolved, list):
        return list(resolved)
    if isinstance(resolved, tuple):
        return list(resolved)
    if isinstance(resolved, set):
        return list(resolved)
    if isinstance(resolved, str):
        normalized = resolved.strip()
        return [normalized] if normalized else None
    return [resolved]


def resolve_ocr_min_confidence(
    context: Any,
    min_confidence: Optional[float] = None,
    *,
    scope: str = "page",
) -> Optional[float]:
    resolved = _resolve_with_fallback(
        context=context,
        scope=scope,
        explicit=min_confidence,
        capability="ocr",
        config_key="ocr_min_confidence",
        global_key="min_confidence",
    )
    if resolved is None:
        return None
    try:
        return float(resolved)
    except (TypeError, ValueError) as exc:  # pragma: no cover
        raise TypeError("min_confidence must be numeric") from exc


def resolve_ocr_device(
    context: Any,
    device: Optional[str] = None,
    *,
    scope: str = "page",
) -> Optional[str]:
    resolved = _resolve_with_fallback(
        context=context,
        scope=scope,
        explicit=device,
        config_key="ocr_device",
        global_key="device",
    )
    if resolved is None:
        return None
    if isinstance(resolved, str):
        normalized = resolved.strip()
        return normalized or None
    raise TypeError("device must be a string if provided")


def run_ocr_apply(
    *,
    target: Any,
    context: Any,
    engine_name: str,
    resolution: int,
    languages: Optional[List[str]] = None,
    min_confidence: Optional[float] = None,
    device: Optional[str] = None,
    detect_only: bool = False,
    options: Optional[BaseOCROptions] = None,
    render_kwargs: Optional[Dict[str, Any]] = None,
) -> OCRRunResult:
    return _run_ocr_capability(
        capability="ocr.apply",
        target=target,
        context=context,
        engine_name=engine_name,
        resolution=resolution,
        languages=languages,
        min_confidence=min_confidence,
        device=device,
        detect_only=detect_only,
        options=options,
        render_kwargs=render_kwargs,
    )


def run_ocr_extract(
    *,
    target: Any,
    context: Any,
    engine_name: str,
    resolution: int,
    languages: Optional[List[str]] = None,
    min_confidence: Optional[float] = None,
    device: Optional[str] = None,
    detect_only: bool = False,
    options: Optional[BaseOCROptions] = None,
    render_kwargs: Optional[Dict[str, Any]] = None,
) -> OCRRunResult:
    return _run_ocr_capability(
        capability="ocr.extract",
        target=target,
        context=context,
        engine_name=engine_name,
        resolution=resolution,
        languages=languages,
        min_confidence=min_confidence,
        device=device,
        detect_only=detect_only,
        options=options,
        render_kwargs=render_kwargs,
    )


def run_ocr_engine(
    images: Union[Image.Image, List[Image.Image]],
    *,
    context: Any,
    engine_name: str,
    languages: Optional[List[str]] = None,
    min_confidence: Optional[float] = None,
    device: Optional[str] = None,
    detect_only: bool = False,
    options: Optional[BaseOCROptions] = None,
) -> Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]]:
    """Backward compatible helper that executes OCR on provided image(s)."""

    provider = get_provider()
    try:
        engine = provider.get("ocr.extract", context=context, name=engine_name)
    except LookupError:
        engine = provider.get("ocr", context=context, name=engine_name)
    lock = _get_engine_inference_lock(engine_name)
    with lock:
        return engine.process_image(
            images=images,
            languages=languages,
            min_confidence=min_confidence,
            device=device,
            detect_only=detect_only,
            options=options,
        )


def _run_ocr_capability(
    *,
    capability: str,
    target: Any,
    context: Any,
    engine_name: str,
    resolution: int,
    languages: Optional[List[str]],
    min_confidence: Optional[float],
    device: Optional[str],
    detect_only: bool,
    options: Optional[BaseOCROptions],
    render_kwargs: Optional[Dict[str, Any]],
) -> OCRRunResult:
    image = _render_target(target, resolution=resolution, render_kwargs=render_kwargs or {})
    engine_output = _call_engine(
        capability=capability,
        context=context,
        engine_name=engine_name,
        image=image,
        languages=languages,
        min_confidence=min_confidence,
        device=device,
        detect_only=detect_only,
        options=options,
    )
    normalized = _normalize_engine_output(engine_output)
    return OCRRunResult(results=normalized, image_size=image.size)


def _render_target(target: Any, *, resolution: int, render_kwargs: Dict[str, Any]) -> Image.Image:
    render_fn = getattr(target, "render", None)
    if not callable(render_fn):
        raise AttributeError("Target object does not support rendering for OCR operations.")
    with pdf_render_lock:
        image = render_fn(resolution=resolution, **render_kwargs)
    if image is None:
        raise RuntimeError("Render call returned None for OCR input.")
    if not isinstance(image, Image.Image):
        raise TypeError(
            f"Expected render() to return a PIL Image, received {type(image).__name__} instead."
        )
    return image


def _call_engine(
    *,
    capability: str,
    context: Any,
    engine_name: str,
    image: Image.Image,
    languages: Optional[List[str]],
    min_confidence: Optional[float],
    device: Optional[str],
    detect_only: bool,
    options: Optional[BaseOCROptions],
) -> Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]]:
    provider = get_provider()
    engine = provider.get(capability, context=context, name=engine_name)

    lock = _get_engine_inference_lock(engine_name)
    with lock:
        return engine.process_image(
            image,
            languages=languages,
            min_confidence=min_confidence,
            device=device,
            detect_only=detect_only,
            options=options,
        )


def _normalize_engine_output(
    payload: Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]]
) -> List[Dict[str, Any]]:
    if isinstance(payload, list):
        if payload and isinstance(payload[0], list):
            # Single image – engines sometimes wrap in an extra list.
            first = payload[0]
            if isinstance(first, list):
                return cast(List[Dict[str, Any]], first)
        elif payload and isinstance(payload[0], dict):
            return cast(List[Dict[str, Any]], payload)
        elif not payload:
            return []
    raise TypeError(f"OCR engine returned unsupported result type: {type(payload).__name__}")


def cleanup_engine(engine_name: Optional[str] = None) -> int:
    cleaned = 0
    targets = [engine_name.lower()] if engine_name else list(_engine_instances.keys())
    for target in targets:
        engine = _engine_instances.pop(target, None)
        if engine is None:
            continue
        cleanup_fn = getattr(engine, "cleanup", None)
        if callable(cleanup_fn):
            try:
                cleanup_fn()
            except Exception:  # pragma: no cover
                logger.debug("Cleanup for OCR engine %s failed", target)
        _engine_locks.pop(target, None)
        _engine_inference_locks.pop(target, None)
        cleaned += 1
    return cleaned


def list_available_engines() -> List[str]:
    available = []
    for name in ENGINE_REGISTRY:
        try:
            engine = _get_engine_instance(name)
            if engine:
                available.append(name)
        except Exception:
            continue
    return available


try:
    register_ocr_engines()
except Exception:  # pragma: no cover
    logger.exception("Failed to register built-in OCR engines")


__all__ = [
    "run_ocr_apply",
    "run_ocr_extract",
    "run_ocr_engine",
    "OCRRunResult",
    "register_ocr_engines",
    "cleanup_engine",
    "list_available_engines",
    "normalize_ocr_options",
    "infer_engine_from_options",
    "resolve_ocr_engine_name",
    "resolve_ocr_languages",
    "resolve_ocr_min_confidence",
    "resolve_ocr_device",
]
