"""OCR provider utilities wrapping EngineProvider registrations."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, fields, is_dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Type, TypedDict, Union, cast

from PIL import Image

from natural_pdf.engine_provider import get_provider
from natural_pdf.engine_registry import register_builtin, register_ocr_engine
from natural_pdf.utils.locks import pdf_render_lock

from .engine import OCREngine
from .engine_chandra import ChandraOCREngine
from .engine_doctr import DoctrOCREngine
from .engine_easyocr import EasyOCREngine
from .engine_paddle import PaddleOCREngine
from .engine_paddleocr_vl import PaddleOCRVLEngine
from .engine_rapidocr import RapidOCREngine
from .engine_surya import SuryaOCREngine
from .ocr_options import (
    BaseOCROptions,
    ChandraOCROptions,
    DoctrOCROptions,
    EasyOCROptions,
    PaddleOCROptions,
    PaddleOCRVLOptions,
    RapidOCROptions,
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
    "chandra2": {"provider": ChandraOCREngine, "options_class": ChandraOCROptions},
    "doctr": {"provider": DoctrOCREngine, "options_class": DoctrOCROptions},
    "rapidocr": {"provider": RapidOCREngine, "options_class": RapidOCROptions},
    "paddlevl": {"provider": PaddleOCRVLEngine, "options_class": PaddleOCRVLOptions},
}


def _instantiate_engine_provider(provider: EngineProviderValue) -> OCREngine:
    if isinstance(provider, OCREngine):
        return provider
    instance = provider()
    return cast(OCREngine, instance)


_engine_inference_locks: Dict[str, threading.Lock] = {}


def _create_engine_instance(engine_name: str) -> OCREngine:
    """Create a new OCR engine instance. EngineProvider handles caching."""
    engine_name = engine_name.lower()
    if engine_name not in ENGINE_REGISTRY:
        raise RuntimeError(
            f"Unknown OCR engine '{engine_name}'. Available: {list(ENGINE_REGISTRY.keys())}"
        )

    registry_entry = ENGINE_REGISTRY[engine_name]
    engine_instance = _instantiate_engine_provider(registry_entry["provider"])
    if not engine_instance.is_available():
        install_hints = {
            "easyocr": "pip install easyocr",
            "paddle": "pip install paddleocr",
            "surya": "pip install surya-ocr",
            "chandra2": "pip install chandra-ocr[hf]",
            "doctr": "pip install python-doctr",
            "rapidocr": "pip install rapidocr",
            "paddlevl": "pip install paddleocr",
        }
        hint = install_hints.get(engine_name, f"pip install {engine_name}")
        raise RuntimeError(f"OCR engine '{engine_name}' is not available. Install it with: {hint}")
    return engine_instance


def _get_engine_inference_lock(engine_name: str) -> threading.Lock:
    return _engine_inference_locks.setdefault(engine_name, threading.Lock())


def register_ocr_engines(provider=None) -> None:
    for engine_name in ENGINE_REGISTRY.keys():

        def factory(*, _engine_name=engine_name, **_opts):
            # EngineProvider handles caching - we just create the instance
            return _create_engine_instance(_engine_name)

        for capability in ("ocr", "ocr.apply", "ocr.extract"):
            register_builtin(provider, capability, engine_name, factory)


def get_ocr_options_class(engine_name: str) -> Optional[Type[BaseOCROptions]]:
    """Return the registered option class for *engine_name*, if it has one."""
    normalized_name = _normalize_engine_name(engine_name)
    if normalized_name is None:
        return None

    entry = ENGINE_REGISTRY.get(normalized_name)
    if entry is not None:
        return entry.get("options_class")

    try:
        from natural_pdf.ocr.unified_dispatch import get_registry

        unified_entry = get_registry().get(normalized_name)
        return getattr(unified_entry, "options_class", None)
    except Exception:
        return None


def normalize_ocr_options(
    options: Optional[Union[BaseOCROptions, Mapping[str, Any]]],
    *,
    engine_name: Optional[str] = None,
) -> Optional[BaseOCROptions]:
    """Normalize OCR options, constructing the registered engine's option type.

    A mapping is meaningful only relative to an OCR engine.  Callers that do
    not yet know the engine may temporarily receive a ``BaseOCROptions``
    wrapper; dispatch must call this function again with the resolved engine.
    This preserves legacy resolution order while ensuring mappings never reach
    an adapter as an incompatible base instance.
    """
    if options is None:
        return None

    if engine_name is None:
        if isinstance(options, Mapping):
            return BaseOCROptions(extra_args=dict(options))
        if isinstance(options, BaseOCROptions):
            return options
        raise TypeError(
            "OCR options must be a BaseOCROptions instance, subclass thereof, or a mapping."
        )

    options_class = get_ocr_options_class(engine_name)
    if options_class is None:
        if isinstance(options, Mapping) or (
            type(options) is BaseOCROptions and bool(options.extra_args)
        ):
            raise TypeError(
                f"OCR engine '{engine_name}' does not declare an options class; "
                "it cannot accept mapping options. Register an engine-specific options class."
            )
        return cast(Optional[BaseOCROptions], options)

    if isinstance(options, options_class):
        return options

    payload: Optional[Dict[str, Any]] = None
    if isinstance(options, Mapping):
        payload = dict(options)
    elif type(options) is BaseOCROptions:
        # ``normalize_ocr_options(mapping)`` is retained for callers which
        # resolve an engine afterwards. Reify that deferred mapping now.
        payload = dict(options.extra_args)
    elif isinstance(options, BaseOCROptions):
        raise TypeError(
            f"OCR engine '{engine_name}' requires {options_class.__name__}; got "
            f"{type(options).__name__}."
        )
    else:
        raise TypeError(
            "OCR options must be a BaseOCROptions instance, subclass thereof, or a mapping."
        )

    if not is_dataclass(options_class):  # pragma: no cover - registry contract
        raise TypeError(
            f"OCR engine '{engine_name}' options class {options_class.__name__} must be a dataclass."
        )
    allowed = {item.name for item in fields(options_class) if item.init}
    unknown = sorted(set(payload) - allowed)
    if unknown:
        available = ", ".join(sorted(allowed))
        raise TypeError(
            f"Unsupported option(s) for OCR engine '{engine_name}': {', '.join(unknown)}. "
            f"Accepted options: {available}."
        )
    try:
        return options_class(**payload)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"Invalid options for OCR engine '{engine_name}' ({options_class.__name__}): {exc}"
        ) from exc


def infer_engine_from_options(options: Optional[BaseOCROptions]) -> Optional[str]:
    if options is None:
        return None
    for name, entry in ENGINE_REGISTRY.items():
        opt_cls = entry.get("options_class")
        if opt_cls is not None and isinstance(options, opt_cls):
            return name
    try:
        from natural_pdf.ocr.unified_dispatch import list_engines
    except Exception:
        return None

    for name, entry in list_engines().items():
        opt_cls = getattr(entry, "options_class", None)
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

    # Also check the unified registry (includes VLM engines)
    try:
        from natural_pdf.ocr.unified_dispatch import get_registry

        unified_names = set(get_registry().keys())
    except Exception:
        unified_names = set()

    all_available = set(available) | unified_names

    # If an engine was explicitly requested, it must exist — fail fast.
    normalized_requested = _normalize_engine_name(requested)
    if normalized_requested is not None:
        if normalized_requested in all_available:
            return normalized_requested
        raise LookupError(
            f"OCR engine '{requested}' is not registered. "
            f"Available engines: {sorted(all_available)}"
        )

    # Otherwise, try inference from options, context, and global defaults.
    candidates = (
        _normalize_engine_name(infer_engine_from_options(options)),
        _normalize_engine_name(_context_option(context, "ocr", "ocr_engine", scope)),
        _normalize_engine_name(_global_ocr_option("engine")),
    )

    for candidate in candidates:
        if candidate and candidate in all_available:
            return candidate

    return available[0] if available else sorted(all_available)[0]


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
        capability="ocr",
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

    options = normalize_ocr_options(options, engine_name=engine_name)
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
    options = normalize_ocr_options(options, engine_name=engine_name)

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
    payload: Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]],
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
    """Clean up OCR engine instances from the provider cache and unified cache."""
    provider = get_provider()
    cleaned = 0
    targets = [engine_name.lower()] if engine_name else list(ENGINE_REGISTRY.keys())

    for target in targets:
        # Remove from provider's instance cache for all OCR capabilities
        for capability in ("ocr", "ocr.apply", "ocr.extract"):
            key = (capability, target)
            engine = provider._instances.pop(key, None)
            if engine is not None:
                cleanup_fn = getattr(engine, "cleanup", None)
                if callable(cleanup_fn):
                    try:
                        cleanup_fn()
                    except Exception:  # pragma: no cover
                        logger.debug("Cleanup for OCR engine %s failed", target)
                cleaned += 1
        _engine_inference_locks.pop(target, None)

    # Also clear the unified dispatch cache
    try:
        from natural_pdf.ocr.unified_dispatch import get_engine_cache

        cleaned += get_engine_cache().clear()
    except Exception:  # pragma: no cover
        pass

    return cleaned


def list_available_engines() -> List[str]:
    """List OCR engines that are available (dependencies installed)."""
    available = []
    for name in ENGINE_REGISTRY:
        try:
            registry_entry = ENGINE_REGISTRY[name]
            engine_instance = _instantiate_engine_provider(registry_entry["provider"])
            if engine_instance.is_available():
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
    "get_ocr_options_class",
    "infer_engine_from_options",
    "resolve_ocr_engine_name",
    "resolve_ocr_languages",
    "resolve_ocr_min_confidence",
    "resolve_ocr_device",
]
