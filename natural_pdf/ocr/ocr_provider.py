"""OCR provider utilities wrapping EngineProvider registrations."""

from __future__ import annotations

import logging
import threading
from dataclasses import fields, is_dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Type, Union, cast

from natural_pdf.engine_provider import get_provider
from natural_pdf.engine_registry import register_builtin

from .engine import OCREngine
from .ocr_options import BaseOCROptions

logger = logging.getLogger(__name__)


EngineProviderValue = Union[Callable[[], OCREngine], Type[OCREngine], OCREngine]


def _classic_engine_entries() -> Dict[str, Any]:
    """Return unified-registry entries that carry an engine provider.

    These are the "classic" OCR engines (including platform-adaptive ones
    like paddlevl) that EngineProvider manages lifecycles for. VLM shorthand
    and generic VLM entries have no provider and are dispatched directly by
    :mod:`natural_pdf.ocr.unified_dispatch`.
    """
    from natural_pdf.ocr.unified_dispatch import get_registry

    return {name: entry for name, entry in get_registry().items() if entry.provider is not None}


def _instantiate_engine_provider(provider: EngineProviderValue) -> OCREngine:
    if isinstance(provider, OCREngine):
        return provider
    instance = provider()
    return cast(OCREngine, instance)


_engine_inference_locks: Dict[str, threading.Lock] = {}


def _create_engine_instance(engine_name: str) -> OCREngine:
    """Create a new OCR engine instance. EngineProvider handles caching."""
    engine_name = engine_name.lower()
    entries = _classic_engine_entries()
    if engine_name not in entries:
        raise RuntimeError(f"Unknown OCR engine '{engine_name}'. Available: {list(entries.keys())}")

    registry_entry = entries[engine_name]
    engine_instance = _instantiate_engine_provider(registry_entry.provider)
    if not engine_instance.is_available():
        hint = registry_entry.install_hint or f"pip install {engine_name}"
        raise RuntimeError(f"OCR engine '{engine_name}' is not available. Install it with: {hint}")
    return engine_instance


def register_ocr_engines(provider=None) -> None:
    for engine_name in _classic_engine_entries():

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


def cleanup_engine(engine_name: Optional[str] = None) -> int:
    """Clean up OCR engine instances from the provider cache and unified cache."""
    provider = get_provider()
    cleaned = 0
    if engine_name:
        targets = [engine_name.lower()]
    else:
        targets = list(_classic_engine_entries().keys())
        for capability in ("ocr", "ocr.apply", "ocr.extract"):
            targets.extend(provider.list(capability).get(capability, ()))
        targets = list(dict.fromkeys(targets))

    for target in targets:
        # Remove from provider's instance cache for all OCR capabilities
        for capability in ("ocr", "ocr.apply", "ocr.extract"):
            cleaned += provider.evict(capability, target)
        _engine_inference_locks.pop(target, None)

    # Also clear the unified dispatch cache
    try:
        from natural_pdf.ocr.unified_dispatch import get_engine_cache

        if engine_name:
            cleaned += get_engine_cache().invalidate(engine_name)
        else:
            cleaned += get_engine_cache().clear()
    except Exception:  # pragma: no cover
        pass

    return cleaned


def list_available_engines() -> List[str]:
    """List OCR engines that are available (dependencies installed)."""
    available = []
    for name, registry_entry in _classic_engine_entries().items():
        try:
            engine_instance = _instantiate_engine_provider(registry_entry.provider)
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
