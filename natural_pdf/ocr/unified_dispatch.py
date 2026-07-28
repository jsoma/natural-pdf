"""Unified OCR dispatch — single source of truth for all OCR engines.

Provides one registry (classic + VLM engines), one dispatch function
(:func:`run_ocr`), and an LRU cache for classic engine instances.
"""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from inspect import Parameter, signature
from typing import Any, Callable, Dict, Hashable, List, Optional, Tuple, Type

from PIL import Image

from natural_pdf.ocr.ocr_options import BaseOCROptions
from natural_pdf.utils.locks import pdf_render_lock

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Registry data structures
# ---------------------------------------------------------------------------


@dataclass
class EngineEntry:
    """Metadata for a registered OCR engine."""

    engine_type: str  # "classic" | "classic_provider" | "vlm_shorthand" | "vlm_generic"

    # Classic engines:
    provider: Optional[Any] = None  # OCREngine class or factory
    # Public custom providers accept language/device constructor variants.
    provider_init_options: bool = False
    options_class: Optional[Type[BaseOCROptions]] = None

    # VLM shorthand engines (dots, glm_ocr, chandra):
    model_resolver: Optional[Callable[[], str]] = None

    # All VLM engines:
    vlm_family: Optional[str] = None  # "dots_mocr", "glm_ocr", "chandra", etc.

    # Locking:
    needs_gpu_lock: bool = True  # False for known remote-API-only engines

    # Install hint for error messages:
    install_hint: Optional[str] = None

    # Explicit stable namespace for disk-cached results from custom engines.
    # Built-ins use their known engine identity; plugin callables must opt in.
    cache_namespace: Optional[str] = None


def _is_apple_silicon() -> bool:
    """Check if running on Apple Silicon."""
    import platform

    return platform.machine() == "arm64" and platform.system() == "Darwin"


def _resolve_paddlevl_model() -> str:
    """Pick the best PaddleOCR-VL model for the current platform."""
    if _is_apple_silicon():
        return "mlx-community/PaddleOCR-VL-1.5-4bit"
    return "PaddlePaddle/PaddleOCR-VL-1.5"


_PADDLEVL_PROMPT = "OCR:"


def _build_registry() -> Dict[str, EngineEntry]:
    """Build the unified engine registry from classic + VLM engines.

    Imports are deferred to avoid circular imports and heavy deps at
    module load time.
    """
    from natural_pdf.ocr.engine_chandra import ChandraOCREngine
    from natural_pdf.ocr.engine_doctr import DoctrOCREngine
    from natural_pdf.ocr.engine_easyocr import EasyOCREngine
    from natural_pdf.ocr.engine_paddle import PaddleOCREngine
    from natural_pdf.ocr.engine_paddleocr_vl import PaddleOCRVLEngine
    from natural_pdf.ocr.engine_rapidocr import RapidOCREngine
    from natural_pdf.ocr.engine_surya import SuryaOCREngine
    from natural_pdf.ocr.ocr_options import (
        ChandraOCROptions,
        DoctrOCROptions,
        EasyOCROptions,
        PaddleOCROptions,
        PaddleOCRVLOptions,
        RapidOCROptions,
        SuryaOCROptions,
    )
    from natural_pdf.ocr.vlm_ocr import (
        resolve_chandra_model,
        resolve_dots_model,
        resolve_glm_ocr_model,
    )

    registry = {
        # Classic engines
        "easyocr": EngineEntry(
            engine_type="classic",
            provider=EasyOCREngine,
            options_class=EasyOCROptions,
            install_hint="pip install easyocr",
        ),
        "rapidocr": EngineEntry(
            engine_type="classic",
            provider=RapidOCREngine,
            options_class=RapidOCROptions,
            install_hint="pip install rapidocr",
        ),
        "surya": EngineEntry(
            engine_type="classic",
            provider=SuryaOCREngine,
            options_class=SuryaOCROptions,
            install_hint="pip install surya-ocr",
        ),
        "paddle": EngineEntry(
            engine_type="classic",
            provider=PaddleOCREngine,
            options_class=PaddleOCROptions,
            install_hint="pip install paddleocr",
        ),
        "doctr": EngineEntry(
            engine_type="classic",
            provider=DoctrOCREngine,
            options_class=DoctrOCROptions,
            install_hint="pip install python-doctr",
        ),
        "chandra2": EngineEntry(  # Chandra v0.2 — larger model via pip package
            engine_type="classic",
            provider=ChandraOCREngine,
            options_class=ChandraOCROptions,
            install_hint="pip install chandra-ocr[hf]",
        ),
        "paddlevl": EngineEntry(
            engine_type="auto_platform",  # MLX on Apple Silicon, pip package elsewhere
            provider=PaddleOCRVLEngine,
            options_class=PaddleOCRVLOptions,
            install_hint="pip install paddleocr",
            model_resolver=_resolve_paddlevl_model,
            vlm_family="paddlevl",
        ),
        # VLM shorthand engines
        "dots": EngineEntry(
            engine_type="vlm_shorthand",
            model_resolver=resolve_dots_model,
            vlm_family="dots_mocr",
        ),
        "glm_ocr": EngineEntry(
            engine_type="vlm_shorthand",
            model_resolver=resolve_glm_ocr_model,
            vlm_family="glm_ocr",
        ),
        "chandra": EngineEntry(  # Chandra v0.1 — smaller/faster, MLX on Apple Silicon
            engine_type="vlm_shorthand",
            model_resolver=resolve_chandra_model,
            vlm_family="chandra",
        ),
        # VLM generic — requires model= and/or client=
        "vlm": EngineEntry(
            engine_type="vlm_generic",
        ),
    }
    for name, entry in registry.items():
        entry.cache_namespace = f"builtin:{name}"
    return registry


_registry: Optional[Dict[str, EngineEntry]] = None
_registry_lock = threading.Lock()


def get_registry() -> Dict[str, EngineEntry]:
    """Return the unified engine registry, building it on first access."""
    global _registry
    if _registry is None:
        with _registry_lock:
            if _registry is None:
                _registry = _build_registry()
    return _registry


def list_engines() -> Dict[str, EngineEntry]:
    """Return all registered engines with their metadata."""
    return dict(get_registry())


def register_engine(name: str, entry: EngineEntry, *, replace: bool = True) -> bool:
    """Register an engine in the unified registry.

    Useful for tests and plugins that add custom engines.

    Returns ``False`` without changing the registry when ``replace=False`` and
    the normalized name already exists; otherwise returns ``True``.
    """
    registry = get_registry()
    normalized = name.strip().lower()
    with _registry_lock:
        if normalized in registry and not replace:
            return False
        registry[normalized] = entry
    _engine_cache.invalidate(normalized)
    return True


def _capture_registration(
    engine_name: str,
    entry: EngineEntry,
) -> tuple[int, Callable[[], bool]]:
    """Capture the registry entry identity and a publication validator."""

    normalized = engine_name.strip().lower()

    def is_current() -> bool:
        current_registry = get_registry()
        with _registry_lock:
            return current_registry.get(normalized) is entry

    return id(entry), is_current


def _instantiate_provider(provider: Any, *, context: Any = None, options: Any = None) -> Any:
    """Instantiate a class/factory registered for a classic OCR engine."""

    if provider is None:
        raise RuntimeError("OCR engine registry entry is missing a provider.")
    if not callable(provider):
        return provider

    try:
        params = signature(provider).parameters
    except (TypeError, ValueError):
        return provider()

    accepts_kwargs = any(param.kind == Parameter.VAR_KEYWORD for param in params.values())
    kwargs: Dict[str, Any] = {}
    if accepts_kwargs or "context" in params:
        kwargs["context"] = context
    if accepts_kwargs or "options" in params:
        kwargs["options"] = options
    return provider(**kwargs)


# ---------------------------------------------------------------------------
# LRU engine cache for classic engines
# ---------------------------------------------------------------------------


@dataclass
class _CacheLease:
    engine: Any
    count: int = 0
    pending_cleanup: bool = False


class EngineCache:
    """Thread-safe leased LRU cache for classic OCR engine instances.

    :meth:`checkout` keeps cached engines alive through inference. Eviction
    detaches entries immediately, defers cleanup until their final lease exits,
    and always invokes hooks outside the cache lock. :meth:`get_or_create`
    remains as the compatible lower-level, unleased API.
    """

    def __init__(self, maxsize: int = 4):
        self._cache: OrderedDict[tuple, Any] = OrderedDict()
        self._leases: Dict[int, _CacheLease] = {}
        self._lock = threading.RLock()
        self._maxsize = max(1, maxsize)

    @property
    def maxsize(self) -> int:
        with self._lock:
            return self._maxsize

    @maxsize.setter
    def maxsize(self, value: int) -> None:
        with self._lock:
            self._maxsize = max(1, value)
            _, cleanup = self._evict_to_capacity_locked()
        self._run_cleanups(cleanup)

    def get_or_create(
        self,
        engine_name: str,
        languages: Tuple[str, ...],
        device: str,
        init_key: str,
        factory: Callable[[], Any],
        provider_identity: Optional[int] = None,
        *,
        registration_identity: Optional[Hashable] = None,
        registration_is_current: Optional[Callable[[], bool]] = None,
    ) -> Any:
        """Get or create an engine without leasing the returned instance."""

        engine, _caller_owned, _leased = self._acquire(
            engine_name=engine_name,
            languages=languages,
            device=device,
            init_key=init_key,
            factory=factory,
            provider_identity=provider_identity,
            registration_identity=registration_identity,
            registration_is_current=registration_is_current,
            lease_checkout=False,
        )
        return engine

    @contextmanager
    def checkout(
        self,
        engine_name: str,
        languages: Tuple[str, ...],
        device: str,
        init_key: str,
        factory: Callable[[], Any],
        provider_identity: Optional[int] = None,
        *,
        registration_identity: Optional[Hashable] = None,
        registration_is_current: Optional[Callable[[], bool]] = None,
    ):
        """Yield a scoped engine lease and clean detached instances on exit."""

        engine, caller_owned, leased = self._acquire(
            engine_name=engine_name,
            languages=languages,
            device=device,
            init_key=init_key,
            factory=factory,
            provider_identity=provider_identity,
            registration_identity=registration_identity,
            registration_is_current=registration_is_current,
            lease_checkout=True,
        )
        try:
            yield engine
        finally:
            if leased:
                self._run_cleanups(self._release_lease(engine))
            elif caller_owned:
                self._run_cleanups([engine])

    def _acquire(
        self,
        *,
        engine_name: str,
        languages: Tuple[str, ...],
        device: str,
        init_key: str,
        factory: Callable[[], Any],
        provider_identity: Optional[int],
        registration_identity: Optional[Hashable],
        registration_is_current: Optional[Callable[[], bool]],
        lease_checkout: bool,
    ) -> tuple[Any, bool, bool]:
        normalized = engine_name.strip().lower()
        key = (
            normalized,
            languages,
            device,
            init_key,
            provider_identity,
            registration_identity,
        )
        with self._lock:
            cached = self._cache.get(key)
            if cached is not None:
                self._cache.move_to_end(key)
                if lease_checkout:
                    self._acquire_lease_locked(cached)
                return cached, False, lease_checkout

        # Heavy construction deliberately happens outside the cache lock.
        engine = factory()
        cleanup: list[Any] = []
        with self._lock:
            cached = self._cache.get(key)
            if cached is not None:
                self._cache.move_to_end(key)
                if lease_checkout:
                    self._acquire_lease_locked(cached)
                cleanup.append(engine)
                result = (cached, False, lease_checkout)
            elif registration_is_current is not None and not registration_is_current():
                # The factory belongs to a registration that was replaced while
                # construction was in flight. It may serve this one checkout,
                # but must never repopulate the cache after invalidation.
                result = (engine, True, False)
            else:
                self._cache[key] = engine
                if lease_checkout:
                    self._acquire_lease_locked(engine)
                _, capacity_cleanup = self._evict_to_capacity_locked()
                cleanup.extend(capacity_cleanup)
                result = (engine, False, lease_checkout)

        self._run_cleanups(cleanup)
        return result

    def clear(self) -> int:
        """Detach every cache entry and clean it after active leases end."""

        with self._lock:
            count, cleanup = self._detach_keys_locked(list(self._cache))
        self._run_cleanups(cleanup)
        return count

    def invalidate(self, engine_name: str) -> int:
        """Detach cached instances for one normalized registration name."""

        normalized = engine_name.strip().lower()
        with self._lock:
            keys = [key for key in self._cache if key[0] == normalized]
            count, cleanup = self._detach_keys_locked(keys)
        self._run_cleanups(cleanup)
        return count

    def _evict_to_capacity_locked(self) -> tuple[int, List[Any]]:
        """Detach least-recently-used entries to capacity. Lock required."""

        excess = max(0, len(self._cache) - self._maxsize)
        return self._detach_keys_locked(list(self._cache)[:excess])

    def _detach_keys_locked(
        self,
        keys: List[tuple],
    ) -> tuple[int, List[Any]]:
        records = [self._cache.pop(key) for key in keys]
        live_ids = {id(engine) for engine in self._cache.values()}
        cleanup: list[Any] = []
        seen: set[int] = set()
        for engine in records:
            identity = id(engine)
            if identity in live_ids or identity in seen:
                continue
            seen.add(identity)
            lease = self._leases.get(identity)
            if lease is not None and lease.engine is engine and lease.count:
                lease.pending_cleanup = True
            else:
                cleanup.append(engine)
        return len(keys), cleanup

    def _acquire_lease_locked(self, engine: Any) -> None:
        identity = id(engine)
        lease = self._leases.get(identity)
        if lease is None:
            lease = _CacheLease(engine=engine)
            self._leases[identity] = lease
        elif lease.engine is not engine:  # pragma: no cover - strong ref prevents id reuse
            raise RuntimeError("OCR engine lease identity collision")
        lease.count += 1

    def _release_lease(self, engine: Any) -> List[Any]:
        with self._lock:
            identity = id(engine)
            lease = self._leases.get(identity)
            if lease is None or lease.engine is not engine or lease.count < 1:
                logger.error("OCR engine cache lease accounting mismatch")
                return []
            lease.count -= 1
            if lease.count:
                return []
            self._leases.pop(identity, None)
            if lease.pending_cleanup and not self._engine_is_cached_locked(engine):
                return [engine]
            return []

    def _engine_is_cached_locked(self, engine: Any) -> bool:
        return any(cached is engine for cached in self._cache.values())

    def _run_cleanups(self, engines: List[Any]) -> None:
        for engine in engines:
            self._cleanup_engine(engine)

    @staticmethod
    def _cleanup_engine(engine: Any) -> None:
        """Call an engine cleanup hook; callers must not hold cache locks."""

        cleanup_fn = getattr(engine, "cleanup", None)
        if not callable(cleanup_fn):
            cleanup_fn = getattr(engine, "close", None)
        if callable(cleanup_fn):
            try:
                cleanup_fn()
            except Exception:
                logger.debug("Cleanup failed for evicted engine", exc_info=True)


_engine_cache = EngineCache(maxsize=4)


def get_engine_cache() -> EngineCache:
    """Return the module-level engine cache."""
    return _engine_cache


# ---------------------------------------------------------------------------
# Inference locks
# ---------------------------------------------------------------------------

_inference_locks: Dict[str, threading.Lock] = {}
_inference_locks_guard = threading.Lock()


def _get_inference_lock(engine_name: str) -> threading.Lock:
    """Get or create a per-engine inference lock."""
    with _inference_locks_guard:
        return _inference_locks.setdefault(engine_name, threading.Lock())


# ---------------------------------------------------------------------------
# Image rendering helper
# ---------------------------------------------------------------------------


def _render_target(
    target: Any,
    resolution: int,
    render_kwargs: Dict[str, Any],
) -> Image.Image:
    """Render a Page or Region to a PIL Image."""
    render_fn = getattr(target, "render", None)
    if not callable(render_fn):
        raise AttributeError("Target object does not support render() for OCR operations.")
    with pdf_render_lock:
        image = render_fn(resolution=resolution, **render_kwargs)
    if image is None:
        raise RuntimeError("Render call returned None for OCR input.")
    if not isinstance(image, Image.Image):
        raise TypeError(
            f"Expected render() to return a PIL Image, received {type(image).__name__} instead."
        )
    return image


# ---------------------------------------------------------------------------
# OCR result container
# ---------------------------------------------------------------------------


@dataclass
class OCRRunResult:
    """Container for OCR execution output."""

    results: List[Dict[str, Any]]
    image_size: Tuple[int, int]
    engine_type: str = "classic"  # "classic" | "vlm"


# ---------------------------------------------------------------------------
# Unified dispatch
# ---------------------------------------------------------------------------


def run_ocr(
    *,
    target: Any,
    engine_name: str,
    resolution: int,
    # Shared params:
    languages: Optional[List[str]] = None,
    min_confidence: Optional[float] = None,
    device: Optional[str] = None,
    render_kwargs: Optional[Dict[str, Any]] = None,
    # Classic engine params:
    context: Any = None,
    detect_only: bool = False,
    options: Optional[BaseOCROptions] = None,
    # VLM engine params:
    model: Optional[str] = None,
    client: Optional[Any] = None,
    prompt: Optional[str] = None,
    instructions: Optional[str] = None,
    max_new_tokens: Optional[int] = None,
    layout: Optional[bool | str] = None,
    preserve_markup: bool = False,
) -> OCRRunResult:
    """Unified OCR dispatch — single entry point for all engines.

    Renders the target once, routes to the appropriate engine backend,
    and returns results in **image pixel coordinates**. The caller is
    responsible for scaling to PDF coordinates and creating elements.

    Args:
        target: Page or Region with a ``render()`` method.
        engine_name: Engine identifier (e.g. ``"easyocr"``, ``"dots"``, ``"vlm"``).
        resolution: DPI for rendering.
        languages: Language codes for OCR.
        min_confidence: Minimum confidence filter (applied by caller, not here).
        device: Compute device (``"cpu"``, ``"cuda"``, ``"mps"``).
        render_kwargs: Extra kwargs for ``target.render()``.
        context: Context object for engine resolution (usually same as target).
        detect_only: Detect text regions without recognition (classic only).
        options: Engine-specific options (classic only).
        model: VLM model name (VLM only).
        client: OpenAI-compatible client (VLM only).
        prompt: Custom VLM prompt (VLM only).
        instructions: Additional VLM instructions (VLM only).
        max_new_tokens: Max generation tokens (VLM only).
        preserve_markup: Preserve raw markup returned by VLM OCR instead of
            normalizing HTML fragments to plain text (VLM layout mode only).

    Returns:
        :class:`OCRRunResult` with results in image pixel coordinates.
    """
    registry = get_registry()
    engine_key = engine_name.strip().lower()

    entry = registry.get(engine_key)
    if entry is None:
        # Check if the engine is registered in EngineProvider (e.g. plugins, tests)
        from natural_pdf.engine_provider import get_provider

        provider = get_provider()
        provider_engines = set()
        for cap in ("ocr", "ocr.apply", "ocr.extract"):
            provider_engines.update(provider.list(cap).get(cap, ()))
        if engine_key in provider_engines:
            # Treat as a classic engine via provider
            entry = EngineEntry(engine_type="classic_provider")
        else:
            available = sorted(set(registry.keys()) | provider_engines)
            raise LookupError(f"Unknown OCR engine {engine_name!r}. Available engines: {available}")

    effective_render_kwargs = dict(render_kwargs or {})

    # Render image once
    image = _render_target(target, resolution, effective_render_kwargs)

    # auto_platform: Apple Silicon → VLM with layout, otherwise → classic pip package
    if entry.engine_type == "auto_platform":
        if _is_apple_silicon():
            return _run_vlm(
                image=image,
                entry=entry,
                engine_name=engine_key,
                model=model or (entry.model_resolver() if entry.model_resolver else None),
                client=client,
                prompt=prompt or _PADDLEVL_PROMPT,
                instructions=instructions,
                max_new_tokens=max_new_tokens,
                languages=languages,
                layout=True if layout is None else layout,
                preserve_markup=preserve_markup,
            )
        else:
            # Fold VLM generation params into PaddleOCRVLOptions for the classic path
            from natural_pdf.ocr.ocr_provider import normalize_ocr_options

            options = normalize_ocr_options(options, engine_name=engine_key)
            if max_new_tokens is not None:
                from natural_pdf.ocr.ocr_options import PaddleOCRVLOptions

                if options is None:
                    options = PaddleOCRVLOptions(max_new_tokens=max_new_tokens)
                elif isinstance(options, PaddleOCRVLOptions) and options.max_new_tokens is None:
                    options.max_new_tokens = max_new_tokens
            return _run_classic(
                image=image,
                engine_name=engine_key,
                entry=entry,
                languages=languages,
                min_confidence=min_confidence,
                device=device,
                detect_only=detect_only,
                options=options,
                context=context,
            )

    if entry.engine_type in ("classic", "classic_provider"):
        if isinstance(layout, str):
            raise ValueError(
                f"layout={layout!r} (detection engine) is only valid with VLM OCR "
                f"engines, but engine={engine_name!r} is a classic engine."
            )
        return _run_classic(
            image=image,
            engine_name=engine_key,
            entry=entry,
            languages=languages,
            min_confidence=min_confidence,
            device=device,
            detect_only=detect_only,
            options=options,
            context=context,
        )
    elif entry.engine_type in ("vlm_shorthand", "vlm_generic"):
        return _run_vlm(
            image=image,
            entry=entry,
            engine_name=engine_key,
            model=model,
            client=client,
            prompt=prompt,
            instructions=instructions,
            max_new_tokens=max_new_tokens,
            languages=languages,
            layout=layout,
            preserve_markup=preserve_markup,
        )
    else:
        raise ValueError(f"Unknown engine type {entry.engine_type!r} for {engine_name!r}")


# ---------------------------------------------------------------------------
# Classic engine dispatch
# ---------------------------------------------------------------------------


def _run_classic(
    *,
    image: Image.Image,
    engine_name: str,
    entry: EngineEntry,
    languages: Optional[List[str]],
    min_confidence: Optional[float],
    device: Optional[str],
    detect_only: bool,
    options: Optional[BaseOCROptions],
    context: Any = None,
) -> OCRRunResult:
    """Dispatch to a classic OCR engine via EngineCache or EngineProvider."""
    # Mapping options may have been accepted before engine resolution. Reify
    # them now that the effective backend is known; this prevents adapters from
    # silently replacing requested settings with their defaults.
    from natural_pdf.ocr.ocr_provider import normalize_ocr_options
    from natural_pdf.utils.option_validation import resolve_auto_device

    options = normalize_ocr_options(options, engine_name=engine_name)
    effective_languages = tuple(languages or ["en"])
    effective_device = device or "auto"
    if effective_device == "auto":
        effective_device = resolve_auto_device()

    if entry.engine_type == "classic_provider":
        # Public custom engines and provider-only plugins are lifecycle-owned
        # by EngineProvider rather than the built-in model LRU.
        return _run_via_provider(
            image=image,
            entry=entry,
            engine_name=engine_name,
            languages=effective_languages,
            min_confidence=min_confidence,
            device=effective_device,
            detect_only=detect_only,
            options=options,
            context=context,
        )

    init_key = options._init_key() if options is not None else ""

    def factory():
        instance = _instantiate_provider(entry.provider, context=context, options=options)
        try:
            is_available = getattr(instance, "is_available", None)
            if callable(is_available) and not is_available():
                hint = entry.install_hint or f"pip install {engine_name}"
                raise RuntimeError(
                    f"OCR engine {engine_name!r} is not available. Install it with: {hint}"
                )
            initialize = getattr(instance, "_initialize_model", None)
            if callable(initialize):
                initialize(list(effective_languages), effective_device, options)
                if hasattr(instance, "_initialized"):
                    instance._initialized = True
        except BaseException:
            EngineCache._cleanup_engine(instance)
            raise
        return instance

    def run_with_engine(engine: Any) -> OCRRunResult:
        def process():
            return engine.process_image(
                image,
                languages=list(effective_languages),
                min_confidence=min_confidence,
                device=effective_device,
                detect_only=detect_only,
                options=options,
            )

        if entry.needs_gpu_lock:
            lock = _get_inference_lock(engine_name)
            with lock:
                raw_output = process()
        else:
            raw_output = process()

        results = _normalize_engine_output(raw_output, engine_name=engine_name)
        return OCRRunResult(results=results, image_size=image.size, engine_type="classic")

    if init_key is None:
        logger.debug(
            "Bypassing OCR engine cache for %s: options do not have a canonical "
            "constructor identity.",
            engine_name,
        )
        engine = factory()
        try:
            return run_with_engine(engine)
        finally:
            EngineCache._cleanup_engine(engine)

    registration_identity, registration_is_current = _capture_registration(engine_name, entry)
    with _engine_cache.checkout(
        engine_name=engine_name,
        languages=effective_languages,
        device=effective_device,
        init_key=init_key,
        factory=factory,
        provider_identity=id(entry.provider),
        registration_identity=registration_identity,
        registration_is_current=registration_is_current,
    ) as engine:
        return run_with_engine(engine)


def _run_via_provider(
    *,
    image: Image.Image,
    entry: EngineEntry,
    engine_name: str,
    languages: tuple,
    min_confidence: Optional[float],
    device: str,
    detect_only: bool,
    options: Optional[BaseOCROptions],
    context: Any,
) -> OCRRunResult:
    """Run a provider-owned classic engine with constructor options intact."""
    from natural_pdf.engine_provider import get_provider

    provider = get_provider()
    constructor_options: Dict[str, Any] = {}
    if options is not None or entry.provider_init_options:
        constructor_options["options"] = options

    # Public register_ocr_engine factories are wrapped to accept these inputs,
    # which makes language/device model variants part of provider identity.
    # Provider-only registrations keep their existing factory contract and
    # receive only the explicit normalized options object.
    if entry.provider_init_options:
        constructor_options["languages"] = list(languages)
        constructor_options["device"] = device

    with ExitStack() as stack:
        try:
            provider_capability = "ocr.apply"
            engine = stack.enter_context(
                provider.checkout(
                    provider_capability,
                    context=context,
                    name=engine_name,
                    **constructor_options,
                )
            )
        except LookupError:
            provider_capability = "ocr"
            engine = stack.enter_context(
                provider.checkout(
                    provider_capability,
                    context=context,
                    name=engine_name,
                    **constructor_options,
                )
            )

        # Public custom wrappers validate before EngineProvider caches the instance.
        # Provider-only registrations still need the dispatch-level availability check.
        if not entry.provider_init_options:
            is_available = getattr(engine, "is_available", None)
            if callable(is_available):
                try:
                    available = is_available()
                except BaseException:
                    provider.evict_instance(provider_capability, engine_name, engine)
                    raise
                if not available:
                    # Direct provider plugins are cached before dispatch can
                    # validate them. Detach only this exact failed instance;
                    # its active checkout lease defers cleanup until stack exit.
                    provider.evict_instance(provider_capability, engine_name, engine)
                    hint = entry.install_hint or f"pip install {engine_name}"
                    raise RuntimeError(
                        f"OCR engine {engine_name!r} is not available. Install it with: {hint}"
                    )

        def process():
            return engine.process_image(
                image,
                languages=list(languages),
                min_confidence=min_confidence,
                device=device,
                detect_only=detect_only,
                options=options,
            )

        if entry.needs_gpu_lock:
            lock = _get_inference_lock(engine_name)
            with lock:
                raw_output = process()
        else:
            raw_output = process()

        results = _normalize_engine_output(raw_output, engine_name=engine_name)
        return OCRRunResult(results=results, image_size=image.size, engine_type="classic")


def _normalize_engine_output(payload, *, engine_name: str):
    """Normalize classic engine output to ``List[Dict]``.

    ``process_image`` may return ``List[Dict]`` (single image) or
    ``List[List[Dict]]`` (batch of exactly one). ``None`` and empty lists mean
    the engine found no text. Any other shape — including more than one batch
    for a single image — is malformed provider output and raises
    :class:`~natural_pdf.exceptions.OCRError`; silently returning ``[]`` or
    dropping extra batches would be indistinguishable from a blank page.
    Every element is validated, not just the first.
    """
    from natural_pdf.exceptions import OCRError

    if payload is None:
        return []
    if isinstance(payload, list):
        if not payload:
            return []
        first = payload[0]
        if isinstance(first, dict):
            for item in payload:
                if not isinstance(item, dict):
                    raise OCRError(
                        f"OCR engine {engine_name!r} returned a malformed payload: "
                        f"expected a list of result dicts, but found a "
                        f"{type(item).__name__} entry."
                    )
            return payload
        if isinstance(first, list):
            if len(payload) > 1:
                raise OCRError(
                    f"OCR engine {engine_name!r} returned {len(payload)} result "
                    f"batches for a single image; expected exactly one. Refusing "
                    f"to silently discard the extra batches."
                )
            for item in first:
                if not isinstance(item, dict):
                    raise OCRError(
                        f"OCR engine {engine_name!r} returned a malformed batch "
                        f"payload: expected a list of result dicts, got a list "
                        f"containing {type(item).__name__}."
                    )
            return first
        raise OCRError(
            f"OCR engine {engine_name!r} returned a malformed payload: expected "
            f"a list of result dicts (or a batch list of such lists), got a "
            f"list of {type(first).__name__}."
        )
    raise OCRError(
        f"OCR engine {engine_name!r} returned unsupported payload type "
        f"{type(payload).__name__}; expected a list of result dicts "
        f"(or a batch list of such lists)."
    )


# ---------------------------------------------------------------------------
# Classic detection on a pre-rendered image
# ---------------------------------------------------------------------------


def run_detection(
    *,
    image: Image.Image,
    engine_name: str,
    languages: Optional[List[str]] = None,
    device: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Run a classic OCR engine in detect-only mode on a pre-rendered image.

    Returns results in the same format as :func:`_detect_layout_regions`
    so they can be consumed by :func:`_run_layout_ocr_on_image`.

    Each result is a dict with ``label`` (always ``"text"``), ``bbox``
    (``[x0, y0, x1, y1]`` in image pixels), and ``confidence``.
    """
    registry = get_registry()
    engine_key = engine_name.strip().lower()
    entry = registry.get(engine_key)

    if entry is None:
        from natural_pdf.engine_provider import get_provider

        provider = get_provider()
        provider_engines = set()
        for cap in ("ocr", "ocr.apply", "ocr.extract"):
            provider_engines.update(provider.list(cap).get(cap, ()))
        if engine_key in provider_engines:
            entry = EngineEntry(engine_type="classic_provider")
        else:
            available = sorted(set(registry.keys()) | provider_engines)
            raise LookupError(f"Unknown detection engine {engine_name!r}. Available: {available}")

    if entry.engine_type not in ("classic", "classic_provider", "auto_platform"):
        raise ValueError(
            f"layout={engine_name!r} is a VLM engine and cannot be used for "
            f"detection. Use a classic engine like 'rapidocr' or 'paddle'."
        )

    result = _run_classic(
        image=image,
        engine_name=engine_key,
        entry=entry,
        languages=languages,
        min_confidence=None,
        device=device,
        detect_only=True,
        options=None,
        context=None,
    )

    regions: List[Dict[str, Any]] = []
    for r in result.results:
        bbox = r.get("bbox")
        if bbox is None:
            continue
        regions.append(
            {
                "label": "text",
                "bbox": list(bbox),
                "confidence": r.get("confidence", 0.5),
            }
        )
    return regions


# ---------------------------------------------------------------------------
# VLM engine dispatch
# ---------------------------------------------------------------------------


def _run_vlm(
    *,
    image: Image.Image,
    entry: EngineEntry,
    engine_name: str,
    model: Optional[str],
    client: Optional[Any],
    prompt: Optional[str],
    instructions: Optional[str],
    max_new_tokens: Optional[int],
    languages: Optional[List[str]],
    layout: Optional[bool | str] = None,
    preserve_markup: bool = False,
) -> OCRRunResult:
    """Dispatch to a VLM OCR engine."""
    from natural_pdf.ocr.vlm_ocr import run_vlm_ocr_on_image

    # Resolve model for shorthand engines
    if entry.engine_type == "vlm_shorthand" and model is None:
        if entry.model_resolver is not None:
            model = entry.model_resolver()
        else:
            raise ValueError(f"VLM engine {engine_name!r} requires a model= parameter.")

    # Validate generic VLM. The default client applies only when neither
    # model= nor client= was passed; it is bound explicitly here so the
    # resolution happens exactly once.
    if entry.engine_type == "vlm_generic" and model is None and client is None:
        from natural_pdf.core.vlm_client import get_default_client

        default_client, default_model = get_default_client()
        if default_client is None:
            raise ValueError(
                'apply_ocr(engine="vlm") requires a model= and/or client= '
                "parameter, or a default client set via "
                "natural_pdf.set_default_client(). Example:\n"
                '  page.apply_ocr(engine="vlm", model="gemini-2.5-flash", client=client)'
            )
        model = default_model
        client = default_client

    if client is None:
        # An explicit (or engine-resolved) model with no client runs locally.
        # Suppress the module-level default client so nested generate() calls
        # cannot backfill it and send images to a remote endpoint.
        from natural_pdf.core.vlm_client import suppress_default_client

        default_guard = suppress_default_client()
    else:
        from contextlib import nullcontext

        default_guard = nullcontext()

    with default_guard:
        results, img_size = run_vlm_ocr_on_image(
            image,
            model=model,
            client=client,
            max_new_tokens=max_new_tokens,
            prompt=prompt,
            instructions=instructions,
            languages=languages,
            layout=layout,
            family=entry.vlm_family,
            preserve_markup=preserve_markup,
        )

    return OCRRunResult(results=results, image_size=img_size, engine_type="vlm")
