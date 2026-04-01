"""Unified OCR dispatch — single source of truth for all OCR engines.

Provides one registry (classic + VLM engines), one dispatch function
(:func:`run_ocr`), and an LRU cache for classic engine instances.
"""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

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

    engine_type: str  # "classic" | "vlm_shorthand" | "vlm_generic"

    # Classic engines:
    provider: Optional[Any] = None  # OCREngine class or factory
    options_class: Optional[Type[BaseOCROptions]] = None

    # VLM shorthand engines (dots, glm_ocr, chandra):
    model_resolver: Optional[Callable[[], str]] = None

    # All VLM engines:
    vlm_family: Optional[str] = None  # "dots_mocr", "glm_ocr", "chandra", etc.

    # Locking:
    needs_gpu_lock: bool = True  # False for known remote-API-only engines

    # Install hint for error messages:
    install_hint: Optional[str] = None


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

    return {
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
            install_hint="pip install rapidocr_onnxruntime",
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


def register_engine(name: str, entry: EngineEntry) -> None:
    """Register an engine in the unified registry.

    Useful for tests and plugins that add custom engines.
    """
    registry = get_registry()
    registry[name.strip().lower()] = entry


# ---------------------------------------------------------------------------
# LRU engine cache for classic engines
# ---------------------------------------------------------------------------


class EngineCache:
    """Thread-safe LRU cache for classic OCR engine instances.

    On eviction, calls ``engine.cleanup()`` if available.
    """

    def __init__(self, maxsize: int = 4):
        self._cache: OrderedDict[tuple, Any] = OrderedDict()
        self._lock = threading.Lock()
        self._maxsize = maxsize

    @property
    def maxsize(self) -> int:
        return self._maxsize

    @maxsize.setter
    def maxsize(self, value: int) -> None:
        self._maxsize = max(1, value)
        with self._lock:
            self._evict_to_capacity()

    def get_or_create(
        self,
        engine_name: str,
        languages: Tuple[str, ...],
        device: str,
        init_key: str,
        factory: Callable[[], Any],
    ) -> Any:
        """Get a cached engine or create a new one."""
        key = (engine_name, languages, device, init_key)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]

            # Evict LRU if at capacity
            self._evict_to_capacity()

            # Create outside lock scope to avoid holding lock during heavy init
        engine = factory()

        with self._lock:
            # Re-check in case another thread created the same key
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]

            self._cache[key] = engine
            self._evict_to_capacity()
            return engine

    def clear(self) -> int:
        """Evict all cached engines, calling cleanup on each."""
        with self._lock:
            count = len(self._cache)
            while self._cache:
                _, engine = self._cache.popitem()
                self._cleanup_engine(engine)
            return count

    def _evict_to_capacity(self) -> None:
        """Evict oldest entries until at or below maxsize. Must hold lock."""
        while len(self._cache) > self._maxsize:
            _, evicted = self._cache.popitem(last=False)
            self._cleanup_engine(evicted)

    @staticmethod
    def _cleanup_engine(engine: Any) -> None:
        """Call cleanup on an evicted engine."""
        cleanup_fn = getattr(engine, "cleanup", None)
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
    layout: Optional[bool] = None,
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
            )
        else:
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
    from natural_pdf.utils.option_validation import resolve_auto_device

    effective_languages = tuple(sorted(languages or ["en"]))
    effective_device = device or "auto"
    if effective_device == "auto":
        effective_device = resolve_auto_device()

    if entry.engine_type == "classic_provider":
        # Engine only exists in EngineProvider (e.g. test/plugin engines)
        return _run_via_provider(
            image=image,
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
        engine_cls = entry.provider
        instance = engine_cls()
        if not instance.is_available():
            hint = entry.install_hint or f"pip install {engine_name}"
            raise RuntimeError(
                f"OCR engine {engine_name!r} is not available. Install it with: {hint}"
            )
        instance._initialize_model(list(effective_languages), effective_device, options)
        instance._initialized = True
        return instance

    engine = _engine_cache.get_or_create(
        engine_name=engine_name,
        languages=effective_languages,
        device=effective_device,
        init_key=init_key,
        factory=factory,
    )

    lock = _get_inference_lock(engine_name)
    with lock:
        raw_output = engine.process_image(
            image,
            languages=list(effective_languages),
            min_confidence=min_confidence,
            device=effective_device,
            detect_only=detect_only,
            options=options,
        )

    # Normalize: process_image may return List[Dict] (single) or List[List[Dict]] (batch)
    results = _normalize_engine_output(raw_output)

    return OCRRunResult(results=results, image_size=image.size, engine_type="classic")


def _run_via_provider(
    *,
    image: Image.Image,
    engine_name: str,
    languages: tuple,
    min_confidence: Optional[float],
    device: str,
    detect_only: bool,
    options: Optional[BaseOCROptions],
    context: Any,
) -> OCRRunResult:
    """Fallback: run OCR via EngineProvider for engines not in unified registry."""
    from natural_pdf.engine_provider import get_provider

    provider = get_provider()
    try:
        engine = provider.get("ocr.apply", context=context, name=engine_name)
    except LookupError:
        engine = provider.get("ocr", context=context, name=engine_name)

    lock = _get_inference_lock(engine_name)
    with lock:
        raw_output = engine.process_image(
            image,
            languages=list(languages),
            min_confidence=min_confidence,
            device=device,
            detect_only=detect_only,
            options=options,
        )

    results = _normalize_engine_output(raw_output)
    return OCRRunResult(results=results, image_size=image.size, engine_type="classic")


def _normalize_engine_output(payload):
    """Normalize engine output to List[Dict]."""
    if isinstance(payload, list):
        if payload and isinstance(payload[0], list):
            return payload[0]
        elif payload and isinstance(payload[0], dict):
            return payload
        elif not payload:
            return []
    return []


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
    layout: Optional[bool] = None,
) -> OCRRunResult:
    """Dispatch to a VLM OCR engine."""
    from natural_pdf.ocr.vlm_ocr import run_vlm_ocr_on_image

    # Resolve model for shorthand engines
    if entry.engine_type == "vlm_shorthand" and model is None:
        if entry.model_resolver is not None:
            model = entry.model_resolver()
        else:
            raise ValueError(f"VLM engine {engine_name!r} requires a model= parameter.")

    # Validate generic VLM
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
        if model is None:
            model = default_model

    results, img_size = run_vlm_ocr_on_image(
        image,
        model=model,
        client=client,
        max_new_tokens=max_new_tokens,
        prompt=prompt,
        instructions=instructions,
        languages=languages,
        layout=layout,
    )

    return OCRRunResult(results=results, image_size=img_size, engine_type="vlm")
