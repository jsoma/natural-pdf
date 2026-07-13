"""OCR Comparison Service — runs multiple OCR engines and compares their outputs."""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Union

from natural_pdf.services.registry import register_delegate

logger = logging.getLogger(__name__)

# Type for engine specs: either a string or a dict with at least "engine" key
EngineSpec = Union[str, Dict[str, Any]]
_SUPPORTED_ENGINE_DEFAULTS = frozenset(
    {
        "options",
        "model",
        "client",
        "prompt",
        "instructions",
        "max_new_tokens",
        "layout",
        "preserve_markup",
    }
)
_SUPPORTED_SPEC_KEYS = _SUPPORTED_ENGINE_DEFAULTS | {
    "engine",
    "name",
    "label",
    "resolution",
    "languages",
    "device",
    "min_confidence",
}


def _normalize_spec(
    spec: EngineSpec,
    *,
    default_resolution: int,
    default_languages: Optional[List[str]],
    default_device: Optional[str],
    default_min_confidence: Optional[float],
    engine_defaults: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Normalize an engine spec into a full dict with defaults applied."""
    engine_defaults = engine_defaults or {}
    if isinstance(spec, str):
        merged = {**engine_defaults, "engine": spec}
    elif isinstance(spec, dict):
        unknown = sorted(set(spec) - _SUPPORTED_SPEC_KEYS)
        if unknown:
            raise TypeError("Unsupported OCR engine spec option(s): " + ", ".join(unknown))
        merged = {**engine_defaults, **spec}
    else:
        raise TypeError("Each OCR engine spec must be a string or mapping")

    engine = merged.get("engine") or merged.get("name")
    if not engine:
        raise ValueError(f"Engine spec must have an 'engine' key: {spec!r}")

    label = merged.get("label") or _auto_label(engine, merged)
    return {
        "engine": engine,
        "label": label,
        "resolution": merged.get("resolution", default_resolution),
        "languages": merged.get("languages", default_languages),
        "device": merged.get("device", default_device),
        "min_confidence": merged.get("min_confidence", default_min_confidence),
        # VLM params:
        "model": merged.get("model"),
        "client": merged.get("client"),
        "prompt": merged.get("prompt"),
        "instructions": merged.get("instructions"),
        "max_new_tokens": merged.get("max_new_tokens"),
        "layout": merged.get("layout"),
        "preserve_markup": bool(merged.get("preserve_markup", False)),
        # Classic params:
        "options": merged.get("options"),
    }


def _auto_label(engine: str, spec: Dict[str, Any]) -> str:
    """Generate a label from engine name + overridden params."""
    # Params that differentiate runs of the same engine
    diff_keys = {"resolution", "languages", "device", "model"}
    parts = []
    for key in sorted(diff_keys):
        if key in spec and key != "engine":
            parts.append(f"{key}={spec[key]}")
    if parts:
        return f"{engine}({', '.join(parts)})"
    return engine


class OcrComparisonService:
    """Runs multiple OCR engines on a page and compares their outputs."""

    def __init__(self, context):
        self._context = context

    @register_delegate("ocr_comparison", "compare_ocr")
    def compare_ocr(
        self,
        host,
        *,
        engines: List[EngineSpec],
        normalize: str = "collapse",
        strategy: str = "auto",
        resolution: int = 150,
        languages: Optional[List[str]] = None,
        min_confidence: Optional[float] = None,
        device: Optional[str] = None,
        engine_options: Optional[Dict[str, Dict[str, Any]]] = None,
        apply_exclusions: bool = True,
        **kwargs,
    ):
        """Compare OCR engines on a page without mutating page elements.

        Args:
            host: The page to OCR.
            engines: List of engine specs. Each can be a string (engine name)
                or a dict with ``"engine"`` key plus overrides like
                ``"resolution"``, ``"languages"``, ``"model"``, ``"label"``.
            normalize: Text normalization mode ("collapse", "strict", "ignore").
            strategy: Alignment strategy ("auto", "rows", "tiles").
            resolution: Default render DPI.
            languages: Language codes for OCR.
            min_confidence: Minimum confidence filter.
            device: Device for OCR ("cpu", "cuda", "mps").
            engine_options: Per-engine overrides (deprecated, use dict specs instead).
            apply_exclusions: Whether to mask exclusion zones.

        Returns:
            OcrComparison result object.
        """
        from natural_pdf.ocr.alignment import align_ocr_outputs
        from natural_pdf.ocr.comparison import OcrComparison

        unknown_kwargs = sorted(set(kwargs) - _SUPPORTED_ENGINE_DEFAULTS)
        if unknown_kwargs:
            raise TypeError("Unsupported compare_ocr option(s): " + ", ".join(unknown_kwargs))
        engine_defaults = {key: kwargs[key] for key in _SUPPORTED_ENGINE_DEFAULTS if key in kwargs}

        engine_options = engine_options or {}
        engine_elements: Dict[str, list] = {}
        failed_engines: Dict[str, str] = {}
        runtimes: Dict[str, float] = {}
        successful_engines: List[str] = []

        ocr_service = self._context.get_service("ocr")

        for spec in engines:
            normalized = _normalize_spec(
                spec,
                default_resolution=resolution,
                default_languages=languages,
                default_device=device,
                default_min_confidence=min_confidence,
                engine_defaults=engine_defaults,
            )

            engine_str = normalized["engine"]
            label = normalized["label"]

            # Apply legacy engine_options if present
            if engine_str in engine_options:
                legacy = engine_options[engine_str]
                for k, v in legacy.items():
                    if k not in normalized or normalized[k] is None:
                        normalized[k] = v

            try:
                logger.info(
                    "Running OCR engine '%s' (label='%s') for comparison...", engine_str, label
                )
                t0 = time.time()

                word_elements = ocr_service.extract_ocr_elements(
                    host,
                    engine=engine_str,
                    resolution=normalized["resolution"],
                    languages=normalized["languages"],
                    min_confidence=normalized["min_confidence"],
                    device=normalized["device"],
                    options=normalized.get("options"),
                    apply_exclusions=apply_exclusions,
                    model=normalized.get("model"),
                    client=normalized.get("client"),
                    prompt=normalized.get("prompt"),
                    instructions=normalized.get("instructions"),
                    max_new_tokens=normalized.get("max_new_tokens"),
                    layout=normalized.get("layout"),
                    preserve_markup=normalized.get("preserve_markup", False),
                )

                elapsed = time.time() - t0
                runtimes[label] = elapsed

                engine_elements[label] = word_elements
                successful_engines.append(label)
                logger.info(
                    "Engine '%s': %d elements in %.1fs",
                    label,
                    len(word_elements),
                    elapsed,
                )

            except Exception as e:
                failed_engines[label] = str(e)
                logger.warning("Engine '%s' failed: %s", label, e)

        if not engine_elements:
            logger.error("All engines failed. Cannot create comparison.")
            return OcrComparison(
                page=host,
                engines=[],
                failed_engines=failed_engines,
                regions=[],
                engine_elements={},
                strategy_used="none",
                diagnostics={"error": "all_engines_failed"},
                runtimes=runtimes,
                resolution=resolution,
                normalize_mode=normalize,
            )

        # Align outputs
        page_bbox = (
            float(getattr(host, "x0", 0)),
            float(getattr(host, "top", 0)),
            float(getattr(host, "width", 612)),
            float(getattr(host, "height", 792)),
        )

        regions, strategy_used, diagnostics = align_ocr_outputs(
            engine_elements,
            page_bbox,
            strategy=strategy,
            normalize=normalize,
        )

        return OcrComparison(
            page=host,
            engines=successful_engines,
            failed_engines=failed_engines,
            regions=regions,
            engine_elements=engine_elements,
            strategy_used=strategy_used,
            diagnostics=diagnostics,
            runtimes=runtimes,
            resolution=resolution,
            normalize_mode=normalize,
        )
