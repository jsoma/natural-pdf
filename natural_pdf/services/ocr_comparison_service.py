"""OCR Comparison Service — runs multiple OCR engines and compares their outputs."""

from __future__ import annotations

import copy
import logging
import time
from typing import Any, Dict, List, Optional

from natural_pdf.ocr.ocr_provider import (
    normalize_ocr_options,
    resolve_ocr_device,
    resolve_ocr_engine_name,
    resolve_ocr_languages,
    resolve_ocr_min_confidence,
    run_ocr_apply,
)
from natural_pdf.services.registry import register_delegate

logger = logging.getLogger(__name__)


class OcrComparisonService:
    """Runs multiple OCR engines on a page and compares their outputs."""

    def __init__(self, context):
        self._context = context

    @register_delegate("ocr_comparison", "compare_ocr")
    def compare_ocr(
        self,
        host,
        *,
        engines: List[str],
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
            engines: List of engine names to compare.
            normalize: Text normalization mode ("collapse", "strict", "ignore").
            strategy: Alignment strategy ("auto", "rows", "tiles").
            resolution: Default render DPI.
            languages: Language codes for OCR.
            min_confidence: Minimum confidence filter.
            device: Device for OCR ("cpu", "cuda", "mps").
            engine_options: Per-engine overrides, e.g. {"glm_ocr": {"resolution": 200}}.
            apply_exclusions: Whether to mask exclusion zones.

        Returns:
            OcrComparison result object.
        """
        from natural_pdf.core.ocr_converter import OCRConverter
        from natural_pdf.ocr.alignment import align_ocr_outputs
        from natural_pdf.ocr.comparison import OcrComparison

        engine_options = engine_options or {}
        engine_elements: Dict[str, list] = {}
        failed_engines: Dict[str, str] = {}
        runtimes: Dict[str, float] = {}
        successful_engines: List[str] = []

        # Resolve render kwargs once
        render_kwargs = {"apply_exclusions": apply_exclusions}

        for engine_str in engines:
            try:
                # Resolve engine name
                engine_name = resolve_ocr_engine_name(
                    context=host,
                    requested=engine_str,
                    options=None,
                    scope="page",
                )

                # Per-engine overrides
                per_engine = engine_options.get(engine_name, {})
                engine_resolution = per_engine.get("resolution", resolution)
                engine_languages = per_engine.get("languages", languages)
                engine_device = per_engine.get("device", device)
                engine_min_conf = per_engine.get("min_confidence", min_confidence)

                logger.info("Running OCR engine '%s' for comparison...", engine_name)
                t0 = time.time()

                # Run OCR
                ocr_payload = run_ocr_apply(
                    target=host,
                    context=host,
                    engine_name=engine_name,
                    resolution=engine_resolution,
                    languages=engine_languages,
                    min_confidence=engine_min_conf,
                    device=engine_device,
                    detect_only=False,
                    options=None,
                    render_kwargs=render_kwargs,
                )

                elapsed = time.time() - t0
                runtimes[engine_name] = elapsed

                # Convert to TextElements without adding to page
                image_width, image_height = ocr_payload.image_size
                if not image_width or not image_height:
                    failed_engines[engine_name] = "OCR returned no image dimensions"
                    continue

                page_width = getattr(host, "width", None) or 0
                page_height = getattr(host, "height", None) or 0
                scale_x = page_width / image_width if page_width else 1.0
                scale_y = page_height / image_height if page_height else 1.0

                converter = OCRConverter(host)
                word_elements, _ = converter.convert(
                    ocr_payload.results,
                    scale_x=scale_x,
                    scale_y=scale_y,
                    engine_name=engine_name,
                )

                engine_elements[engine_name] = word_elements
                successful_engines.append(engine_name)
                logger.info(
                    "Engine '%s': %d elements in %.1fs",
                    engine_name,
                    len(word_elements),
                    elapsed,
                )

            except Exception as e:
                failed_engines[engine_str] = str(e)
                logger.warning("Engine '%s' failed: %s", engine_str, e)

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
