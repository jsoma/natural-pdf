from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional

from natural_pdf.services.registry import register_delegate

logger = logging.getLogger(__name__)


class LayoutService:
    """Shared layout-analysis helpers for pages, flows, and collections."""

    def __init__(self, context):
        self._context = context

    def _normalize_engine_arg(self, args, kwargs):
        """Allow the first positional arg to act as the engine name."""
        if not args:
            return kwargs

        if len(args) > 1:
            raise TypeError(
                "analyze_layout() accepts at most one positional argument (the engine name)"
            )

        if "engine" in kwargs and kwargs.get("engine") is not None:
            raise TypeError("analyze_layout() got multiple values for 'engine'")

        engine_arg = args[0]
        if engine_arg is None:
            return kwargs

        if not isinstance(engine_arg, str):
            raise TypeError(
                "The positional argument to analyze_layout() must be a string engine name; "
                f"received {type(engine_arg).__name__!r}"
            )

        merged_kwargs = dict(kwargs)
        merged_kwargs["engine"] = engine_arg
        return merged_kwargs

    # ------------------------------------------------------------------
    # Delegates
    # ------------------------------------------------------------------
    @register_delegate("layout", "layout_analyzer")
    def layout_analyzer(self, host):
        from natural_pdf.core.page import Page

        if not isinstance(host, Page):
            raise TypeError("layout_analyzer() is only available on Page objects.")
        return self._get_page_analyzer(host)

    @register_delegate("layout", "analyze_layout")
    def analyze_layout(self, host, *args, **kwargs):
        kwargs = self._normalize_engine_arg(args, kwargs)
        from natural_pdf.core.page import Page
        from natural_pdf.core.page_collection import PageCollection
        from natural_pdf.core.pdf import PDF
        from natural_pdf.flows.flow import Flow

        if isinstance(host, Page):
            return self._analyze_page(host, **kwargs)
        if isinstance(host, Flow):
            return self._analyze_flow(host, **kwargs)
        if isinstance(host, PageCollection):
            return self._analyze_page_collection(host, **kwargs)
        if isinstance(host, PDF):
            return self._analyze_pdf(host, **kwargs)
        raise TypeError(f"Host type {type(host)!r} is not layout-capable.")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _get_page_analyzer(self, page):
        analyzer = getattr(page, "_layout_analyzer", None)
        if analyzer is None:
            from natural_pdf.analyzers.layout.layout_analyzer import LayoutAnalyzer

            analyzer = LayoutAnalyzer(page)
            setattr(page, "_layout_analyzer", analyzer)
        return analyzer

    def _analyze_page(self, page, **kwargs):
        from natural_pdf.elements.element_collection import ElementCollection

        kwargs = dict(kwargs)
        requested_existing = kwargs.get("existing")
        if isinstance(requested_existing, str) and requested_existing.lower() != "replace":
            logger.warning(
                "Layout analysis always replaces existing detected regions; ignoring existing=%r",
                requested_existing,
            )
        kwargs["existing"] = "replace"

        engine = kwargs.get("engine")

        try:
            page.clear_detected_layout_regions()
        except Exception:
            logger.debug("Failed to clear detected layout regions for %s", page)

        analyzer = self._get_page_analyzer(page)
        analyzer.analyze_layout(**kwargs)

        detected_regions = [
            region
            for region in getattr(page._element_mgr, "regions", [])
            if getattr(region, "source", None) == "detected"
            and (not engine or getattr(region, "model", None) == engine)
        ]
        return ElementCollection(detected_regions)

    def _analyze_flow(self, flow, **kwargs):
        from natural_pdf.elements.element_collection import ElementCollection

        logger.info(
            "Analyzing layout across Flow with %d segments (engine: %s)",
            len(flow.segments),
            kwargs.get("engine") or "default",
        )

        if not flow.segments:
            raise ValueError("Flow has no segments; cannot analyze layout")

        segments_by_page: Dict[Any, List[Any]] = {}
        for idx, segment in enumerate(flow.segments):
            page_obj = None
            segment_type = "region"
            if hasattr(segment, "analyze_layout"):
                page_obj = segment
                segment_type = "page"
            elif hasattr(segment, "page") and hasattr(segment.page, "analyze_layout"):
                page_obj = segment.page
            if page_obj is None:
                raise TypeError(f"Segment {idx + 1} does not support layout analysis: {segment!r}")
            segments_by_page.setdefault(page_obj, []).append((segment, segment_type))

        all_detected_regions = []
        for page_obj, page_segments in segments_by_page.items():
            try:
                page_results = self._analyze_page(page_obj, **kwargs)
            except Exception as exc:  # pragma: no cover - defensive
                logger.error(
                    "Error analyzing layout for page %s: %s",
                    getattr(page_obj, "number", "?"),
                    exc,
                    exc_info=True,
                )
                continue

            page_regions: Iterable[Any]
            if hasattr(page_results, "elements"):
                page_regions = page_results.elements
            else:
                page_regions = page_results

            for segment, seg_type in page_segments:
                if seg_type == "page":
                    all_detected_regions.extend(page_regions)
                else:
                    intersecting = []
                    for region in page_regions:
                        try:
                            if segment.intersects(region):
                                intersecting.append(region)
                        except Exception:
                            intersecting.append(region)
                    all_detected_regions.extend(intersecting)

        unique_regions = []
        seen = set()
        for region in all_detected_regions:
            region_id = (
                getattr(region.page, "index", id(region.page)),
                getattr(region, "bbox", id(region)),
            )
            if region_id not in seen:
                unique_regions.append(region)
                seen.add(region_id)

        return ElementCollection(unique_regions)

    def _analyze_page_collection(self, collection, **kwargs):
        from natural_pdf.elements.element_collection import ElementCollection

        show_progress = kwargs.pop("show_progress", True)
        pages_iter = collection.pages
        if show_progress:
            try:
                from tqdm.auto import tqdm

                pages_iter = tqdm(collection.pages, desc="Analyzing layout")
            except Exception:  # pragma: no cover - optional dependency
                pass

        all_regions = []
        for page in pages_iter:
            page_regions = self._analyze_page(page, **kwargs)
            if page_regions:
                all_regions.extend(page_regions.elements)

        return ElementCollection(all_regions)

    def _analyze_pdf(self, pdf, **kwargs):
        return self._analyze_page_collection(pdf.pages, **kwargs)
