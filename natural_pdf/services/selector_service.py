from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union, cast

from natural_pdf.core.selector_utils import execute_selector_query, normalize_selector_input
from natural_pdf.elements.element_collection import ElementCollection
from natural_pdf.services.registry import register_delegate

logger = logging.getLogger(__name__)


class SelectorService:
    """Shared selector logic powering find/find_all across host types."""

    def __init__(self, context):
        self._context = context

    @register_delegate("selector", "find")
    def find(self, host, **kwargs):
        collection = self.find_all(host, **kwargs)
        if collection is None:
            return None
        first = getattr(collection, "first", None)
        if first is not None:
            return first
        elements = getattr(collection, "elements", None)
        if elements:
            return elements[0]
        if isinstance(collection, list):
            return collection[0] if collection else None
        return None

    @register_delegate("selector", "find_all")
    def find_all(self, host, **kwargs):
        from natural_pdf.core.page import Page
        from natural_pdf.core.page_collection import PageCollection
        from natural_pdf.core.pdf import PDF
        from natural_pdf.core.pdf_collection import PDFCollection
        from natural_pdf.elements.base import Element
        from natural_pdf.elements.region import Region
        from natural_pdf.flows.flow import Flow
        from natural_pdf.flows.region import FlowRegion

        if isinstance(host, Page):
            return self._find_all_page(host, **kwargs)
        if isinstance(host, Region):
            return self._find_all_region(host, **kwargs)
        if isinstance(host, FlowRegion):
            return self._find_all_flow_region(host, **kwargs)
        if isinstance(host, Flow):
            return self._find_all_flow(host, **kwargs)
        if isinstance(host, PDF):
            return self._find_all_pdf(host, **kwargs)
        if isinstance(host, PageCollection):
            return self._find_all_page_collection(host, **kwargs)
        if isinstance(host, PDFCollection):
            return self._find_all_pdf_collection(host, **kwargs)
        if isinstance(host, Element):
            return self._find_all_element(host, **kwargs)
        raise TypeError(f"Host type {type(host)!r} is not selector-capable.")

    def _find_all_page(self, page, **kwargs) -> ElementCollection:
        from natural_pdf.core.selector_utils import execute_selector_query

        selector = kwargs.get("selector")
        text = kwargs.get("text")
        apply_exclusions = kwargs.get("apply_exclusions", True)
        regex = kwargs.get("regex", False)
        case = kwargs.get("case", True)
        text_tolerance = kwargs.get("text_tolerance")
        auto_text_tolerance = kwargs.get("auto_text_tolerance")
        reading_order = kwargs.get("reading_order", True)
        near_threshold = kwargs.get("near_threshold")
        engine = kwargs.get("engine")
        if engine is None:
            engine = self._context.get_option("selector", "engine", host=page)

        effective_selector = normalize_selector_input(
            selector,
            text,
            logger=logger,
            context="Page.find_all",
        )

        results_collection = execute_selector_query(
            page,
            effective_selector,
            text_tolerance=text_tolerance,
            auto_text_tolerance=auto_text_tolerance,
            regex=regex,
            case=case,
            reading_order=reading_order,
            near_threshold=near_threshold,
            engine=engine,
        )

        if apply_exclusions and results_collection:
            filtered_elements = page._filter_elements_by_exclusions(results_collection.elements)
            return ElementCollection(filtered_elements)
        return results_collection

    def _find_all_region(self, region, **kwargs) -> ElementCollection:
        overlap_mode = (kwargs.get("overlap") or "full").lower()
        if overlap_mode not in {"full", "partial", "center"}:
            raise ValueError(
                f"Invalid overlap value: {overlap_mode}. Must be 'full', 'partial', or 'center'"
            )

        page_kwargs = dict(kwargs)
        page_kwargs.pop("overlap", None)
        page_results = self._find_all_page(region.page, **page_kwargs)
        if not page_results:
            return ElementCollection([])

        filtered = region._filter_elements_by_overlap_mode(
            page_results.elements,
            overlap_mode,
        )
        unique: List[Any] = []
        seen = set()
        for element in filtered:
            marker = id(element)
            if marker in seen:
                continue
            seen.add(marker)
            unique.append(element)
        return ElementCollection(unique)

    def _find_all_element(self, element, **kwargs) -> ElementCollection:
        from natural_pdf.elements.region import Region

        temp_region = Region(element.page, element.bbox)
        return self._find_all_region(temp_region, **kwargs)

    def _find_all_flow_region(self, host, **kwargs) -> ElementCollection:
        from natural_pdf.elements.element_collection import ElementCollection

        combined: List[Any] = []
        for region in host.constituent_regions:
            collection = self._find_all_region(region, **kwargs)
            if collection:
                combined.extend(collection.elements)

        unique: List[Any] = []
        seen: set[Any] = set()
        for el in combined:
            if el not in seen:
                unique.append(el)
                seen.add(el)

        return ElementCollection(unique)

    def _find_all_flow(self, flow, **kwargs):
        from natural_pdf.flows.collections import FlowElementCollection
        from natural_pdf.flows.element import FlowElement

        selector = kwargs.get("selector")
        text = kwargs.get("text")
        apply_exclusions = kwargs.get("apply_exclusions", True)
        regex = kwargs.get("regex", False)
        case = kwargs.get("case", True)
        text_tolerance = kwargs.get("text_tolerance")
        auto_text_tolerance = kwargs.get("auto_text_tolerance")
        reading_order = kwargs.get("reading_order", True)
        engine = kwargs.get("engine")

        segments_by_page: Dict[Any, List[Tuple[Any, str]]] = {}
        for i, segment in enumerate(flow.segments):
            if hasattr(segment, "page") and hasattr(segment.page, "find_all"):
                page_obj = segment.page
                segment_type = "region"
            elif (
                hasattr(segment, "find_all")
                and hasattr(segment, "width")
                and hasattr(segment, "height")
                and not hasattr(segment, "page")
            ):
                page_obj = segment
                segment_type = "page"
            else:
                raise TypeError(f"Segment {i+1} does not support find_all: {segment!r}")

            segments_by_page.setdefault(page_obj, []).append((segment, segment_type))

        all_flow_elements: List[FlowElement] = []
        for page_obj, page_segments in segments_by_page.items():
            page_collection = self._find_all_page(
                page_obj,
                selector=selector,
                text=text,
                apply_exclusions=apply_exclusions,
                regex=regex,
                case=case,
                text_tolerance=text_tolerance,
                auto_text_tolerance=auto_text_tolerance,
                reading_order=reading_order,
                engine=engine,
            )
            if not page_collection:
                continue

            for segment, segment_type in page_segments:
                if segment_type == "page":
                    for phys_elem in page_collection.elements:
                        all_flow_elements.append(FlowElement(physical_object=phys_elem, flow=flow))
                else:
                    for phys_elem in page_collection.elements:
                        try:
                            if segment.intersects(phys_elem):
                                all_flow_elements.append(
                                    FlowElement(physical_object=phys_elem, flow=flow)
                                )
                        except Exception as exc:
                            logger.debug("Error checking intersection: %s", exc)
                            all_flow_elements.append(
                                FlowElement(physical_object=phys_elem, flow=flow)
                            )

        unique: List[FlowElement] = []
        seen_ids = set()
        for flow_elem in all_flow_elements:
            phys_elem = flow_elem.physical_object
            elem_id = (
                getattr(phys_elem.page, "index", id(phys_elem.page))
                if hasattr(phys_elem, "page")
                else id(phys_elem)
            )
            marker = (elem_id, getattr(phys_elem, "bbox", id(phys_elem)))
            if marker in seen_ids:
                continue
            seen_ids.add(marker)
            unique.append(flow_elem)

        return FlowElementCollection(unique)

    def _find_all_pdf(self, pdf, **kwargs) -> ElementCollection:
        from natural_pdf.elements.element_collection import ElementCollection

        selector = kwargs.get("selector")
        text = kwargs.get("text")

        effective_selector = normalize_selector_input(
            selector,
            text,
            logger=logger,
            context="PDF.find_all",
        )

        per_page_kwargs = dict(kwargs)
        per_page_kwargs["selector"] = effective_selector
        per_page_kwargs.pop("text", None)

        matches: List[Any] = []
        for page in pdf.pages:
            result = self._find_all_page(page, **per_page_kwargs)
            if result:
                matches.extend(result.elements)
        return ElementCollection(matches)

    def _find_all_page_collection(self, collection, **kwargs) -> ElementCollection:
        from natural_pdf.elements.element_collection import ElementCollection

        selector = kwargs.get("selector")
        text = kwargs.get("text")

        effective_selector = normalize_selector_input(
            selector,
            text,
            logger=logger,
            context="PageCollection.find_all",
        )

        per_page_kwargs = dict(kwargs)
        per_page_kwargs["selector"] = effective_selector
        per_page_kwargs.pop("text", None)

        matches: List[Any] = []
        for page in collection.pages:
            result = self._find_all_page(page, **per_page_kwargs)
            if result:
                matches.extend(result.elements)
        return ElementCollection(matches)

    def _find_all_pdf_collection(self, collection, **kwargs) -> ElementCollection:
        from natural_pdf.elements.element_collection import ElementCollection

        selector = kwargs.get("selector")
        text = kwargs.get("text")

        effective_selector = normalize_selector_input(
            selector,
            text,
            logger=logger,
            context="PDFCollection.find_all",
        )

        per_pdf_kwargs = dict(kwargs)
        per_pdf_kwargs["selector"] = effective_selector
        per_pdf_kwargs.pop("text", None)

        matches: List[Any] = []
        for pdf in collection._pdfs:
            result = self._find_all_pdf(pdf, **per_pdf_kwargs)
            if result:
                matches.extend(result.elements)
        return ElementCollection(matches)
