from __future__ import annotations

from typing import Any

from natural_pdf.services.registry import register_delegate


class ToLLMService:
    """Service that produces LLM-optimized text representations of PDF objects."""

    def __init__(self, context):
        self._context = context

    @register_delegate("to_llm", "to_llm")
    def to_llm(self, host, **kwargs) -> str:
        from natural_pdf.core.page import Page
        from natural_pdf.core.page_collection import PageCollection
        from natural_pdf.core.pdf import PDF
        from natural_pdf.describe.to_llm import (
            collection_to_llm,
            element_to_llm,
            page_to_llm,
            pdf_to_llm,
            region_to_llm,
        )
        from natural_pdf.elements.base import Element
        from natural_pdf.elements.element_collection import ElementCollection
        from natural_pdf.elements.region import Region

        if isinstance(host, PDF):
            return pdf_to_llm(host, **kwargs)
        if isinstance(host, PageCollection):
            return pdf_to_llm(host, **kwargs)
        if isinstance(host, Page):
            return page_to_llm(host, **kwargs)
        if isinstance(host, Region):
            return region_to_llm(host, **kwargs)
        if isinstance(host, ElementCollection):
            return collection_to_llm(host, **kwargs)
        if isinstance(host, Element):
            return element_to_llm(host, **kwargs)

        return f"to_llm() not implemented for {type(host).__name__}"
