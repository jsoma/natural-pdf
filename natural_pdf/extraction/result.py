"""Structured extraction result types."""

from __future__ import annotations

import logging
import textwrap
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Tuple, Type

from pydantic import BaseModel

logger = logging.getLogger(__name__)


def build_enriched_label(field_name: str, value: Any) -> str:
    """Build a multi-line legend label from a field name and value.

    Format::

        field_name:
          wrapped value line 1
          wrapped value line 2

    Newlines in the value are replaced with spaces, text is wrapped at 25
    characters, and truncated to 4 value lines with ``...`` if needed.
    """
    value_str = str(value).replace("\n", " ").strip()
    max_lines = 4
    wrapped = textwrap.fill(value_str, width=25)
    lines = wrapped.split("\n")
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1][:22] + "..."
    return field_name + ":\n" + "\n".join("  " + line.strip() for line in lines)


@dataclass
class FieldResult:
    """Bundles an extracted field value with its source citations and confidence.

    Attributes:
        value: The extracted value (str, int, list, etc.).
        citations: ElementCollection of source TextElements, or an empty
            collection when citations were not requested.
        confidence: Per-field confidence score (float or str), or ``None``
            when confidence was not requested.
        sources: Resolved source text strings (e.g.
            ``["Invoice #12345", "Date: 2024"]``), or ``None`` when
            citations were not requested.
    """

    value: Any
    citations: Any  # ElementCollection — typed as Any to avoid circular import
    confidence: Optional[Any] = None
    sources: Optional[List[str]] = None

    def __str__(self) -> str:
        return str(self.value)

    def __repr__(self) -> str:
        cit_len = len(self.citations) if self.citations is not None else 0
        parts = [f"value={self.value!r}", f"citations={cit_len} elements"]
        if self.confidence is not None:
            parts.append(f"confidence={self.confidence!r}")
        return f"FieldResult({', '.join(parts)})"

    def show(self, **kwargs):
        """Show the citation elements for this field."""
        if self.citations is not None and len(self.citations) > 0:
            return self.citations.show(**kwargs)
        return None


class StructuredDataResult:
    """Result of a structured data extraction.

    Provides multiple access patterns:

    - **Attribute access**: ``result.site`` returns the raw value
    - **Item access**: ``result["site"]`` returns a :class:`FieldResult`
      with ``.value`` and ``.citations``
    - **Iteration**: ``for name, field in result.items()`` yields
      ``(str, FieldResult)`` pairs
    - **Dict export**: ``result.to_dict()`` returns a plain dict of values
    - **Visualization**: ``result.show()`` highlights all citations on the page

    Attributes:
        data: The validated Pydantic model, or None on failure.
        success: Whether extraction succeeded.
        error_message: Error details if extraction failed.
        raw_output: Raw output from the language model.
        model_used: Identifier of the model used.
    """

    def __init__(
        self,
        *,
        data: Optional[BaseModel] = None,
        success: bool = True,
        error_message: Optional[str] = None,
        raw_output: Any = None,
        model_used: Optional[str] = None,
        citations: Optional[Dict[str, Any]] = None,
        confidences: Optional[Dict[str, Any]] = None,
        sources: Optional[Dict[str, Any]] = None,
    ):
        self.data = data
        self.success = success
        self.error_message = error_message
        self.raw_output = raw_output
        self.model_used = model_used

        # Build FieldResult map from data + citations + confidences + sources
        self._fields: Dict[str, FieldResult] = {}
        if data is not None:
            from natural_pdf.elements.element_collection import ElementCollection

            empty = ElementCollection([])
            data_dict = data.model_dump() if hasattr(data, "model_dump") else data.dict()
            citations = citations or {}
            confidences = confidences or {}
            sources = sources or {}
            for field_name, value in data_dict.items():
                # Null-valued fields get no confidence, sources, or citations
                if value is None:
                    self._fields[field_name] = FieldResult(
                        value=None,
                        citations=empty,
                        confidence=None,
                        sources=None,
                    )
                else:
                    self._fields[field_name] = FieldResult(
                        value=value,
                        citations=citations.get(field_name, empty),
                        confidence=confidences.get(field_name),
                        sources=sources.get(field_name),
                    )

    # ------------------------------------------------------------------
    # Value access via attribute
    # ------------------------------------------------------------------

    def __getattr__(self, name: str) -> Any:
        # Only intercept if we have fields and the name is a field
        if name.startswith("_") or name in (
            "data",
            "success",
            "error_message",
            "raw_output",
            "model_used",
            "_fields",
        ):
            raise AttributeError(name)
        fields = self.__dict__.get("_fields", {})
        if name in fields:
            return fields[name].value
        raise AttributeError(
            f"'{type(self).__name__}' has no field '{name}'. "
            f"Available fields: {list(fields.keys())}"
        )

    # ------------------------------------------------------------------
    # Dict-like interface (returns FieldResult)
    # ------------------------------------------------------------------

    def __getitem__(self, key: str) -> FieldResult:
        try:
            return self._fields[key]
        except KeyError:
            raise KeyError(
                f"Field '{key}' not found. " f"Available: {list(self._fields.keys())}"
            ) from None

    def __contains__(self, key: str) -> bool:
        return key in self._fields

    def __iter__(self) -> Iterator[str]:
        return iter(self._fields)

    def __len__(self) -> int:
        return len(self._fields)

    def keys(self):
        return self._fields.keys()

    def values(self):
        return self._fields.values()

    def items(self) -> Iterator[Tuple[str, FieldResult]]:
        return self._fields.items()

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------

    def to_dict(self, confidence: bool = True, citations: bool = False) -> Dict[str, Any]:
        """Return a plain dict of ``{field_name: value}``.

        When *confidence* is ``True`` (the default) and confidence scores
        exist, each field's confidence is included as a
        ``{field_name}_confidence`` key.

        When *citations* is ``True`` and raw source quotes exist, they are
        included as ``{field_name}_sources`` keys (lists of strings as
        returned by the LLM).  Defaults to ``False`` since source quotes
        can be verbose.

        Pass ``confidence=False`` to also omit confidence scores.
        """
        d: Dict[str, Any] = {}
        for name, fr in self._fields.items():
            d[name] = fr.value
            if confidence and fr.confidence is not None:
                d[f"{name}_confidence"] = fr.confidence
            if citations and fr.sources is not None:
                d[f"{name}_sources"] = fr.sources
        return d

    @property
    def citations(self) -> Dict[str, Any]:
        """Dict mapping field names to their citation ElementCollections."""
        return {name: fr.citations for name, fr in self._fields.items()}

    @property
    def confidences(self) -> Dict[str, Any]:
        """Dict mapping field names to their confidence scores."""
        return {name: fr.confidence for name, fr in self._fields.items()}

    @property
    def sources(self) -> Dict[str, Any]:
        """Dict mapping field names to their raw LLM source quote strings."""
        return {name: fr.sources for name, fr in self._fields.items()}

    def show(self, *, pages="cited", **kwargs):
        """Highlight all citation elements on page(s), labeled by field name.

        Each field's source elements are highlighted with the field name as
        the annotation label.  Returns the rendered image.  When citations
        span multiple pages, all affected pages are rendered.

        Args:
            pages: Which pages to include. ``"cited"`` (default) shows only
                pages with citation elements. ``"all"`` shows every page in
                the source PDF. A list of page numbers is reserved for future
                use and currently raises :class:`NotImplementedError`.
        """
        if isinstance(pages, list):
            raise NotImplementedError(
                "Passing a list of page numbers to pages= is not yet supported."
            )

        if not self._fields:
            return None

        # Collect all citation elements and track unique pages
        pages_seen = set()
        has_any = False
        for name, fr in self._fields.items():
            if fr.citations is not None and len(fr.citations) > 0:
                has_any = True
                label = build_enriched_label(name, fr.value)
                fr.citations.highlight(label=label)
                for elem in fr.citations:
                    if hasattr(elem, "page") and elem.page is not None:
                        pages_seen.add(elem.page)

        if not has_any or not pages_seen:
            logger.warning("No citation elements to show.")
            return None

        if pages == "all":
            # Find the source PDF and include all its pages
            source_pdf = None
            for p in pages_seen:
                if hasattr(p, "_pdf"):
                    source_pdf = p._pdf
                    break
            if source_pdf is not None:
                page_list = list(source_pdf.pages)
            else:
                page_list = sorted(pages_seen, key=lambda p: getattr(p, "index", 0))
        else:
            page_list = sorted(pages_seen, key=lambda p: getattr(p, "index", 0))

        if len(page_list) == 1:
            return page_list[0].show(**kwargs)

        # Multiple pages — render them all via a PageCollection
        from natural_pdf.core.page_collection import PageCollection

        collection = PageCollection(page_list)
        kwargs.setdefault("columns", 1)
        return collection.show(**kwargs)

    def save_pdf(self, path: str, *, pages="all", **kwargs) -> None:
        """Save an annotated PDF with highlight annotations for all citations.

        Each field's citation elements become ``/Highlight`` annotations on the
        corresponding PDF pages, preserving the native text layer.  Popup text
        includes the field name, extracted value, and confidence (if present).

        Args:
            path: Output file path for the annotated PDF.
            pages: Which pages to include. ``"all"`` (default) includes every
                page from the source PDF. ``"cited"`` includes only pages that
                have citation annotations. A list of page numbers is reserved
                for future use and currently raises :class:`NotImplementedError`.
            **kwargs: Forwarded to :func:`create_annotated_pdf`.
        """
        if isinstance(pages, list):
            raise NotImplementedError(
                "Passing a list of page numbers to pages= is not yet supported."
            )

        from natural_pdf.exporters.annotated_pdf import create_annotated_pdf

        create_annotated_pdf(self._fields, path, pages=pages, **kwargs)

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        if not self.success:
            return f"StructuredDataResult(success=False, error={self.error_message!r})"
        fields_preview = ", ".join(f"{k}={v.value!r}" for k, v in list(self._fields.items())[:3])
        if len(self._fields) > 3:
            fields_preview += ", ..."
        return f"StructuredDataResult({fields_preview})"
