from __future__ import annotations

from collections.abc import Mapping as MappingABC
from collections.abc import Sequence as SequenceABC
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence, Tuple, cast

from natural_pdf.core.interfaces import Bounds, HasPages, HasSinglePage, SupportsBBox
from natural_pdf.elements.base import extract_bbox

if TYPE_CHECKING:  # pragma: no cover
    from natural_pdf.core.page import Page
    from natural_pdf.elements.region import Region


def _unique_page(sequence: Sequence["Page"]) -> "Page":
    """Return the single unique page in the sequence or raise if ambiguous."""
    seen: dict[int, "Page"] = {}
    for page in sequence:
        seen.setdefault(id(page), page)

    if not seen:
        raise ValueError("Object does not reference any pages.")
    if len(seen) > 1:
        raise ValueError("Object spans multiple pages; cannot determine a single page context.")
    return next(iter(seen.values()))


def _ensure_bounds(bounds: Optional[Bounds]) -> Optional[Bounds]:
    if bounds is None:
        return None
    return cast(Bounds, tuple(float(coord) for coord in bounds))  # type: ignore[arg-type]


def _maybe_extract_bounds(obj: Any) -> Optional[Bounds]:
    """Attempt to extract bounds only when the object satisfies extract_bbox requirements."""
    if isinstance(obj, SupportsBBox):
        return extract_bbox(obj)

    if isinstance(obj, MappingABC):
        return extract_bbox(cast(Mapping[str, Any], obj))

    if isinstance(obj, SequenceABC):
        # Guard sequences that look like coordinate tuples
        if len(obj) >= 4:
            return extract_bbox(cast(Sequence[float], obj))

    return None


def resolve_page_context(obj: Any) -> Tuple["Page", Optional[Bounds]]:
    """
    Resolve a mixed layout/ocr context into a concrete Page and optional bounds.

    The function understands Pages, Regions, core interface protocols, and plain bbox-bearing
    objects that expose ``page``/``_page`` attributes.  A ``ValueError`` is raised when the page
    cannot be determined unambiguously.
    """
    from natural_pdf.core.page import Page
    from natural_pdf.elements.region import Region

    # Direct Page ---------------------------------------------------------
    if isinstance(obj, Page):
        page_bounds: Bounds = (0.0, 0.0, float(obj.width), float(obj.height))
        return obj, page_bounds

    # Region --------------------------------------------------------------
    if isinstance(obj, Region):
        page = obj.page
        if page is None:
            raise ValueError("Region is not associated with a Page.")
        return page, cast(Bounds, tuple(float(coord) for coord in obj.bbox))

    # Protocol helpers ----------------------------------------------------
    if isinstance(obj, HasSinglePage):
        page = obj.page
        if page is None:
            raise ValueError("Object exposing HasSinglePage does not reference a Page.")
        return page, _ensure_bounds(_maybe_extract_bounds(obj))

    if isinstance(obj, HasPages):
        page = _unique_page(obj.pages)
        return page, _ensure_bounds(_maybe_extract_bounds(obj))

    # Duck-typed fallbacks ------------------------------------------------
    page_candidate = getattr(obj, "page", None) or getattr(obj, "_page", None)
    if page_candidate is not None:
        if not isinstance(page_candidate, Page):
            raise ValueError(f"Resolved page attribute is not a Page: {type(page_candidate)}")
        return page_candidate, _ensure_bounds(_maybe_extract_bounds(obj))

    if isinstance(obj, SupportsBBox):
        bounds = extract_bbox(obj)
        page_candidate = getattr(obj, "page", None) or getattr(obj, "_page", None)
        if page_candidate is not None:
            if not isinstance(page_candidate, Page):
                raise ValueError(f"Resolved page attribute is not a Page: {type(page_candidate)}")
            return page_candidate, _ensure_bounds(bounds)

    raise ValueError(f"Cannot resolve page context from object of type {type(obj).__name__}.")
