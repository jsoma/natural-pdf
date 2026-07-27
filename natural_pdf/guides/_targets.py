"""Target and page-resolution helpers for guides build/extract paths."""

from __future__ import annotations

from collections.abc import Iterable as IterableABC
from collections.abc import Sequence as SequenceABC
from typing import TYPE_CHECKING, Any, Iterable, List, Tuple, cast

from natural_pdf.flows.region import FlowRegion

from .helpers import Bounds, _require_bounds, _resolve_single_page

if TYPE_CHECKING:
    from natural_pdf.core.page import Page
else:  # pragma: no cover
    Page = Any  # type: ignore[misc, assignment]


def iter_page_regions(page_obj: Any) -> List[Any]:
    iterator = getattr(page_obj, "iter_regions", None)
    regions: Any
    if callable(iterator):
        regions = iterator()
    else:
        regions = iterator

    if regions is None or not isinstance(regions, IterableABC):
        return []
    return list(cast(Iterable[Any], regions))


def resolve_page_for_materialization(obj: Any) -> "Page":
    if hasattr(obj, "add_element"):
        return cast("Page", obj)

    page_obj = getattr(obj, "_page", None) or getattr(obj, "page", None)
    if page_obj is None:
        try:
            page_obj = _resolve_single_page(obj)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Target object {obj} is not a Page or Region") from exc

    if page_obj is None or not hasattr(page_obj, "add_element"):
        raise ValueError(f"Target object {obj} is not a Page or Region")

    return cast("Page", page_obj)


def resolve_single_page_grid_target(obj: Any) -> Tuple["Page", Bounds]:
    if isinstance(obj, FlowRegion):
        raise ValueError(
            "FlowRegion targets require multi-page handling; use the outer build_grid dispatch."
        )

    bounds = _require_bounds(obj, context="grid target")
    page = resolve_page_for_materialization(obj)
    if not hasattr(page, "remove_element"):
        raise ValueError("Target page does not expose element registration helpers")

    return page, bounds


def resolve_cleanup_pages(obj: Any) -> List["Page"]:
    pages: List["Page"] = []
    pages_attr = getattr(obj, "pages", None)
    if isinstance(pages_attr, SequenceABC) and not isinstance(pages_attr, (str, bytes)):
        candidates = tuple(pages_attr)
    else:
        page = getattr(obj, "_page", None) or getattr(obj, "page", None)
        candidates = (page,) if page is not None else ()

    seen: set[int] = set()
    for candidate in candidates:
        if candidate is None or not hasattr(candidate, "remove_element"):
            continue
        marker = id(candidate)
        if marker in seen:
            continue
        seen.add(marker)
        pages.append(cast("Page", candidate))

    if not pages:
        pages.append(resolve_page_for_materialization(obj))

    return pages
