from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Literal, Sequence

from natural_pdf.elements.element_collection import ElementCollection

Side = Literal["right", "left", "both"]
ItemSource = str | Iterable[Any] | Callable[[Any], Iterable[Any]]


@dataclass(frozen=True)
class AnchoredRow:
    """Same-row content collected from a visual anchor.

    Attributes:
        anchor: The element that identified the row, such as a margin number or ID.
        elements: Text-like elements collected on the requested side of the anchor.
        text: Joined text extracted from ``elements`` in reading order.
        bbox: Union of the anchor and collected element bounds, when available.
        page_number: One-based page number for the anchor, when available.
    """

    anchor: Any
    elements: ElementCollection[Any]
    text: str
    bbox: tuple[float, float, float, float] | None
    page_number: int | None

    @property
    def words(self) -> ElementCollection[Any]:
        """Alias for text-like elements in the row."""

        return self.elements


def extract_anchored_rows(
    context: Any,
    anchors: ItemSource,
    *,
    content_selector: str = "text",
    elements: ItemSource | None = None,
    side: Side = "right",
    y_tolerance: float | None = None,
    x_gap: float = 0,
    include_anchor: bool = False,
    sort: bool = True,
    apply_exclusions: bool = True,
) -> list[AnchoredRow]:
    """Collect same-row content around anchor elements.

    This is intentionally small: it covers the common PDF task shape where a
    stable anchor such as a margin line number or first-column ID identifies the
    row, and nearby text on the same baseline is the row content.

    Args:
        context: Page-like object used to resolve selector/callable inputs.
        anchors: Selector, iterable, or callable returning anchor elements.
        content_selector: Selector used for row content when ``elements`` is not
            supplied.
        elements: Optional selector, iterable, or callable for candidate row
            content. Use this to pre-filter text to a table or section band.
        side: Which side of each anchor to collect: ``"right"``, ``"left"``, or
            ``"both"``.
        y_tolerance: Maximum vertical midpoint distance for same-row matching.
            When omitted, a conservative tolerance is derived from anchor height.
        x_gap: Required horizontal gap between anchor and content for left/right
            matching.
        include_anchor: Include the anchor itself in ``elements`` and ``text``.
        sort: Sort collected content by x-position before joining text.
        apply_exclusions: Respect page exclusions when resolving selector inputs.

    Returns:
        One :class:`AnchoredRow` per anchor, in anchor order.
    """

    anchor_items = _resolve_items(context, anchors, apply_exclusions=apply_exclusions)
    content_items = _resolve_items(
        context,
        elements if elements is not None else content_selector,
        apply_exclusions=apply_exclusions,
    )
    rows: list[AnchoredRow] = []

    for anchor in anchor_items:
        tolerance = y_tolerance if y_tolerance is not None else _default_y_tolerance(anchor)
        row_items = [
            item
            for item in content_items
            if (include_anchor or item is not anchor)
            and _same_row(anchor, item, tolerance)
            and _on_requested_side(anchor, item, side=side, x_gap=x_gap)
        ]
        if sort:
            row_items.sort(key=lambda item: (_x0(item), _top(item)))
        collection = ElementCollection(row_items, context=getattr(context, "_context", None))
        rows.append(
            AnchoredRow(
                anchor=anchor,
                elements=collection,
                text=_join_text(row_items),
                bbox=_union_bbox([anchor, *row_items]),
                page_number=getattr(getattr(anchor, "page", None), "number", None),
            )
        )

    return rows


def _resolve_items(
    context: Any,
    value: str | Iterable[Any] | Callable[[Any], Iterable[Any]],
    *,
    apply_exclusions: bool,
):
    if callable(value):
        return list(value(context))
    if isinstance(value, str):
        finder = getattr(context, "find_all", None)
        if finder is None:
            raise TypeError("selector inputs require a context with find_all()")
        return list(finder(value, apply_exclusions=apply_exclusions))
    return list(value)


def _same_row(anchor: Any, item: Any, tolerance: float) -> bool:
    return abs(_mid_y(anchor) - _mid_y(item)) <= tolerance


def _on_requested_side(anchor: Any, item: Any, *, side: Side, x_gap: float) -> bool:
    if side == "right":
        return _x0(item) >= _x1(anchor) + x_gap
    if side == "left":
        return _x1(item) <= _x0(anchor) - x_gap
    if side == "both":
        return True
    raise ValueError("side must be 'right', 'left', or 'both'")


def _default_y_tolerance(anchor: Any) -> float:
    return max(3.0, _height(anchor) * 0.75)


def _mid_y(item: Any) -> float:
    return (_top(item) + _bottom(item)) / 2


def _height(item: Any) -> float:
    return max(0.0, _bottom(item) - _top(item))


def _bbox(item: Any) -> Sequence[float]:
    bbox = getattr(item, "bbox", None)
    if bbox is None:
        return (_x0(item), _top(item), _x1(item), _bottom(item))
    return bbox


def _x0(item: Any) -> float:
    value = getattr(item, "x0", None)
    if value is not None:
        return float(value)
    return _bbox_value(item, 0)


def _top(item: Any) -> float:
    value = getattr(item, "top", None)
    if value is not None:
        return float(value)
    return _bbox_value(item, 1)


def _x1(item: Any) -> float:
    value = getattr(item, "x1", None)
    if value is not None:
        return float(value)
    return _bbox_value(item, 2)


def _bottom(item: Any) -> float:
    value = getattr(item, "bottom", None)
    if value is not None:
        return float(value)
    return _bbox_value(item, 3)


def _bbox_value(item: Any, index: int) -> float:
    bbox = getattr(item, "bbox", None)
    if bbox is None:
        raise AttributeError(f"item does not expose bbox or coordinate attributes: {item!r}")
    return float(bbox[index])


def _union_bbox(items: Sequence[Any]) -> tuple[float, float, float, float] | None:
    if not items:
        return None
    bboxes = [_bbox(item) for item in items]
    return (
        min(float(bbox[0]) for bbox in bboxes),
        min(float(bbox[1]) for bbox in bboxes),
        max(float(bbox[2]) for bbox in bboxes),
        max(float(bbox[3]) for bbox in bboxes),
    )


def _join_text(items: Sequence[Any]) -> str:
    parts: list[str] = []
    for item in items:
        extractor = getattr(item, "extract_text", None)
        if callable(extractor):
            text = extractor()
        else:
            text = getattr(item, "text", "")
            if callable(text):
                text = text()
        text = str(text).strip()
        if text:
            parts.append(text)
    return " ".join(parts)
