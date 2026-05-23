"""Guide system for table extraction and layout analysis."""

import logging
from collections import UserList
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Literal,
    Optional,
    Protocol,
    Sequence,
    SupportsIndex,
    Tuple,
    TypeGuard,
    Union,
    cast,
    overload,
)

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw

from natural_pdf.core.interfaces import HasPages, HasSinglePage
from natural_pdf.elements.element_collection import ElementCollection
from natural_pdf.elements.line import LineElement
from natural_pdf.elements.region import Region
from natural_pdf.flows.region import FlowRegion
from natural_pdf.guides.guides_provider import run_guides_detect

from ._axis_ops import apply_generation_result, set_axis_coordinates, update_axis_coordinates
from ._generation import (
    AxisGenerationResult,
    build_content_options,
    build_headers_options,
    build_line_options,
    build_stripes_options,
    build_whitespace_options,
    generate_axis_coordinates,
    generate_both_axis_coordinates,
    resolve_generation_context,
)
from ._grid_builder import build_single_page_grid
from ._grid_types import GridBuildCounts
from ._table_extract import extract_table_from_guides
from ._targets import resolve_page_for_materialization
from .flow_adapter import FlowGuideAdapter
from .grid_helpers import collect_constituent_pages, register_regions_with_pages
from .helpers import (
    BoolArray,
    Bounds,
    GuidesContext,
    IntArray,
    SupportsGuidesContext,
    _bounds_from_object,
    _collect_line_elements,
    _constituent_regions,
    _ensure_bounds_tuple,
    _has_size,
    _is_flow_region,
    _is_guides_context,
    _label_contiguous_regions,
    _normalize_markers,
    _require_bounds,
    _resolve_single_page,
    _SupportsSize,
)
from .separators import (
    find_min_crossing_separator,
    find_seam_carving_separator,
    stabilize_with_rows,
)
from .text_detect import (
    collect_text_elements,
    find_horizontal_element_gaps,
    find_horizontal_whitespace_gaps,
    find_vertical_element_gaps,
    find_vertical_whitespace_gaps,
)

if TYPE_CHECKING:
    from natural_pdf.core.page import Page
    from natural_pdf.core.page_collection import PageCollection
    from natural_pdf.elements.base import Element
    from natural_pdf.flows.region import FlowRegion

from natural_pdf.tables.result import TableResult

logger = logging.getLogger(__name__)

Bounds = Tuple[float, float, float, float]
OuterBoundaryMode = Union[bool, Literal["first", "last"]]
BoolArray = NDArray[np.bool_]
IntArray = NDArray[np.int_]

_OCR_WINDOW_DEBUG_COLORS: Tuple[Tuple[int, int, int, int], ...] = (
    (37, 99, 235, 34),  # blue
    (22, 163, 74, 34),  # green
    (124, 58, 237, 34),  # purple
    (8, 145, 178, 34),  # cyan
    (202, 138, 4, 34),  # amber
    (15, 118, 110, 34),  # teal
    (79, 70, 229, 34),  # indigo
    (101, 163, 13, 34),  # lime
)


@dataclass
class GuidesOcrResult:
    """Debug/result information for :meth:`Guides.apply_ocr`."""

    guides: Any = field(repr=False)
    resolution: int
    window: Union[str, Tuple[int, int], List[int]]
    windows: List[Dict[str, Any]]
    max_side_px: int
    max_area_px: int
    target_cell_px: int
    representative_percentile: float
    min_confidence: Optional[float]
    ran: bool
    counts: List[Optional[int]] = field(default_factory=list)
    target: Any = field(default=None, repr=False)

    def __len__(self) -> int:
        return len(self.windows)

    @property
    def total_created(self) -> Optional[int]:
        known_counts = [count for count in self.counts if count is not None]
        if not known_counts:
            return None
        return int(sum(known_counts))

    def summary(self) -> Dict[str, Any]:
        """Return a compact summary useful for notebooks/logging."""

        return {
            "ran": self.ran,
            "resolution": self.resolution,
            "window": self.window,
            "window_count": len(self.windows),
            "max_side_px": self.max_side_px,
            "max_area_px": self.max_area_px,
            "target_cell_px": self.target_cell_px,
            "representative_percentile": self.representative_percentile,
            "min_confidence": self.min_confidence,
            "total_created": self.total_created,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Return the full OCR plan/result as plain data."""

        data = self.summary()
        data["windows"] = list(self.windows)
        if self.counts:
            data["counts"] = list(self.counts)
        return data

    def extract_table(self, *args: Any, **kwargs: Any) -> TableResult:
        """Extract from the Guides instance that produced this OCR result."""

        return self.guides.extract_table(*args, **kwargs)

    @staticmethod
    def _as_list(value: Any) -> List[Any]:
        if value is None:
            return []
        if hasattr(value, "elements"):
            return list(value.elements)
        if isinstance(value, list):
            return value
        try:
            return list(value)
        except TypeError:
            return [value]

    @staticmethod
    def _bbox_center_in(
        bbox: Bounds,
        window_bbox: Bounds,
        *,
        tolerance: float = 0.25,
    ) -> bool:
        x0, top, x1, bottom = bbox
        wx0, wtop, wx1, wbottom = window_bbox
        cx = (x0 + x1) / 2.0
        cy = (top + bottom) / 2.0
        return (
            wx0 - tolerance <= cx <= wx1 + tolerance
            and wtop - tolerance <= cy <= wbottom + tolerance
        )

    @staticmethod
    def _bbox_intersects(
        bbox: Bounds,
        window_bbox: Bounds,
        *,
        tolerance: float = 0.25,
    ) -> bool:
        x0, top, x1, bottom = bbox
        wx0, wtop, wx1, wbottom = window_bbox
        return not (
            x1 < wx0 - tolerance
            or x0 > wx1 + tolerance
            or bottom < wtop - tolerance
            or top > wbottom + tolerance
        )

    @staticmethod
    def _padded_bbox(
        bbox: Bounds,
        *,
        padding: float,
        source: Any,
    ) -> Bounds:
        x0, top, x1, bottom = bbox
        page = getattr(source, "page", None) or source
        width = getattr(page, "width", None)
        height = getattr(page, "height", None)
        padded = (
            x0 - padding,
            top - padding,
            x1 + padding,
            bottom + padding,
        )
        if isinstance(width, (int, float)) and isinstance(height, (int, float)):
            return (
                max(0.0, padded[0]),
                max(0.0, padded[1]),
                min(float(width), padded[2]),
                min(float(height), padded[3]),
            )
        return padded

    @staticmethod
    def _combine_images(
        images: Sequence[Any],
        *,
        layout: Literal["grid", "stack"] = "grid",
        columns: int = 2,
        gap: int = 8,
    ) -> Any:
        valid_images = [image for image in images if image is not None]
        if not valid_images:
            return None
        if len(valid_images) == 1:
            return valid_images[0]

        columns = max(1, int(columns or 1))
        if layout == "stack":
            columns = 1

        rows = (len(valid_images) + columns - 1) // columns
        cell_width = max(image.width for image in valid_images)
        cell_height = max(image.height for image in valid_images)
        out_width = columns * cell_width + gap * (columns - 1)
        out_height = rows * cell_height + gap * (rows - 1)
        combined = Image.new("RGB", (out_width, out_height), "white")

        for idx, image in enumerate(valid_images):
            row, col = divmod(idx, columns)
            x = col * (cell_width + gap)
            y = row * (cell_height + gap)
            if image.mode not in ("RGB", "RGBA"):
                image = image.convert("RGB")
            combined.paste(image, (x, y))
        return combined

    def _show_source(self) -> Any:
        if self.target is not None and hasattr(self.target, "show"):
            return self.target
        if self.target is not None:
            try:
                return resolve_page_for_materialization(self.target)
            except Exception:
                pass
        context = getattr(self.guides, "context", None)
        if context is not None and hasattr(context, "show"):
            return context
        if context is not None:
            try:
                return resolve_page_for_materialization(context)
            except Exception:
                pass
        raise ValueError("Cannot show GuidesOcrResult without a visualizable OCR target.")

    @staticmethod
    def _window_color(
        idx: int,
        *,
        window_color: Optional[Any],
        window_colors: Optional[Sequence[Any]],
    ) -> Any:
        if window_color is not None:
            return window_color
        palette = window_colors or _OCR_WINDOW_DEBUG_COLORS
        if isinstance(palette, (str, bytes)):
            return palette
        if not palette:
            return "#2563eb"
        return palette[idx % len(palette)]

    @staticmethod
    def _render_source(source: Any, **kwargs: Any) -> Any:
        renderer = getattr(source, "render", None)
        if callable(renderer):
            return renderer(**kwargs)
        return source.show(**kwargs)

    def ocr_elements(self) -> List[Any]:
        """Return OCR text elements currently visible to the OCR target."""

        candidates: List[Any] = []
        if self.target is not None:
            candidates.append(self.target)
            page = getattr(self.target, "page", None) or getattr(self.target, "_page", None)
            if page is not None:
                candidates.append(page)
        context = getattr(self.guides, "context", None)
        if context is not None:
            candidates.append(context)

        seen: set[int] = set()
        for candidate in candidates:
            marker = id(candidate)
            if marker in seen:
                continue
            seen.add(marker)

            finder = getattr(candidate, "find_all", None)
            if not callable(finder):
                continue
            try:
                return self._as_list(finder("text[source=ocr]", apply_exclusions=False))
            except Exception:
                continue
        return []

    def window_text_elements(
        self,
        *,
        overlap: Literal["center", "partial"] = "center",
        tolerance: float = 0.25,
    ) -> List[List[Any]]:
        """Bucket OCR text elements by the OCR window that contains them."""

        buckets: List[List[Any]] = [[] for _ in self.windows]
        elements = self.ocr_elements()
        for element in elements:
            bbox = _bounds_from_object(element)
            if bbox is None:
                continue
            for idx, planned in enumerate(self.windows):
                window_bbox = _bounds_from_object(planned.get("bbox"))
                if window_bbox is None:
                    continue
                if overlap == "partial":
                    matches = self._bbox_intersects(bbox, window_bbox, tolerance=tolerance)
                else:
                    matches = self._bbox_center_in(bbox, window_bbox, tolerance=tolerance)
                if matches:
                    buckets[idx].append(element)
                    break
        return buckets

    @staticmethod
    def _text_attrs(element: Any) -> Dict[str, Any]:
        attrs: Dict[str, Any] = {}
        for name in ("text", "text_content"):
            value = getattr(element, name, None)
            if value:
                attrs["text"] = value
                break
        confidence = getattr(element, "confidence", None)
        if confidence is not None:
            attrs["confidence"] = confidence
        return attrs

    def _window_highlights(
        self,
        window_indices: Sequence[int],
        buckets: Sequence[Sequence[Any]],
        *,
        include_windows: bool,
        include_text: bool,
        window_color: Optional[Any],
        window_colors: Optional[Sequence[Any]],
        text_color: Any,
        empty_window_color: Optional[Any],
        window_line_width: float,
        window_fill: bool,
        annotate_text: bool,
    ) -> List[Dict[str, Any]]:
        highlights: List[Dict[str, Any]] = []
        for idx in window_indices:
            planned = self.windows[idx]
            window_bbox = _bounds_from_object(planned.get("bbox"))
            if window_bbox is None:
                continue
            texts = list(buckets[idx])
            if include_windows:
                color = self._window_color(
                    idx,
                    window_color=window_color,
                    window_colors=window_colors,
                )
                highlights.append(
                    {
                        "bbox": window_bbox,
                        "color": (
                            color if texts or empty_window_color is None else empty_window_color
                        ),
                        "label": f"window {idx + 1}",
                        "fill": window_fill,
                        "line_width": window_line_width,
                        "vertices": False,
                    }
                )
            if include_text:
                for element in texts:
                    entry: Dict[str, Any] = {
                        "element": element,
                        "color": text_color,
                        "label": f"window {idx + 1} OCR",
                    }
                    if annotate_text:
                        attrs = self._text_attrs(element)
                        if attrs:
                            entry["attributes_to_draw"] = attrs
                    highlights.append(entry)
        return highlights

    def show(
        self,
        *,
        window: Optional[Union[int, Sequence[int]]] = None,
        per_window: bool = False,
        overlap: Literal["center", "partial"] = "center",
        include_windows: bool = True,
        include_text: bool = True,
        window_color: Optional[Any] = None,
        window_colors: Optional[Sequence[Any]] = None,
        window_line_width: float = 2.0,
        window_fill: bool = True,
        text_color: Any = "red",
        empty_window_color: Optional[Any] = None,
        annotate_text: bool = False,
        crop_padding: float = 4.0,
        layout: Literal["grid", "stack"] = "grid",
        columns: int = 2,
        gap: int = 8,
        labels: bool = True,
        **kwargs: Any,
    ) -> Any:
        """Show OCR windows and the OCR text boxes assigned to each window.

        By default this renders all windows and OCR boxes together on the
        source page/region. Pass ``per_window=True`` to render one cropped
        panel per OCR window for closer inspection. ``window`` is zero-based
        and can be an int or a sequence of ints. Window boxes are lightly
        shaded, thick outlines; OCR text boxes are red.
        """

        source = self._show_source()
        if window is None:
            window_indices = list(range(len(self.windows)))
        elif isinstance(window, int):
            window_indices = [window]
        else:
            window_indices = list(window)

        for idx in window_indices:
            if idx < 0 or idx >= len(self.windows):
                raise IndexError(f"window index {idx} out of range for {len(self.windows)} windows")

        buckets = self.window_text_elements(overlap=overlap)

        if not per_window:
            highlights = self._window_highlights(
                window_indices,
                buckets,
                include_windows=include_windows,
                include_text=include_text,
                window_color=window_color,
                window_colors=window_colors,
                text_color=text_color,
                empty_window_color=empty_window_color,
                window_line_width=window_line_width,
                window_fill=window_fill,
                annotate_text=annotate_text,
            )
            return self._render_source(source, highlights=highlights, labels=labels, **kwargs)

        images = []
        for idx in window_indices:
            window_bbox = _bounds_from_object(self.windows[idx].get("bbox"))
            if window_bbox is None:
                continue
            highlights = self._window_highlights(
                [idx],
                buckets,
                include_windows=include_windows,
                include_text=include_text,
                window_color=window_color,
                window_colors=window_colors,
                text_color=text_color,
                empty_window_color=empty_window_color,
                window_line_width=window_line_width,
                window_fill=window_fill,
                annotate_text=annotate_text,
            )
            crop_bbox = self._padded_bbox(window_bbox, padding=crop_padding, source=source)
            images.append(
                self._render_source(
                    source,
                    highlights=highlights,
                    labels=labels,
                    crop_bbox=crop_bbox,
                    **kwargs,
                )
            )

        return self._combine_images(images, layout=layout, columns=columns, gap=gap)


class GuidesList(UserList[float]):
    """A list of guide coordinates that also provides methods for creating guides."""

    def __init__(
        self,
        parent_guides: "Guides",
        axis: Literal["vertical", "horizontal"],
        data: Optional[Iterable[float]] = None,
    ):
        self._parent = parent_guides
        self._axis: Literal["vertical", "horizontal"] = axis
        self._initializing = True
        super().__init__([])
        self._set_data_direct(data or [])
        self._initializing = False

    if TYPE_CHECKING:
        data: List[float]
    else:

        @property
        def data(self) -> List[float]:
            """Access the underlying coordinate list."""
            return cast(List[float], self.__dict__.setdefault("_data", []))

        @data.setter
        def data(self, value: Iterable[float]) -> None:
            if getattr(self, "_initializing", False):
                self._set_data_direct(value)
                return
            set_axis_coordinates(self._parent, self._axis, value)

    def _set_data_direct(self, value: Iterable[float], *, dedupe: bool = True) -> None:
        values = [float(v) for v in value] if value else []
        if dedupe:
            values = sorted(set(values))
        else:
            values = sorted(values)
        self.__dict__["_data"] = values

    def __setitem__(self, i, item):
        """Override to maintain sorted order."""
        values = list(self.data)
        if isinstance(i, slice):
            values[i] = [float(value) for value in item]
        else:
            values[int(i)] = float(item)
        set_axis_coordinates(self._parent, self._axis, values)

    def append(self, item):
        """Override to maintain sorted order."""
        update_axis_coordinates(self._parent, self._axis, [float(item)], append=True)

    def extend(self, other):
        """Override to maintain sorted order."""
        update_axis_coordinates(self._parent, self._axis, other, append=True)

    @overload
    def __getitem__(self, i: SupportsIndex) -> float: ...

    @overload
    def __getitem__(self, i: slice) -> "GuidesList": ...

    def __getitem__(self, i: Union[SupportsIndex, slice]) -> Union[float, "GuidesList"]:
        """Return float for indices and GuidesList for slices."""
        if isinstance(i, slice):
            return self.__class__(self._parent, self._axis, self.data[i])
        return self.data[int(i)]

    def insert(self, i, item):
        """Override to maintain sorted order."""
        update_axis_coordinates(self._parent, self._axis, [float(item)], append=True)

    def __iadd__(self, other):
        """Override to maintain sorted order."""
        update_axis_coordinates(self._parent, self._axis, other, append=True)
        return self

    def from_content(
        self,
        markers: Union[str, List[str], "ElementCollection", Callable, None],
        obj: Optional[GuidesContext] = None,
        align: Union[
            Literal["left", "right", "center", "between"], Literal["top", "bottom"]
        ] = "left",
        outer: OuterBoundaryMode = True,
        tolerance: float = 5,
        *,
        append: bool = False,
        apply_exclusions: bool = True,
    ) -> "Guides":
        """
        Create guides from content markers and add to this axis.

        Args:
            markers: Content to search for. Can be:
                - str: single selector (e.g., 'text:contains("Name")') or literal text
                - List[str]: list of selectors or literal text strings
                - ElementCollection: collection of elements to extract text from
                - Callable: function that takes a page and returns markers
                - None: no markers
            obj: Page/Region/FlowRegion to search (uses parent's context if None)
            align: How to align guides relative to found elements:
                - For vertical guides: 'left', 'right', 'center', 'between'
                - For horizontal guides: 'top', 'bottom', 'center', 'between'
                - Note: 'left'/'right' also work for horizontal (mapped to top/bottom)
            outer: Whether to add outer boundary guides
            tolerance: Tolerance for snapping to element edges
            apply_exclusions: Whether to apply exclusion zones when searching for text

        Returns:
            Parent Guides object for chaining
        """
        target_obj = resolve_generation_context(obj, self._parent.context)

        # Store callable markers for later evaluation
        if callable(markers):
            self._callable = markers
            # For now, evaluate with the current target object to get initial guides
            actual_markers = markers(target_obj)
        else:
            self._callable = None
            actual_markers = markers

        result = generate_axis_coordinates(
            axis=self._axis,
            method="content",
            context=target_obj,
            options=build_content_options(
                self._axis,
                actual_markers,
                align,
                outer,
                tolerance,
                apply_exclusions,
            ),
        )
        apply_generation_result(self._parent, result, append=append)
        return self._parent  # Return parent for chaining

    def from_lines(
        self,
        obj: Optional[GuidesContext] = None,
        threshold: Union[float, str] = "auto",
        source_label: Optional[str] = None,
        max_lines: Optional[int] = None,
        outer: bool = False,
        detection_method: str = "auto",
        resolution: int = 192,
        *,
        n: Optional[int] = None,
        min_gap: Optional[int] = None,
        append: bool = False,
        **detect_kwargs,
    ) -> "Guides":
        """
        Create guides from detected line elements.

        Args:
            obj: Page/Region/FlowRegion to search (uses parent's context if None)
            threshold: Line detection threshold ('auto' or float 0.0-1.0)
            source_label: Filter lines by source label (for vector method)
            max_lines: Maximum lines to use (alias: n)
            n: Convenience alias for max_lines. If provided, overrides max_lines.
            min_gap: Minimum pixel gap enforced between detected lines. Mapped to
                ``min_gap_h`` or ``min_gap_v`` depending on axis (ignored if those
                keys are already supplied via ``detect_kwargs``).
            outer: Whether to add outer boundary guides
            detection_method: 'auto' (default), 'vector', or 'pixels'. 'auto'
                uses vector line information when line elements exist and falls
                back to pixel detection otherwise.
            resolution: DPI for pixel-based detection (default: 192)
            **detect_kwargs: Additional parameters for pixel-based detection
                (e.g., min_gap_h, min_gap_v, binarization_method, etc.)

        Returns:
            Parent Guides object for chaining
        """
        target_obj = resolve_generation_context(obj, self._parent.context)

        # Resolve max_lines via alias `n` (n takes priority)
        if n is not None:
            if n <= 0:
                raise ValueError("n must be a positive integer")
            max_lines = n

        # Set appropriate max_lines parameter for underlying API
        max_lines_h = max_lines if self._axis == "horizontal" else None
        max_lines_v = max_lines if self._axis == "vertical" else None

        # Map generic `min_gap` to axis-specific argument expected by detection
        if min_gap is not None:
            if min_gap < 1:
                raise ValueError("min_gap must be ≥ 1 pixel")
            axis_key = "min_gap_h" if self._axis == "horizontal" else "min_gap_v"
            detect_kwargs.setdefault(axis_key, min_gap)

        result = generate_axis_coordinates(
            axis=self._axis,
            method="lines",
            context=target_obj,
            options=build_line_options(
                self._axis,
                threshold=threshold,
                source_label=source_label,
                max_lines_h=max_lines_h,
                max_lines_v=max_lines_v,
                outer=outer,
                detection_method=detection_method,
                resolution=resolution,
                detect_kwargs=detect_kwargs,
            ),
        )
        apply_generation_result(self._parent, result, append=append)
        return self._parent

    def from_whitespace(
        self,
        obj: Optional[GuidesContext] = None,
        min_gap: float = 10,
        *,
        append: bool = False,
    ) -> "Guides":
        target_obj = resolve_generation_context(obj, self._parent.context)
        result = generate_axis_coordinates(
            axis=self._axis,
            method="whitespace",
            context=target_obj,
            options=build_whitespace_options(min_gap),
        )
        apply_generation_result(self._parent, result, append=append)
        return self._parent

    def divide(self, n: int = 2, obj: Optional[Union["Page", "Region"]] = None) -> "Guides":
        """
        Divide the space evenly along this axis.

        Args:
            n: Number of divisions (creates n-1 guides)
            obj: Object to divide (uses parent's context if None)

        Returns:
            Parent Guides object for chaining
        """
        target_obj = obj or self._parent.context
        if target_obj is None:
            raise ValueError("No object provided and no context available")

        axis_literal: Literal["vertical", "horizontal"] = self._axis

        if _is_flow_region(target_obj):
            adapter = FlowGuideAdapter(self._parent)
            region_values: Dict[Any, Sequence[float]] = {}
            for region in adapter.regions:
                region_guides = Guides.divide(obj=region, n=n, axis=axis_literal)
                axis_values = (
                    [float(value) for value in region_guides.vertical]
                    if axis_literal == "vertical"
                    else [float(value) for value in region_guides.horizontal]
                )
                region_values[region] = axis_values

            apply_generation_result(
                self._parent,
                AxisGenerationResult(
                    axis=axis_literal,
                    region_coordinates={
                        region: [float(value) for value in values]
                        for region, values in region_values.items()
                    },
                ),
                append=False,
            )
            return self._parent
        else:
            # Create guides using divide
            new_guides = Guides.divide(
                obj=cast(Union["Page", "Region"], target_obj), n=n, axis=axis_literal
            )

            # Replace existing guides instead of extending (no append option here)
            axis_values = (
                new_guides.vertical if axis_literal == "vertical" else new_guides.horizontal
            )
            set_axis_coordinates(self._parent, axis_literal, axis_values)

        return self._parent

    def snap_to_whitespace(
        self,
        min_gap: float = 10.0,
        detection_method: str = "pixels",
        threshold: Union[float, str] = "auto",
        on_no_snap: str = "warn",
        obj: Optional[Union["Page", "Region"]] = None,
    ) -> "Guides":
        """
        Snap guides in this axis to whitespace gaps.

        Args:
            min_gap: Minimum gap size to consider
            detection_method: 'pixels' or 'text' for gap detection
            threshold: Threshold for whitespace detection (0.0-1.0) or 'auto'
            on_no_snap: What to do when snapping fails ('warn', 'raise', 'ignore')
            obj: Object to analyze (uses parent's context if None)

        Returns:
            Parent Guides object for chaining
        """
        target_obj = obj or self._parent.context
        if target_obj is None:
            raise ValueError("No object provided and no context available")

        # Use the parent's snap_to_whitespace but only for this axis
        original_horizontal: List[float] = []
        original_vertical: List[float] = []
        # Temporarily set the parent's guides to only this axis
        if self._axis == "vertical":
            original_horizontal = self._parent.horizontal.data.copy()
            self._parent.horizontal._set_data_direct([])
        else:
            original_vertical = self._parent.vertical.data.copy()
            self._parent.vertical._set_data_direct([])

        try:
            # Call the parent's method
            self._parent.snap_to_whitespace(
                axis=self._axis,
                min_gap=min_gap,
                detection_method=detection_method,
                threshold=threshold,
                on_no_snap=on_no_snap,
            )

            # Update our data from the parent
            if self._axis == "vertical":
                self._set_data_direct(self._parent.vertical.data.copy())
            else:
                self._set_data_direct(self._parent.horizontal.data.copy())

        finally:
            # Restore the other axis
            if self._axis == "vertical":
                self._parent.horizontal._set_data_direct(original_horizontal)
            else:
                self._parent.vertical._set_data_direct(original_vertical)

        return self._parent

    def snap_to_content(
        self,
        markers: Union[str, List[str], "ElementCollection", None] = "text",
        align: Literal["left", "right", "center"] = "left",
        tolerance: float = 5,
        obj: Optional[Union["Page", "Region"]] = None,
    ) -> "Guides":
        """
        Snap guides in this axis to nearby text content.

        Args:
            markers: Content to snap to. Can be:
                - str: single selector or literal text (default: 'text' for all text)
                - List[str]: list of selectors or literal text strings
                - ElementCollection: collection of elements
                - None: no markers (no snapping)
            align: How to align to the found text
            tolerance: Maximum distance to move when snapping
            obj: Object to search (uses parent's context if None)

        Returns:
            Parent Guides object for chaining
        """
        target_obj = obj or self._parent.context
        if target_obj is None:
            raise ValueError("No object provided and no context available")

        # Handle special case of 'text' as a selector for all text
        if markers == "text":
            # Get all text elements
            if hasattr(target_obj, "find_all"):
                text_elements = target_obj.find_all("text")
                if hasattr(text_elements, "elements"):
                    text_elements = text_elements.elements

                # Snap each guide to the nearest text element
                for i, guide_pos in enumerate(self.data):
                    best_distance = float("inf")
                    best_pos = guide_pos

                    for elem in text_elements:
                        # Calculate target position based on alignment
                        if self._axis == "vertical":
                            if align == "left":
                                elem_pos = elem.x0
                            elif align == "right":
                                elem_pos = elem.x1
                            else:  # center
                                elem_pos = (elem.x0 + elem.x1) / 2
                        else:  # horizontal
                            if align == "left":  # top for horizontal
                                elem_pos = elem.top
                            elif align == "right":  # bottom for horizontal
                                elem_pos = elem.bottom
                            else:  # center
                                elem_pos = (elem.top + elem.bottom) / 2

                        # Check if this is closer than current best
                        distance = abs(guide_pos - elem_pos)
                        if distance < best_distance and distance <= tolerance:
                            best_distance = distance
                            best_pos = elem_pos

                    # Update guide position if we found a good snap
                    if best_pos != guide_pos:
                        self.data[i] = best_pos
                        logger.debug(
                            f"Snapped {self._axis} guide from {guide_pos:.1f} to {best_pos:.1f}"
                        )
            else:
                logger.warning("Object does not support find_all for text snapping")
        else:
            # Original behavior for specific markers
            marker_texts = _normalize_markers(markers, target_obj)

            # Find each marker and snap guides
            for marker in marker_texts:
                if hasattr(target_obj, "find"):
                    element = target_obj.find(f'text:contains("{marker}")')
                    if not element:
                        logger.warning(f"Could not find text '{marker}' for snapping")
                        continue

                    # Determine target position based on alignment
                    if self._axis == "vertical":
                        if align == "left":
                            target_pos = element.x0
                        elif align == "right":
                            target_pos = element.x1
                        else:  # center
                            target_pos = (element.x0 + element.x1) / 2
                    else:  # horizontal
                        if align == "left":  # top for horizontal
                            target_pos = element.top
                        elif align == "right":  # bottom for horizontal
                            target_pos = element.bottom
                        else:  # center
                            target_pos = (element.top + element.bottom) / 2

                    # Find closest guide and snap if within tolerance
                    if self.data:
                        closest_idx = min(
                            range(len(self.data)), key=lambda i: abs(self.data[i] - target_pos)
                        )
                        if abs(self.data[closest_idx] - target_pos) <= tolerance:
                            self.data[closest_idx] = target_pos

        set_axis_coordinates(self._parent, self._axis, self.data)
        return self._parent

    def shift(self, index: int, offset: float) -> "Guides":
        """
        Move a specific guide in this axis by a offset amount.

        Args:
            index: Index of the guide to move
            offset: Amount to move (positive = right/down)

        Returns:
            Parent Guides object for chaining
        """
        if 0 <= index < len(self.data):
            values = list(self.data)
            values[index] += offset
            set_axis_coordinates(self._parent, self._axis, values)
        else:
            logger.warning(f"Guide index {index} out of range for {self._axis} axis")

        return self._parent

    def add(self, position: Union[float, List[float]]) -> "Guides":
        """
        Add one or more guides at the specified position(s).

        Args:
            position: Coordinate(s) to add guide(s) at. Can be:
                - float: single position
                - List[float]: multiple positions

        Returns:
            Parent Guides object for chaining
        """
        if isinstance(position, (list, tuple)):
            # Add multiple positions
            for pos in position:
                self.append(float(pos))
        else:
            # Add single position
            self.append(float(position))

        self.data.sort()
        return self._parent

    def remove_at(self, index: int) -> "Guides":
        """
        Remove a guide by index.

        Args:
            index: Index of guide to remove

        Returns:
            Parent Guides object for chaining
        """
        if 0 <= index < len(self.data):
            values = list(self.data)
            values.pop(index)
            set_axis_coordinates(self._parent, self._axis, values)
        return self._parent

    def clear_all(self) -> "Guides":
        """
        Remove all guides from this axis.

        Returns:
            Parent Guides object for chaining
        """
        set_axis_coordinates(self._parent, self._axis, [])
        return self._parent

    def from_headers(
        self,
        headers: Union["ElementCollection", List["Element"], List[str]],
        obj: Optional[Union["Page", "Region"]] = None,
        method: Literal["min_crossings", "seam_carving"] = "min_crossings",
        min_width: Optional[float] = None,
        max_width: Optional[float] = None,
        margin: float = 0.5,
        row_stabilization: bool = True,
        num_samples: int = 400,
        *,
        append: bool = False,
    ) -> "Guides":
        """Create vertical guides for columns based on headers and whitespace valleys.

        This method detects column boundaries by finding optimal vertical separators
        between headers that minimize text crossings, regardless of text alignment.

        Args:
            headers: Column header elements. Can be:
                - ElementCollection: collection of header elements
                - List[Element]: list of header elements
                - List[str]: list of header text to search for
            obj: Page/Region to analyze (uses parent's context if None)
            method: Detection method:
                - 'min_crossings': Fast vector-based minimum intersection count
                - 'seam_carving': Dynamic programming for curved boundaries
            min_width: Minimum column width constraint (pixels)
            max_width: Maximum column width constraint (pixels)
            margin: Buffer space from header edges when searching for separators (default: 0.5)
            row_stabilization: Whether to use row-wise median for stability
            num_samples: Number of x-positions to test per gap (for min_crossings)
            append: Whether to append to existing guides

        Returns:
            Parent Guides object for chaining

        Examples:
            # Create column guides from headers
            headers = page.find_all('text[size=16]')
            guides.vertical.from_headers(headers)

            # From header text strings
            guides.vertical.from_headers(["Statute", "Description", "Level", "Repeat"])

            # With width constraints
            guides.vertical.from_headers(headers, min_width=50, max_width=200)

            # Seam carving for complex layouts
            guides.vertical.from_headers(headers, method='seam_carving')
        """

        if self._axis != "vertical":
            raise ValueError("from_headers() only works for vertical guides (columns)")

        target_obj = resolve_generation_context(obj, self._parent.context)
        result = generate_axis_coordinates(
            axis="vertical",
            method="headers",
            context=target_obj,
            options=build_headers_options(
                headers,
                method=method,
                min_width=min_width,
                max_width=max_width,
                margin=margin,
                row_stabilization=row_stabilization,
                num_samples=num_samples,
            ),
        )
        apply_generation_result(self._parent, result, append=append)
        return self._parent

    @staticmethod
    def _find_min_crossing_separator(
        x0: float,
        x1: float,
        bboxes: List[Tuple[float, float, float, float]],
        num_samples: int,
    ) -> float:
        """Backward-compatible shim for separator helper."""
        return find_min_crossing_separator(x0, x1, bboxes, num_samples)

    @staticmethod
    def _find_seam_carving_separator(
        x0: float,
        x1: float,
        obj,  # Retained for compatibility, unused now
        header_y: float,
        page_bottom: float,
        bboxes: List[Tuple[float, float, float, float]],
    ) -> float:
        return find_seam_carving_separator(x0, x1, header_y, page_bottom, bboxes)

    @staticmethod
    def _stabilize_with_rows(
        separators: List[float],
        obj,
        bboxes: List[Tuple[float, float, float, float]],
        header_y: float,
    ) -> List[float]:
        return stabilize_with_rows(separators, bboxes, header_y)

    def from_stripes(
        self,
        stripes=None,
        color=None,  # Explicitly specify stripe color
    ) -> "Guides":
        """Create guides from striped table rows or columns.

        Creates guides at both edges of stripe elements (e.g., colored table rows).
        Perfect for zebra-striped tables where you need guides at every row boundary.

        Args:
            stripes: Elements representing stripes. If None, auto-detects.
            color: Specific color to look for (e.g., '#00ffff'). If None, finds most common.

        Examples:
            # Auto-detect zebra stripes
            guides.horizontal.from_stripes()

            # Specific color
            guides.horizontal.from_stripes(color='#00ffff')

            # Manual selection
            stripes = page.find_all('rect[fill=#00ffff]')
            guides.horizontal.from_stripes(stripes)

            # Vertical stripes
            guides.vertical.from_stripes(color='#e0e0e0')

        Returns:
            Parent Guides object for chaining
        """
        target_obj = resolve_generation_context(None, self._parent.context)
        result = generate_axis_coordinates(
            axis=self._axis,
            method="stripes",
            context=target_obj,
            options=build_stripes_options(stripes, color=color),
        )
        apply_generation_result(self._parent, result, append=True)
        return self._parent

    def __add__(self, other):
        """Handle addition of GuidesList objects by returning combined data."""
        if isinstance(other, GuidesList):
            return self.data + other.data
        elif isinstance(other, list):
            return self.data + other
        else:
            return NotImplemented


class Guides:
    """
    Manages vertical and horizontal guide lines for table extraction and layout analysis.

    Guides are collections of coordinates that can be used to define table boundaries,
    column positions, or general layout structures. They can be created through various
    detection methods or manually specified.

    Attributes:
        verticals: List of x-coordinates for vertical guide lines
        horizontals: List of y-coordinates for horizontal guide lines
        context: Optional Page/Region that these guides relate to
        bounds: Optional bounding box (x0, y0, x1, y1) for relative coordinate conversion
        snap_behavior: How to handle failed snapping operations ('warn', 'ignore', 'raise')
    """

    def __init__(
        self,
        verticals: Optional[Union[Iterable[float], GuidesContext]] = None,
        horizontals: Optional[Iterable[float]] = None,
        context: Optional[GuidesContext] = None,
        bounds: Optional[Tuple[float, float, float, float]] = None,
        relative: bool = False,
        snap_behavior: Literal["raise", "warn", "ignore"] = "warn",
    ):
        """
        Initialize a Guides object.

        Args:
            verticals: Iterable of x-coordinates for vertical guides, or a context object shorthand
            horizontals: Iterable of y-coordinates for horizontal guides
            context: Object providing spatial context (page, region, flow, etc.)
            bounds: Bounding box (x0, top, x1, bottom) if context not provided
            relative: Whether coordinates are relative (0-1) or absolute
            snap_behavior: How to handle snapping conflicts ('raise', 'warn', or 'ignore')
        """
        context_obj = context
        vertical_seed: Iterable[float] = ()

        # Handle Guides(page) or Guides(flow_region) shorthand
        if (
            verticals is not None
            and horizontals is None
            and context_obj is None
            and _is_guides_context(verticals)
        ):
            context_obj = cast(GuidesContext, verticals)
        elif verticals is not None:
            vertical_seed = cast(Iterable[float], verticals)

        self.context = context_obj
        coerced_bounds = _bounds_from_object(bounds) if bounds is not None else None
        self.bounds: Optional[Bounds] = coerced_bounds
        self.relative = relative
        self.snap_behavior = snap_behavior
        # Backwards compatibility alias for legacy options
        self.on_no_snap = snap_behavior
        self._ocr_applied = False
        self._ocr_prefer_words = False
        self._last_ocr_plan: Optional[Dict[str, Any]] = None
        self._last_ocr_result: Optional[GuidesOcrResult] = None

        # Check if we're dealing with a FlowRegion
        self.is_flow_region = _is_flow_region(context_obj)

        # If FlowRegion, we'll store guides per constituent region
        if self.is_flow_region:
            self._flow_guides: Dict["Region", Tuple[List[float], List[float]]] = {}
            # For unified view across all regions
            self._unified_vertical: List[Tuple[float, "Region"]] = []
            self._unified_horizontal: List[Tuple[float, "Region"]] = []
            # Cache for sorted unique coordinates
            self._vertical_cache: Optional[List[float]] = None
            self._horizontal_cache: Optional[List[float]] = None

        # Initialize with GuidesList instances
        horizontal_seed: Iterable[float] = horizontals if horizontals is not None else ()
        self._vertical = GuidesList(
            self,
            "vertical",
            sorted([float(x) for x in vertical_seed]),
        )
        self._horizontal = GuidesList(
            self,
            "horizontal",
            sorted([float(y) for y in horizontal_seed]),
        )

        # Determine bounds from context if needed
        if self.bounds is None and self.context is not None:
            self.bounds = _bounds_from_object(self.context)

        # Convert relative to absolute if needed
        if self.relative and self.bounds is not None:
            x0, top, x1, bottom = self.bounds
            width = x1 - x0
            height = bottom - top

            self._vertical._set_data_direct([x0 + float(v) * width for v in self._vertical])
            self._horizontal._set_data_direct([top + float(h) * height for h in self._horizontal])
            self.relative = False

    def _extract_with_table_service(self, host, **kwargs) -> TableResult:
        """Helper to route all table extraction through host helpers when possible."""
        extractor = getattr(host, "extract_table", None)
        if callable(extractor):
            extracted = extractor(**kwargs)
            if isinstance(extracted, TableResult):
                return extracted
            rows_iter = cast(Optional[Iterable[List[Any]]], extracted)
            rows: List[List[Any]] = list(rows_iter or [])
            return TableResult(rows)

        from natural_pdf.services.base import resolve_service

        return resolve_service(host, "table").extract_table(host, **kwargs)

    def _flow_context(self) -> FlowRegion:
        if not _is_flow_region(self.context):
            raise AttributeError("Flow context is not available for these guides")
        return cast(FlowRegion, self.context)

    def _flow_constituent_regions(self) -> Sequence["Region"]:
        return _constituent_regions(self._flow_context())

    @property
    def vertical(self) -> GuidesList:
        """Get vertical guide coordinates."""
        if self.is_flow_region and self._vertical_cache is not None:
            # Return cached unified view
            self._vertical._set_data_direct(self._vertical_cache)
        elif self.is_flow_region and self._unified_vertical:
            # Build unified view from flow guides
            all_verticals = []
            for coord, region in self._unified_vertical:
                all_verticals.append(coord)
            # Remove duplicates and sort
            self._vertical_cache = sorted(list(set(all_verticals)))
            self._vertical._set_data_direct(self._vertical_cache)
        return self._vertical

    @vertical.setter
    def vertical(self, value: Union[List[float], "Guides", None]):
        """Set vertical guides from a list of coordinates or another Guides object."""
        if value is None:
            set_axis_coordinates(self, "vertical", [])
        elif isinstance(value, Guides):
            # Extract vertical coordinates from another Guides object
            set_axis_coordinates(self, "vertical", value.vertical)
        elif isinstance(value, str):
            # Explicitly reject strings to avoid confusing iteration over characters
            raise TypeError(
                f"vertical cannot be a string, got '{value}'. Use a list of coordinates or Guides object."
            )
        elif hasattr(value, "__iter__"):
            # Handle list/tuple of coordinates
            try:
                set_axis_coordinates(self, "vertical", value)
            except (ValueError, TypeError) as e:
                raise TypeError(f"vertical must contain numeric values, got {value}: {e}")
        else:
            raise TypeError(f"vertical must be a list, Guides object, or None, got {type(value)}")

    @property
    def horizontal(self) -> GuidesList:
        """Get horizontal guide coordinates."""
        if self.is_flow_region and self._horizontal_cache is not None:
            # Return cached unified view
            self._horizontal._set_data_direct(self._horizontal_cache)
        elif self.is_flow_region and self._unified_horizontal:
            # Build unified view from flow guides
            all_horizontals = []
            for coord, region in self._unified_horizontal:
                all_horizontals.append(coord)
            # Remove duplicates and sort
            self._horizontal_cache = sorted(list(set(all_horizontals)))
            self._horizontal._set_data_direct(self._horizontal_cache)
        return self._horizontal

    @horizontal.setter
    def horizontal(self, value: Union[List[float], "Guides", None]):
        """Set horizontal guides from a list of coordinates or another Guides object."""
        if value is None:
            set_axis_coordinates(self, "horizontal", [])
        elif isinstance(value, Guides):
            # Extract horizontal coordinates from another Guides object
            set_axis_coordinates(self, "horizontal", value.horizontal)
        elif isinstance(value, str):
            # Explicitly reject strings
            raise TypeError(
                f"horizontal cannot be a string, got '{value}'. Use a list of coordinates or Guides object."
            )
        elif hasattr(value, "__iter__"):
            # Handle list/tuple of coordinates
            try:
                set_axis_coordinates(self, "horizontal", value)
            except (ValueError, TypeError) as e:
                raise TypeError(f"horizontal must contain numeric values, got {value}: {e}")
        else:
            raise TypeError(f"horizontal must be a list, Guides object, or None, got {type(value)}")

    def _get_context_bounds(self) -> Tuple[float, float, float, float]:
        """Return bounding box for the current context, ensuring it exists."""
        if self.context is None:
            raise ValueError("No context available for bounds computation")
        return _require_bounds(self.context, context="guide context")

    def from_headers_and_row_anchors(
        self,
        headers: Union["ElementCollection", Sequence[Any], None],
        row_anchors: Union[
            str, "ElementCollection", Sequence[Any], Callable[[GuidesContext], Iterable[Any]]
        ],
        *,
        header_anchor: Optional[Any] = None,
        obj: Optional[GuidesContext] = None,
        header_method: Literal["min_crossings", "seam_carving"] = "min_crossings",
        min_width: Optional[float] = None,
        max_width: Optional[float] = None,
        margin: float = 0.5,
        row_stabilization: bool = True,
        num_samples: int = 400,
        snap_vertical: bool = True,
        snap_vertical_kwargs: Optional[Dict[str, Any]] = None,
        row_align: Union[
            Literal["left", "right", "center", "between"],
            Literal["top", "bottom"],
        ] = "between",
        row_outer: Union[bool, Literal["first", "last"]] = True,
        row_tolerance: float = 5,
        apply_exclusions: bool = True,
    ) -> "Guides":
        """Build table guides from column headers and stable row anchors.

        Use this for crowded or borderless native-text tables where headers
        define columns and a first-column ID, case number, or similar marker
        defines each row. The helper intentionally composes the existing guide
        primitives: vertical guides from headers, optional whitespace snapping,
        and horizontal guides from row-anchor content.

        Args:
            headers: Header-row elements used to derive vertical column guides.
                Pass the visible table headers, not output schema names.
            row_anchors: Selector, elements, or callable identifying stable row
                markers such as first-column IDs or case numbers.
            header_anchor: Optional header marker to include as the first
                horizontal guide marker, keeping the header row in the grid.
            obj: Optional page/region/flow context. Defaults to this guide
                object's context.
            header_method: Strategy passed to ``vertical.from_headers(...)``.
            min_width: Optional minimum column width for header-derived guides.
            max_width: Optional maximum column width for header-derived guides.
            margin: Header-search margin used by ``from_headers``.
            row_stabilization: Stabilize header separators with nearby row text.
            num_samples: Sample count used by seam/min-crossing guide detection.
            snap_vertical: Whether to snap vertical guides into whitespace gaps.
            snap_vertical_kwargs: Options for ``vertical.snap_to_whitespace``.
            row_align: Alignment mode for row-anchor horizontal guides.
            row_outer: Whether to add outer horizontal boundary guides.
            row_tolerance: Tolerance for resolving row-anchor content.
            apply_exclusions: Respect exclusions when resolving row-anchor
                selectors.

        Returns:
            This ``Guides`` object, with vertical and horizontal guides populated.
        """

        target_obj = resolve_generation_context(obj, self.context)
        self.vertical.from_headers(
            headers,
            obj=target_obj,
            method=header_method,
            min_width=min_width,
            max_width=max_width,
            margin=margin,
            row_stabilization=row_stabilization,
            num_samples=num_samples,
        )
        if snap_vertical:
            snap_options = {
                "min_gap": 2,
                "detection_method": "text",
                "on_no_snap": "ignore",
            }
            if snap_vertical_kwargs:
                snap_options.update(snap_vertical_kwargs)
            self.vertical.snap_to_whitespace(obj=target_obj, **snap_options)

        row_markers = self._resolve_row_anchor_markers(
            target_obj,
            row_anchors,
            header_anchor=header_anchor,
            apply_exclusions=apply_exclusions,
        )
        self.horizontal.from_content(
            row_markers,
            obj=target_obj,
            align=row_align,
            outer=row_outer,
            tolerance=row_tolerance,
            apply_exclusions=apply_exclusions,
        )
        return self

    @staticmethod
    def _resolve_row_anchor_markers(
        target_obj: GuidesContext,
        row_anchors: Union[
            str, "ElementCollection", Sequence[Any], Callable[[GuidesContext], Iterable[Any]]
        ],
        *,
        header_anchor: Optional[Any],
        apply_exclusions: bool,
    ) -> list[Any]:
        markers: list[Any] = []
        if header_anchor is not None:
            markers.extend(
                Guides._resolve_marker_items(
                    target_obj,
                    header_anchor,
                    find_one=True,
                    apply_exclusions=apply_exclusions,
                )
            )
        markers.extend(
            Guides._resolve_marker_items(
                target_obj,
                row_anchors,
                find_one=False,
                apply_exclusions=apply_exclusions,
            )
        )
        return markers

    @staticmethod
    def _resolve_marker_items(
        target_obj: GuidesContext,
        markers: Any,
        *,
        find_one: bool,
        apply_exclusions: bool,
    ) -> list[Any]:
        if callable(markers):
            return list(markers(target_obj))
        if isinstance(markers, str):
            if find_one:
                finder = getattr(target_obj, "find", None)
                if finder is None:
                    return []
                item = finder(markers, apply_exclusions=apply_exclusions)
                return [item] if item is not None else []
            finder_all = getattr(target_obj, "find_all", None)
            if finder_all is None:
                return []
            return list(finder_all(markers, apply_exclusions=apply_exclusions))
        if isinstance(markers, ElementCollection):
            return list(markers)
        if _bounds_from_object(markers) is not None or hasattr(markers, "x0"):
            return [markers]
        try:
            return list(markers)
        except TypeError:
            return [markers]

    @property
    def last_ocr_result(self) -> Optional[GuidesOcrResult]:
        """Most recent result/plan returned by :meth:`apply_ocr`, if any."""

        return self._last_ocr_result

    def _ocr_boundaries(
        self,
        target_obj: GuidesContext,
        *,
        include_outer_boundaries: bool = False,
    ) -> Tuple[List[float], List[float]]:
        """Return guide boundaries for guide-window OCR."""

        verticals = sorted(float(v) for v in self.vertical)
        horizontals = sorted(float(h) for h in self.horizontal)

        if include_outer_boundaries:
            x0, top, x1, bottom = _require_bounds(target_obj, context="OCR guide target")
            if not verticals or verticals[0] > x0:
                verticals.insert(0, x0)
            if not verticals or verticals[-1] < x1:
                verticals.append(x1)
            if not horizontals or horizontals[0] > top:
                horizontals.insert(0, top)
            if not horizontals or horizontals[-1] < bottom:
                horizontals.append(bottom)

        return sorted(set(verticals)), sorted(set(horizontals))

    @staticmethod
    def _trimmed_percentile(values: Sequence[float], percentile: float) -> Optional[float]:
        """Percentile after dropping extreme tiny/large layout outliers."""

        valid = sorted(float(value) for value in values if np.isfinite(value) and value > 0)
        if not valid:
            return None

        median = float(np.percentile(valid, 50))
        if median <= 0:
            return None

        lower = max(1e-6, median * 0.1)
        upper = median * 3.0
        trimmed = [value for value in valid if lower <= value <= upper]
        if not trimmed:
            trimmed = valid

        return float(np.percentile(trimmed, percentile))

    @staticmethod
    def _ocr_cell_text_boxes(
        verticals: Sequence[float], horizontals: Sequence[float]
    ) -> List[float]:
        """Return min(width, height) for each guide cell as a text-size proxy."""

        boxes: List[float] = []
        for col_idx in range(len(verticals) - 1):
            width = float(verticals[col_idx + 1] - verticals[col_idx])
            if width <= 0:
                continue
            for row_idx in range(len(horizontals) - 1):
                height = float(horizontals[row_idx + 1] - horizontals[row_idx])
                if height <= 0:
                    continue
                boxes.append(min(width, height))
        return boxes

    def _resolve_ocr_resolution_for_guides(
        self,
        verticals: Sequence[float],
        horizontals: Sequence[float],
        *,
        resolution: Optional[int],
        target_cell_px: int,
        representative_percentile: float,
        min_resolution: int,
        max_resolution: int,
    ) -> int:
        """Resolve OCR DPI from cell geometry when the caller did not provide one."""

        if resolution is not None:
            return int(resolution)

        representative = self._trimmed_percentile(
            self._ocr_cell_text_boxes(verticals, horizontals),
            representative_percentile,
        )
        if representative is None or representative <= 0:
            return int(min_resolution)

        resolved = int(round((float(target_cell_px) * 72.0) / representative))
        return max(int(min_resolution), min(int(max_resolution), resolved))

    @staticmethod
    def _ocr_window_budget(
        engine: Optional[str],
        *,
        max_side_px: Optional[int],
        max_area_px: Optional[int],
    ) -> Tuple[int, int]:
        """Return conservative rendered-image limits for guide-window OCR."""

        if max_side_px is not None and max_area_px is not None:
            return int(max_side_px), int(max_area_px)

        engine_name = (engine or "").strip().lower()
        if engine_name in {"vlm", "paddlevl", "glm_ocr", "dots", "chandra"}:
            default_side = 2000
            default_area = 4_000_000
        else:
            default_side = 1600
            default_area = 2_500_000

        return int(max_side_px or default_side), int(max_area_px or default_area)

    @staticmethod
    def _guide_runs(values: Sequence[Any]) -> List[Tuple[int, int]]:
        """Return contiguous runs of equal values as (start, stop)."""

        if not values:
            return []

        runs: List[Tuple[int, int]] = []
        start = 0
        current = values[0]
        for idx, value in enumerate(values[1:], start=1):
            if value == current:
                continue
            runs.append((start, idx))
            start = idx
            current = value
        runs.append((start, len(values)))
        return runs

    def _plan_auto_ocr_windows(
        self,
        verticals: Sequence[float],
        horizontals: Sequence[float],
        *,
        resolution: int,
        max_side_px: int,
        max_area_px: int,
        vertical_ratio: float,
        mixed_cell_threshold: float,
        large_cell_ratio: float,
    ) -> List[Dict[str, Any]]:
        """Plan non-overlapping, cell-aligned OCR windows under a pixel budget."""

        num_cols = len(verticals) - 1
        num_rows = len(horizontals) - 1
        if num_cols <= 0 or num_rows <= 0:
            return []

        scale = float(resolution) / 72.0

        row_heights = [
            float(horizontals[row_idx + 1] - horizontals[row_idx]) for row_idx in range(num_rows)
        ]
        representative_row_height = self._trimmed_percentile(row_heights, 50.0)
        if representative_row_height is None:
            representative_row_height = max(row_heights) if row_heights else 0.0

        verticalish: List[List[bool]] = []
        for row_idx in range(num_rows):
            row_flags: List[bool] = []
            height = row_heights[row_idx]
            for col_idx in range(num_cols):
                width = float(verticals[col_idx + 1] - verticals[col_idx])
                row_flags.append(height > width * float(vertical_ratio))
            verticalish.append(row_flags)

        row_classes = [
            (
                (sum(1 for flag in row if flag) / max(1, len(row))) >= mixed_cell_threshold,
                row_heights[row_idx] > representative_row_height * float(large_cell_ratio),
            )
            for row_idx, row in enumerate(verticalish)
        ]

        windows: List[Dict[str, Any]] = []

        def width_px(col_start: int, col_stop: int) -> float:
            return (float(verticals[col_stop]) - float(verticals[col_start])) * scale

        def height_px(row_start: int, row_stop: int) -> float:
            return (float(horizontals[row_stop]) - float(horizontals[row_start])) * scale

        for row_band_start, row_band_stop in self._guide_runs(row_classes):
            band_rows = max(1, row_band_stop - row_band_start)
            col_classes: List[bool] = []
            for col_idx in range(num_cols):
                count = sum(
                    1
                    for row_idx in range(row_band_start, row_band_stop)
                    if verticalish[row_idx][col_idx]
                )
                col_classes.append((count / band_rows) >= mixed_cell_threshold)

            for col_run_start, col_run_stop in self._guide_runs(col_classes):
                col_start = col_run_start
                while col_start < col_run_stop:
                    col_stop = col_start + 1
                    while col_stop < col_run_stop:
                        candidate_stop = col_stop + 1
                        if width_px(col_start, candidate_stop) > max_side_px:
                            break
                        col_stop = candidate_stop

                    row_start = row_band_start
                    while row_start < row_band_stop:
                        row_stop = row_start + 1
                        while row_stop < row_band_stop:
                            candidate_stop = row_stop + 1
                            candidate_width = width_px(col_start, col_stop)
                            candidate_height = height_px(row_start, candidate_stop)
                            if candidate_height > max_side_px:
                                break
                            if candidate_width * candidate_height > max_area_px:
                                break
                            row_stop = candidate_stop

                        windows.append(
                            self._ocr_window_dict(
                                verticals, horizontals, col_start, col_stop, row_start, row_stop
                            )
                        )
                        row_start = row_stop

                    col_start = col_stop

        return windows

    @staticmethod
    def _ocr_window_dict(
        verticals: Sequence[float],
        horizontals: Sequence[float],
        col_start: int,
        col_stop: int,
        row_start: int,
        row_stop: int,
    ) -> Dict[str, Any]:
        """Build a serializable OCR window description."""

        return {
            "cols": (int(col_start), int(col_stop)),
            "rows": (int(row_start), int(row_stop)),
            "bbox": (
                float(verticals[col_start]),
                float(horizontals[row_start]),
                float(verticals[col_stop]),
                float(horizontals[row_stop]),
            ),
        }

    def _plan_ocr_windows(
        self,
        verticals: Sequence[float],
        horizontals: Sequence[float],
        *,
        window: Union[str, Tuple[int, int], List[int]],
        resolution: int,
        max_side_px: int,
        max_area_px: int,
        vertical_ratio: float,
        mixed_cell_threshold: float,
        large_cell_ratio: float,
    ) -> List[Dict[str, Any]]:
        """Plan guide-cell OCR windows."""

        num_cols = len(verticals) - 1
        num_rows = len(horizontals) - 1
        if num_cols <= 0 or num_rows <= 0:
            raise ValueError(
                "Guides must contain at least two vertical and two horizontal boundaries for OCR."
            )

        if window == "table":
            return [self._ocr_window_dict(verticals, horizontals, 0, num_cols, 0, num_rows)]

        if window == "cell":
            return [
                self._ocr_window_dict(verticals, horizontals, col, col + 1, row, row + 1)
                for row in range(num_rows)
                for col in range(num_cols)
            ]

        if isinstance(window, (tuple, list)) and len(window) == 2:
            cols_per_window = max(1, int(window[0]))
            rows_per_window = max(1, int(window[1]))
            windows: List[Dict[str, Any]] = []
            for row_start in range(0, num_rows, rows_per_window):
                row_stop = min(num_rows, row_start + rows_per_window)
                for col_start in range(0, num_cols, cols_per_window):
                    col_stop = min(num_cols, col_start + cols_per_window)
                    windows.append(
                        self._ocr_window_dict(
                            verticals,
                            horizontals,
                            col_start,
                            col_stop,
                            row_start,
                            row_stop,
                        )
                    )
            return windows

        if window != "auto":
            raise ValueError("window must be 'auto', 'table', 'cell', or a (cols, rows) tuple.")

        return self._plan_auto_ocr_windows(
            verticals,
            horizontals,
            resolution=resolution,
            max_side_px=max_side_px,
            max_area_px=max_area_px,
            vertical_ratio=vertical_ratio,
            mixed_cell_threshold=mixed_cell_threshold,
            large_cell_ratio=large_cell_ratio,
        )

    @staticmethod
    def _create_ocr_window_region(target_obj: GuidesContext, bbox: Bounds) -> "Region":
        """Create a page-backed Region for an OCR window without registering it."""

        x0, top, x1, bottom = bbox
        if isinstance(target_obj, Region):
            return target_obj.create_region(x0, top, x1, bottom, relative=False)

        creator = getattr(target_obj, "create_region", None)
        if callable(creator):
            return creator(x0, top, x1, bottom)

        page = resolve_page_for_materialization(target_obj)
        return page.create_region(x0, top, x1, bottom)

    @staticmethod
    def _annotate_ocr_windows(windows: List[Dict[str, Any]], *, resolution: int) -> None:
        """Add rendered pixel dimensions to planned OCR windows in-place."""

        scale = float(resolution) / 72.0
        for planned in windows:
            x0, top, x1, bottom = planned["bbox"]
            width_px = int(round((float(x1) - float(x0)) * scale))
            height_px = int(round((float(bottom) - float(top)) * scale))
            planned["image_size"] = (width_px, height_px)
            planned["area_px"] = width_px * height_px

    @staticmethod
    def _count_ocr_elements(target_obj: GuidesContext) -> Optional[int]:
        """Count OCR text elements on a target, when selector APIs are available."""

        candidates: List[Any] = [target_obj]
        page = getattr(target_obj, "page", None) or getattr(target_obj, "_page", None)
        if page is not None:
            candidates.append(page)

        for candidate in candidates:
            finder = getattr(candidate, "find_all", None)
            if not callable(finder):
                continue
            try:
                return len(finder("text[source=ocr]", apply_exclusions=False))
            except Exception:
                continue
        return None

    def apply_ocr(
        self,
        target: Optional[GuidesContext] = None,
        *,
        engine: Optional[str] = None,
        options: Optional[Any] = None,
        languages: Optional[List[str]] = None,
        min_confidence: Optional[float] = None,
        device: Optional[str] = None,
        resolution: Optional[int] = None,
        window: Union[str, Tuple[int, int], List[int]] = "auto",
        replace: Union[bool, str] = "ocr",
        include_outer_boundaries: bool = False,
        target_cell_px: int = 40,
        representative_percentile: float = 25.0,
        min_resolution: int = 150,
        max_resolution: int = 400,
        max_side_px: Optional[int] = None,
        max_area_px: Optional[int] = None,
        vertical_ratio: float = 2.0,
        mixed_cell_threshold: float = 0.5,
        large_cell_ratio: float = 2.0,
        detect_only: bool = False,
        apply_exclusions: bool = True,
        prefer_words: bool = True,
        show_progress: bool = True,
        dry_run: bool = False,
        **kwargs,
    ) -> GuidesOcrResult:
        """Apply OCR to cell-aligned guide windows and add text to the page.

        This is a guide-scoped OCR pre-pass for tables whose full-page or
        full-table OCR is too large/dense for the OCR engine.  The default
        ``window="auto"`` chooses non-overlapping rectangular groups of whole
        cells, estimates DPI from the trimmed 25th percentile of
        ``min(cell_width, cell_height)``, and keeps rendered crops under a
        conservative engine budget.

        Returns a :class:`GuidesOcrResult` containing the planned windows and
        OCR settings.  After a non-dry run, ``extract_table()`` on either the
        result or the same Guides instance will prefer word-based cell
        assignment unless the caller explicitly passes ``cell_extract`` /
        ``cell_overlap``.
        """

        target_obj = target if target is not None else self.context
        if target_obj is None:
            raise ValueError(
                "No target object available. Provide target or initialize Guides with a context."
            )
        if _is_flow_region(target_obj):
            raise ValueError(
                "guides.apply_ocr() currently supports single-page Page/Region targets only."
            )

        verticals, horizontals = self._ocr_boundaries(
            target_obj,
            include_outer_boundaries=include_outer_boundaries,
        )
        resolved_resolution = self._resolve_ocr_resolution_for_guides(
            verticals,
            horizontals,
            resolution=resolution,
            target_cell_px=target_cell_px,
            representative_percentile=representative_percentile,
            min_resolution=min_resolution,
            max_resolution=max_resolution,
        )
        budget_side, budget_area = self._ocr_window_budget(
            engine,
            max_side_px=max_side_px,
            max_area_px=max_area_px,
        )
        from natural_pdf.ocr import resolve_ocr_min_confidence

        resolved_min_confidence = resolve_ocr_min_confidence(
            target_obj,
            min_confidence,
            scope="region",
        )
        windows = self._plan_ocr_windows(
            verticals,
            horizontals,
            window=window,
            resolution=resolved_resolution,
            max_side_px=budget_side,
            max_area_px=budget_area,
            vertical_ratio=vertical_ratio,
            mixed_cell_threshold=mixed_cell_threshold,
            large_cell_ratio=large_cell_ratio,
        )
        self._annotate_ocr_windows(windows, resolution=resolved_resolution)

        result = GuidesOcrResult(
            guides=self,
            resolution=resolved_resolution,
            window=window,
            windows=windows,
            max_side_px=budget_side,
            max_area_px=budget_area,
            target_cell_px=target_cell_px,
            representative_percentile=representative_percentile,
            min_confidence=resolved_min_confidence,
            ran=not dry_run,
            target=target_obj,
        )
        self._last_ocr_result = result

        if dry_run:
            self._last_ocr_plan = result.to_dict()
            return result

        clear_target = (
            target_obj
            if hasattr(target_obj, "remove_ocr_elements")
            else resolve_page_for_materialization(target_obj)
        )
        if replace is True or replace == "all":
            clearer = getattr(clear_target, "clear_text_layer", None)
            if callable(clearer):
                clearer()
        elif replace == "ocr":
            remover = getattr(clear_target, "remove_ocr_elements", None)
            if callable(remover):
                remover()
        elif replace is False or replace is None:
            pass
        else:
            raise ValueError("replace must be True, False, 'all', or 'ocr'.")

        iterator: Iterable[Dict[str, Any]] = windows
        if show_progress and len(windows) > 1:
            from tqdm.auto import tqdm

            iterator = tqdm(windows, desc="Applying guide-window OCR", unit="window")

        for planned in iterator:
            region = self._create_ocr_window_region(target_obj, planned["bbox"])
            before_count = self._count_ocr_elements(target_obj)
            region.apply_ocr(
                engine=engine,
                options=options,
                languages=languages,
                min_confidence=resolved_min_confidence,
                device=device,
                resolution=resolved_resolution,
                detect_only=detect_only,
                apply_exclusions=apply_exclusions,
                replace=False,
                **kwargs,
            )
            after_count = self._count_ocr_elements(target_obj)
            if before_count is not None and after_count is not None:
                created_count: Optional[int] = max(0, after_count - before_count)
            else:
                created_count = None
            planned["created"] = created_count
            result.counts.append(created_count)

        self._ocr_applied = True
        self._ocr_prefer_words = bool(prefer_words)
        self._last_ocr_plan = result.to_dict()
        return result

    # -------------------------------------------------------------------------
    # Factory Methods
    # -------------------------------------------------------------------------

    @classmethod
    def divide(
        cls,
        obj: Union["Page", "Region", Tuple[float, float, float, float]],
        n: Optional[int] = None,
        cols: Optional[int] = None,
        rows: Optional[int] = None,
        axis: Literal["vertical", "horizontal", "both"] = "both",
    ) -> "Guides":
        """
        Create guides by evenly dividing an object.

        Args:
            obj: Object to divide (Page, Region, or bbox tuple)
            n: Number of divisions (creates n+1 guides). Used if cols/rows not specified.
            cols: Number of columns (creates cols+1 vertical guides)
            rows: Number of rows (creates rows+1 horizontal guides)
            axis: Which axis to divide along

        Returns:
            New Guides object with evenly spaced lines

        Examples:
            # Divide into 3 columns
            guides = Guides.divide(page, cols=3)

            # Divide into 5 rows
            guides = Guides.divide(region, rows=5)

            # Divide both axes
            guides = Guides.divide(page, cols=3, rows=5)
        """
        # Extract bounds from object
        if isinstance(obj, tuple) and len(obj) == 4:
            bounds = _ensure_bounds_tuple(obj)
            context = None
        else:
            context = obj
            bounds = _require_bounds(obj, context="object to divide")

        x0, y0, x1, y1 = bounds
        verticals = []
        horizontals = []

        # Handle vertical guides
        if axis in ("vertical", "both"):
            n_vertical = cols + 1 if cols is not None else (n + 1 if n is not None else 0)
            if n_vertical > 0:
                for i in range(n_vertical):
                    x = x0 + (x1 - x0) * i / (n_vertical - 1)
                    verticals.append(float(x))

        # Handle horizontal guides
        if axis in ("horizontal", "both"):
            n_horizontal = rows + 1 if rows is not None else (n + 1 if n is not None else 0)
            if n_horizontal > 0:
                for i in range(n_horizontal):
                    y = y0 + (y1 - y0) * i / (n_horizontal - 1)
                    horizontals.append(float(y))

        return cls(verticals=verticals, horizontals=horizontals, context=context, bounds=bounds)

    @classmethod
    def from_lines(
        cls,
        obj: GuidesContext,
        axis: Literal["vertical", "horizontal", "both"] = "both",
        threshold: Union[float, str] = "auto",
        source_label: Optional[str] = None,
        max_lines_h: Optional[int] = None,
        max_lines_v: Optional[int] = None,
        outer: bool = False,
        detection_method: str = "auto",
        resolution: int = 192,
        **detect_kwargs,
    ) -> "Guides":
        """
        Create guides from detected line elements.

        Args:
            obj: Page, Region, or FlowRegion to detect lines from
            axis: Which orientations to detect
            threshold: Detection threshold ('auto' or float 0.0-1.0) - used for pixel detection
            source_label: Filter for line source (vector method) or label for detected lines (pixel method)
            max_lines_h: Maximum number of horizontal lines to keep
            max_lines_v: Maximum number of vertical lines to keep
            outer: Whether to add outer boundary guides
            detection_method: 'auto' (default), 'vector', or 'pixels'. 'auto'
                uses vector line information when line elements exist and falls
                back to pixel detection otherwise.
            resolution: DPI for pixel-based detection (default: 192)
            **detect_kwargs: Additional parameters for pixel-based detection:
                - min_gap_h: Minimum gap between horizontal lines (pixels)
                - min_gap_v: Minimum gap between vertical lines (pixels)
                - binarization_method: 'adaptive' or 'otsu'
                - morph_op_h/v: Morphological operations ('open', 'close', 'none')
                - smoothing_sigma_h/v: Gaussian smoothing sigma
                - method: 'projection' (default) or 'lsd' (requires opencv)

        Returns:
            New Guides object with detected line positions
        """
        bounds = _bounds_from_object(obj)
        guides = cls(context=obj, bounds=bounds)

        if axis == "both":
            vertical_result, horizontal_result = generate_both_axis_coordinates(
                method="lines",
                context=obj,
                options=build_line_options(
                    "both",
                    threshold=threshold,
                    source_label=source_label,
                    max_lines_h=max_lines_h,
                    max_lines_v=max_lines_v,
                    outer=outer,
                    detection_method=detection_method,
                    resolution=resolution,
                    detect_kwargs=detect_kwargs,
                ),
            )
            apply_generation_result(guides, vertical_result, append=False)
            apply_generation_result(guides, horizontal_result, append=False)
            return guides

        result = generate_axis_coordinates(
            axis=axis,
            method="lines",
            context=obj,
            options=build_line_options(
                axis,
                threshold=threshold,
                source_label=source_label,
                max_lines_h=max_lines_h,
                max_lines_v=max_lines_v,
                outer=outer,
                detection_method=detection_method,
                resolution=resolution,
                detect_kwargs=detect_kwargs,
            ),
        )
        apply_generation_result(guides, result, append=False)
        return guides

    @classmethod
    def from_content(
        cls,
        obj: GuidesContext,
        axis: Literal["vertical", "horizontal"] = "vertical",
        markers: Union[str, List[str], "ElementCollection", None] = None,
        align: Union[
            Literal["left", "right", "center", "between"], Literal["top", "bottom"]
        ] = "left",
        outer: OuterBoundaryMode = True,
        tolerance: float = 5,
        apply_exclusions: bool = True,
    ) -> "Guides":
        """
        Create guides based on text content positions.

        Args:
            obj: Page, Region, or FlowRegion to search for content
            axis: Whether to create vertical or horizontal guides
            markers: Content to search for. Can be:
                - str: single selector (e.g., 'text:contains("Name")') or literal text
                - List[str]: list of selectors or literal text strings
                - ElementCollection: collection of elements to extract text from
                - None: no markers
            align: Where to place guides relative to found text:
                - For vertical guides: 'left', 'right', 'center', 'between'
                - For horizontal guides: 'top', 'bottom', 'center', 'between'
            outer: Whether to add guides at the boundaries
            tolerance: Maximum distance to search for text
            apply_exclusions: Whether to apply exclusion zones when searching for text

        Returns:
            New Guides object aligned to text content
        """
        guides = cls(context=obj, bounds=_bounds_from_object(obj))
        result = generate_axis_coordinates(
            axis=axis,
            method="content",
            context=obj,
            options=build_content_options(
                axis,
                markers,
                align,
                outer,
                tolerance,
                apply_exclusions,
            ),
        )
        apply_generation_result(guides, result, append=False)
        return guides

    @classmethod
    def from_whitespace(
        cls,
        obj: GuidesContext,
        axis: Literal["vertical", "horizontal", "both"] = "both",
        min_gap: float = 10,
    ) -> "Guides":
        """Create guides by detecting whitespace gaps (divide + snap placeholder)."""
        guides = cls(context=obj, bounds=_bounds_from_object(obj))
        if axis == "both":
            vertical_result = generate_axis_coordinates(
                axis="vertical",
                method="whitespace",
                context=obj,
                options=build_whitespace_options(min_gap),
            )
            horizontal_result = generate_axis_coordinates(
                axis="horizontal",
                method="whitespace",
                context=obj,
                options=build_whitespace_options(min_gap),
            )
            apply_generation_result(guides, vertical_result, append=False)
            apply_generation_result(guides, horizontal_result, append=False)
            return guides

        result = generate_axis_coordinates(
            axis=axis,
            method="whitespace",
            context=obj,
            options=build_whitespace_options(min_gap),
        )
        apply_generation_result(guides, result, append=False)
        return guides

    @classmethod
    def from_headers(
        cls,
        obj: GuidesContext,
        axis: Literal["vertical", "horizontal"] = "vertical",
        headers: Union["ElementCollection", Sequence[Any], None] = None,
        method: Literal["min_crossings", "seam_carving"] = "min_crossings",
        min_width: Optional[float] = None,
        max_width: Optional[float] = None,
        margin: float = 0.5,
        row_stabilization: bool = True,
        num_samples: int = 400,
    ) -> "Guides":
        """Create vertical guides by analyzing header elements."""

        if axis != "vertical":
            raise ValueError("from_headers() only works for vertical guides (columns)")

        guides = cls(context=obj, bounds=_bounds_from_object(obj))
        result = generate_axis_coordinates(
            axis="vertical",
            method="headers",
            context=obj,
            options=build_headers_options(
                headers,
                method=method,
                min_width=min_width,
                max_width=max_width,
                margin=margin,
                row_stabilization=row_stabilization,
                num_samples=num_samples,
            ),
        )
        apply_generation_result(guides, result, append=False)
        return guides

    @classmethod
    def from_stripes(
        cls,
        obj: GuidesContext,
        axis: Literal["vertical", "horizontal"] = "horizontal",
        stripes: Optional[Union["ElementCollection", Sequence[Any]]] = None,
        color: Optional[str] = None,
    ) -> "Guides":
        """Create guides from zebra stripes or colored bands."""

        axis_lower = axis.lower()
        if axis_lower not in {"vertical", "horizontal"}:
            raise ValueError("axis must be 'vertical' or 'horizontal'")
        axis = cast(Literal["vertical", "horizontal"], axis_lower)

        guides = cls(context=obj, bounds=_bounds_from_object(obj))
        result = generate_axis_coordinates(
            axis=axis,
            method="stripes",
            context=obj,
            options=build_stripes_options(stripes, color=color),
        )
        apply_generation_result(guides, result, append=True)
        return guides

    @classmethod
    def new(cls, context: Optional[Union["Page", "Region"]] = None) -> "Guides":
        """
        Create a new empty Guides object, optionally with a context.

        This provides a clean way to start building guides through chaining:
        guides = Guides.new(page).add_content(axis='vertical', markers=[...])

        Args:
            context: Optional Page or Region to use as default context for operations

        Returns:
            New empty Guides object
        """
        return cls(verticals=[], horizontals=[], context=context)

    # -------------------------------------------------------------------------
    # Manipulation Methods
    # -------------------------------------------------------------------------

    def snap_to_whitespace(
        self,
        axis: str = "vertical",
        min_gap: float = 10.0,
        detection_method: str = "pixels",  # 'pixels' or 'text'
        threshold: Union[
            float, str
        ] = "auto",  # threshold for what counts as a trough (0.0-1.0) or 'auto'
        on_no_snap: str = "warn",
    ) -> "Guides":
        """
        Snap guides to nearby whitespace gaps (troughs) using optimal assignment.
        Modifies this Guides object in place.

        Args:
            axis: Direction to snap ('vertical' or 'horizontal')
            min_gap: Minimum gap size to consider as a valid trough
            detection_method: Method for detecting troughs:
                            'pixels' - use pixel-based density analysis (default)
                            'text' - use text element spacing analysis
            threshold: Threshold for what counts as a trough:
                      - float (0.0-1.0): areas with this fraction or less of max density count as troughs
                      - 'auto': automatically find threshold that creates enough troughs for guides
                      (only applies when detection_method='pixels')
            on_no_snap: Action when snapping fails ('warn', 'ignore', 'raise')

        Returns:
            Self for method chaining.
        """
        if not self.context:
            logger.warning("No context available for whitespace detection")
            return self

        detection_mode = detection_method.lower()
        if detection_mode not in {"pixels", "text"}:
            raise ValueError("detection_method must be 'pixels' or 'text'")

        text_elements = collect_text_elements(self.context)

        def _compute_gaps(axis: str, guide_positions: Sequence[float]) -> List[Tuple[float, float]]:
            if detection_mode == "pixels":
                if axis == "vertical":
                    return find_vertical_whitespace_gaps(
                        self.bounds,
                        text_elements,
                        min_gap,
                        threshold,
                        guide_positions=guide_positions,
                    )
                return find_horizontal_whitespace_gaps(
                    self.bounds,
                    text_elements,
                    min_gap,
                    threshold,
                    guide_positions=guide_positions,
                )
            if axis == "vertical":
                return find_vertical_element_gaps(self.bounds, text_elements, min_gap)
            return find_horizontal_element_gaps(self.bounds, text_elements, min_gap)

        # Handle FlowRegion case - collect all text elements across regions
        if self.is_flow_region:
            if not text_elements:
                logger.warning(
                    "No text elements found across flow regions for whitespace detection"
                )
                return self

            if axis == "vertical":
                gaps = _compute_gaps("vertical", self.vertical.data)
                all_guides = []
                guide_to_region_map = {}
                for coord, region in self._unified_vertical:
                    all_guides.append(coord)
                    guide_to_region_map.setdefault(coord, []).append(region)

                if gaps and all_guides:
                    original_guides = all_guides.copy()
                    self._snap_guides_to_gaps(all_guides, gaps, axis)

                    self._unified_vertical = []
                    for i, new_coord in enumerate(all_guides):
                        original_coord = original_guides[i]
                        regions = guide_to_region_map.get(original_coord, [])
                        for region in regions:
                            self._unified_vertical.append((new_coord, region))

                    for region in self._flow_guides:
                        region_verticals = [
                            coord for coord, r in self._unified_vertical if r == region
                        ]
                        self._flow_guides[region] = (
                            sorted(list(set(region_verticals))),
                            self._flow_guides[region][1],
                        )

                    self._vertical_cache = None

            elif axis == "horizontal":
                gaps = _compute_gaps("horizontal", self.horizontal.data)
                all_guides = []
                guide_to_region_map = {}
                for coord, region in self._unified_horizontal:
                    all_guides.append(coord)
                    guide_to_region_map.setdefault(coord, []).append(region)

                if gaps and all_guides:
                    original_guides = all_guides.copy()
                    self._snap_guides_to_gaps(all_guides, gaps, axis)

                    self._unified_horizontal = []
                    for i, new_coord in enumerate(all_guides):
                        original_coord = original_guides[i]
                        regions = guide_to_region_map.get(original_coord, [])
                        for region in regions:
                            self._unified_horizontal.append((new_coord, region))

                    for region in self._flow_guides:
                        region_horizontals = [
                            coord for coord, r in self._unified_horizontal if r == region
                        ]
                        self._flow_guides[region] = (
                            self._flow_guides[region][0],
                            sorted(list(set(region_horizontals))),
                        )

                    self._horizontal_cache = None

            else:
                raise ValueError("axis must be 'vertical' or 'horizontal'")

            return self

        if not text_elements:
            logger.warning("No text elements found for whitespace detection")
            return self

        if axis == "vertical":
            gaps = _compute_gaps("vertical", self.vertical.data)
            if gaps:
                self._snap_guides_to_gaps(self.vertical.data, gaps, axis)
        elif axis == "horizontal":
            gaps = _compute_gaps("horizontal", self.horizontal.data)
            if gaps:
                self._snap_guides_to_gaps(self.horizontal.data, gaps, axis)
        else:
            raise ValueError("axis must be 'vertical' or 'horizontal'")

        # Ensure all coordinates are Python floats (not numpy types)
        self.vertical.data[:] = [float(x) for x in self.vertical.data]
        self.horizontal.data[:] = [float(y) for y in self.horizontal.data]

        return self

    def shift(
        self, index: int, offset: float, axis: Literal["vertical", "horizontal"] = "vertical"
    ) -> "Guides":
        """
        Move a specific guide by a offset amount.

        Args:
            index: Index of the guide to move
            offset: Amount to move (positive = right/down)
            axis: Which guide list to modify

        Returns:
            Self for method chaining
        """
        if axis == "vertical":
            if 0 <= index < len(self.vertical):
                self.vertical[index] += offset
                self.vertical = sorted(self.vertical)
            else:
                logger.warning(f"Vertical guide index {index} out of range")
        else:
            if 0 <= index < len(self.horizontal):
                self.horizontal[index] += offset
                self.horizontal = sorted(self.horizontal)
            else:
                logger.warning(f"Horizontal guide index {index} out of range")

        return self

    def add_vertical(self, x: float) -> "Guides":
        """Add a vertical guide at the specified x-coordinate."""
        update_axis_coordinates(self, "vertical", [x], append=True)
        return self

    def add_horizontal(self, y: float) -> "Guides":
        """Add a horizontal guide at the specified y-coordinate."""
        update_axis_coordinates(self, "horizontal", [y], append=True)
        return self

    def remove_vertical(self, index: int) -> "Guides":
        """Remove a vertical guide by index."""
        if 0 <= index < len(self.vertical):
            self.vertical.pop(index)
        return self

    def remove_horizontal(self, index: int) -> "Guides":
        """Remove a horizontal guide by index."""
        if 0 <= index < len(self.horizontal):
            self.horizontal.pop(index)
        return self

    # -------------------------------------------------------------------------
    # Region extraction properties
    # -------------------------------------------------------------------------

    @property
    def columns(self):
        """Access columns by index like guides.columns[0]."""
        return _ColumnAccessor(self)

    @property
    def rows(self):
        """Access rows by index like guides.rows[0]."""
        return _RowAccessor(self)

    @property
    def cells(self):
        """Access cells by index like guides.cells[row][col] or guides.cells[row, col]."""
        return _CellAccessor(self)

    # -------------------------------------------------------------------------
    # Region extraction methods (alternative API)
    # -------------------------------------------------------------------------

    def column(self, index: int, obj: Optional[Union["Page", "Region"]] = None) -> "Region":
        """
        Get a column region from the guides.

        Args:
            index: Column index (0-based)
            obj: Page or Region to create the column on (uses self.context if None)

        Returns:
            Region representing the specified column

        Raises:
            IndexError: If column index is out of range
        """
        target = obj or self.context
        if target is None:
            raise ValueError("No context available for region creation")

        vertical_guides = list(self.vertical.data)
        if not vertical_guides or index < 0 or index >= len(vertical_guides) - 1:
            raise IndexError(
                f"Column index {index} out of range (have {len(vertical_guides)-1} columns)"
            )

        # Get bounds from context
        _, y0, _, y1 = self._get_context_bounds()

        # Get column boundaries
        x0 = vertical_guides[index]
        x1 = vertical_guides[index + 1]

        # Create region using absolute coordinates
        if hasattr(target, "create_region"):
            if isinstance(target, Region):
                return target.create_region(x0, y0, x1, y1, relative=False)
            return target.create_region(x0, y0, x1, y1)

        try:
            page = _resolve_single_page(target)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"Cannot create region on {type(target)}") from exc

        return page.create_region(x0, y0, x1, y1)

    def row(self, index: int, obj: Optional[Union["Page", "Region"]] = None) -> "Region":
        """
        Get a row region from the guides.

        Args:
            index: Row index (0-based)
            obj: Page or Region to create the row on (uses self.context if None)

        Returns:
            Region representing the specified row

        Raises:
            IndexError: If row index is out of range
        """
        target = obj or self.context
        if target is None:
            raise ValueError("No context available for region creation")

        horizontal_guides = list(self.horizontal.data)
        if not horizontal_guides or index < 0 or index >= len(horizontal_guides) - 1:
            raise IndexError(
                f"Row index {index} out of range (have {len(horizontal_guides)-1} rows)"
            )

        # Get bounds from context
        x0, _, x1, _ = self._get_context_bounds()

        # Get row boundaries
        y0 = horizontal_guides[index]
        y1 = horizontal_guides[index + 1]

        # Create region using absolute coordinates
        if hasattr(target, "create_region"):
            if isinstance(target, Region):
                return target.create_region(x0, y0, x1, y1, relative=False)
            return target.create_region(x0, y0, x1, y1)

        try:
            page = _resolve_single_page(target)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"Cannot create region on {type(target)}") from exc

        return page.create_region(x0, y0, x1, y1)

    def cell(self, row: int, col: int, obj: Optional[Union["Page", "Region"]] = None) -> "Region":
        """
        Get a cell region from the guides.

        Args:
            row: Row index (0-based)
            col: Column index (0-based)
            obj: Page or Region to create the cell on (uses self.context if None)

        Returns:
            Region representing the specified cell

        Raises:
            IndexError: If row or column index is out of range
        """
        target = obj or self.context
        if target is None:
            raise ValueError("No context available for region creation")

        vertical_guides = list(self.vertical.data)
        horizontal_guides = list(self.horizontal.data)
        if not vertical_guides or col < 0 or col >= len(vertical_guides) - 1:
            raise IndexError(
                f"Column index {col} out of range (have {len(vertical_guides)-1} columns)"
            )
        if not horizontal_guides or row < 0 or row >= len(horizontal_guides) - 1:
            raise IndexError(f"Row index {row} out of range (have {len(horizontal_guides)-1} rows)")

        # Get cell boundaries
        x0 = vertical_guides[col]
        x1 = vertical_guides[col + 1]
        y0 = horizontal_guides[row]
        y1 = horizontal_guides[row + 1]

        # Create region using absolute coordinates
        if hasattr(target, "create_region"):
            if isinstance(target, Region):
                return target.create_region(x0, y0, x1, y1, relative=False)
            return target.create_region(x0, y0, x1, y1)

        try:
            page = _resolve_single_page(target)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"Cannot create region on {type(target)}") from exc

        return page.create_region(x0, y0, x1, y1)

    def left_of(self, guide_index: int, obj: Optional[Union["Page", "Region"]] = None) -> "Region":
        """
        Get a region to the left of a vertical guide.

        Args:
            guide_index: Vertical guide index
            obj: Page or Region to create the region on (uses self.context if None)

        Returns:
            Region to the left of the specified guide
        """
        target = obj or self.context
        if target is None:
            raise ValueError("No context available for region creation")

        if not self.vertical or guide_index < 0 or guide_index >= len(self.vertical):
            raise IndexError(f"Guide index {guide_index} out of range")

        # Get bounds from context
        bounds = self._get_context_bounds()
        if not bounds:
            raise ValueError("Could not determine bounds")
        x0, y0, _, y1 = bounds

        # Create region from left edge to guide
        x1 = self.vertical[guide_index]

        if hasattr(target, "region"):
            return target.region(x0, y0, x1, y1)
        else:
            raise TypeError(f"Cannot create region on {type(target)}")

    def right_of(self, guide_index: int, obj: Optional[Union["Page", "Region"]] = None) -> "Region":
        """
        Get a region to the right of a vertical guide.

        Args:
            guide_index: Vertical guide index
            obj: Page or Region to create the region on (uses self.context if None)

        Returns:
            Region to the right of the specified guide
        """
        target = obj or self.context
        if target is None:
            raise ValueError("No context available for region creation")

        if not self.vertical or guide_index < 0 or guide_index >= len(self.vertical):
            raise IndexError(f"Guide index {guide_index} out of range")

        # Get bounds from context
        bounds = self._get_context_bounds()
        if not bounds:
            raise ValueError("Could not determine bounds")
        _, y0, x1, y1 = bounds

        # Create region from guide to right edge
        x0 = self.vertical[guide_index]

        if hasattr(target, "region"):
            return target.region(x0, y0, x1, y1)
        else:
            raise TypeError(f"Cannot create region on {type(target)}")

    def above(self, guide_index: int, obj: Optional[Union["Page", "Region"]] = None) -> "Region":
        """
        Get a region above a horizontal guide.

        Args:
            guide_index: Horizontal guide index
            obj: Page or Region to create the region on (uses self.context if None)

        Returns:
            Region above the specified guide
        """
        target = obj or self.context
        if target is None:
            raise ValueError("No context available for region creation")

        if not self.horizontal or guide_index < 0 or guide_index >= len(self.horizontal):
            raise IndexError(f"Guide index {guide_index} out of range")

        # Get bounds from context
        bounds = self._get_context_bounds()
        if not bounds:
            raise ValueError("Could not determine bounds")
        x0, y0, x1, _ = bounds

        # Create region from top edge to guide
        y1 = self.horizontal[guide_index]

        if hasattr(target, "region"):
            return target.region(x0, y0, x1, y1)
        else:
            raise TypeError(f"Cannot create region on {type(target)}")

    def below(self, guide_index: int, obj: Optional[Union["Page", "Region"]] = None) -> "Region":
        """
        Get a region below a horizontal guide.

        Args:
            guide_index: Horizontal guide index
            obj: Page or Region to create the region on (uses self.context if None)

        Returns:
            Region below the specified guide
        """
        target = obj or self.context
        if target is None:
            raise ValueError("No context available for region creation")

        if not self.horizontal or guide_index < 0 or guide_index >= len(self.horizontal):
            raise IndexError(f"Guide index {guide_index} out of range")

        # Get bounds from context
        bounds = self._get_context_bounds()
        if not bounds:
            raise ValueError("Could not determine bounds")
        x0, _, x1, y1 = bounds

        # Create region from guide to bottom edge
        y0 = self.horizontal[guide_index]

        if hasattr(target, "region"):
            return target.region(x0, y0, x1, y1)
        else:
            raise TypeError(f"Cannot create region on {type(target)}")

    def between_vertical(
        self, start_index: int, end_index: int, obj: Optional[Union["Page", "Region"]] = None
    ) -> "Region":
        """
        Get a region between two vertical guides.

        Args:
            start_index: Starting vertical guide index
            end_index: Ending vertical guide index
            obj: Page or Region to create the region on (uses self.context if None)

        Returns:
            Region between the specified guides
        """
        target = obj or self.context
        if target is None:
            raise ValueError("No context available for region creation")

        if not self.vertical:
            raise ValueError("No vertical guides available")
        if start_index < 0 or start_index >= len(self.vertical):
            raise IndexError(f"Start index {start_index} out of range")
        if end_index < 0 or end_index >= len(self.vertical):
            raise IndexError(f"End index {end_index} out of range")
        if start_index >= end_index:
            raise ValueError("Start index must be less than end index")

        # Get bounds from context
        bounds = self._get_context_bounds()
        if not bounds:
            raise ValueError("Could not determine bounds")
        _, y0, _, y1 = bounds

        # Get horizontal boundaries
        x0 = self.vertical[start_index]
        x1 = self.vertical[end_index]

        if hasattr(target, "region"):
            return target.region(x0, y0, x1, y1)
        else:
            raise TypeError(f"Cannot create region on {type(target)}")

    def between_horizontal(
        self, start_index: int, end_index: int, obj: Optional[Union["Page", "Region"]] = None
    ) -> "Region":
        """
        Get a region between two horizontal guides.

        Args:
            start_index: Starting horizontal guide index
            end_index: Ending horizontal guide index
            obj: Page or Region to create the region on (uses self.context if None)

        Returns:
            Region between the specified guides
        """
        target = obj or self.context
        if target is None:
            raise ValueError("No context available for region creation")

        if not self.horizontal:
            raise ValueError("No horizontal guides available")
        if start_index < 0 or start_index >= len(self.horizontal):
            raise IndexError(f"Start index {start_index} out of range")
        if end_index < 0 or end_index >= len(self.horizontal):
            raise IndexError(f"End index {end_index} out of range")
        if start_index >= end_index:
            raise ValueError("Start index must be less than end index")

        # Get bounds from context
        bounds = self._get_context_bounds()
        if not bounds:
            raise ValueError("Could not determine bounds")
        x0, _, x1, _ = bounds

        # Get vertical boundaries
        y0 = self.horizontal[start_index]
        y1 = self.horizontal[end_index]

        if hasattr(target, "region"):
            return target.region(x0, y0, x1, y1)
        else:
            raise TypeError(f"Cannot create region on {type(target)}")

    # -------------------------------------------------------------------------
    # Operations
    # -------------------------------------------------------------------------

    def __add__(self, other: "Guides") -> "Guides":
        """
        Combine two guide sets.

        Returns:
            New Guides object with combined coordinates
        """
        # Combine and deduplicate coordinates, ensuring Python floats
        combined_verticals = sorted([float(x) for x in set(self.vertical + other.vertical)])
        combined_horizontals = sorted([float(y) for y in set(self.horizontal + other.horizontal)])

        # Handle FlowRegion context merging
        new_context = self.context or other.context

        # If both are flow regions, we might need a more complex merge,
        # but for now, just picking one context is sufficient.

        # Create the new Guides object
        new_guides = Guides(
            verticals=combined_verticals,
            horizontals=combined_horizontals,
            context=new_context,
            bounds=self.bounds or other.bounds,
        )

        # If the new context is a FlowRegion, we need to rebuild the flow-related state
        if new_guides.is_flow_region:
            # Re-initialize flow guides from both sources
            # This is a simplification; a true merge would be more complex.
            # For now, we combine the flow_guides dictionaries.
            if hasattr(self, "_flow_guides"):
                new_guides._flow_guides.update(self._flow_guides)
            if hasattr(other, "_flow_guides"):
                new_guides._flow_guides.update(other._flow_guides)

            # Re-initialize unified views
            if hasattr(self, "_unified_vertical"):
                new_guides._unified_vertical.extend(self._unified_vertical)
            if hasattr(other, "_unified_vertical"):
                new_guides._unified_vertical.extend(other._unified_vertical)

            if hasattr(self, "_unified_horizontal"):
                new_guides._unified_horizontal.extend(self._unified_horizontal)
            if hasattr(other, "_unified_horizontal"):
                new_guides._unified_horizontal.extend(other._unified_horizontal)

            # Invalidate caches to force rebuild
            new_guides._vertical_cache = None
            new_guides._horizontal_cache = None

        return new_guides

    def show(self, on=None, **kwargs):
        """
        Display the guides overlaid on a page or region.

        Args:
            on: Page, Region, PIL Image, or string to display guides on.
                If None, uses self.context (the object guides were created from).
                If string 'page', uses the page from self.context.
            **kwargs: Additional arguments passed to render() if applicable.

        Returns:
            PIL Image with guides drawn on it.
        """
        # Handle FlowRegion case
        if self.is_flow_region and (on is None or on == self.context):
            if not self._flow_guides:
                raise ValueError("No guides to show for FlowRegion")

            # Get stacking parameters from kwargs or use defaults
            stack_direction = kwargs.get("stack_direction", "vertical")
            stack_gap = kwargs.get("stack_gap", 5)
            stack_background_color = kwargs.get("stack_background_color", (255, 255, 255))

            # First, render all constituent regions without guides to get base images
            base_images = []
            region_infos = []  # Store region info for guide coordinate mapping

            for region in list(self._flow_constituent_regions()):
                render_fn = getattr(region, "render", None)
                if render_fn is None:
                    raise AttributeError(f"Region {region} does not support rendering")
                img = render_fn(
                    resolution=kwargs.get("resolution", 150),
                    width=kwargs.get("width", None),
                    crop=True,
                )
                if img:
                    base_images.append(img)

                    scale_x = img.width / region.width
                    scale_y = img.height / region.height

                    region_infos.append(
                        {
                            "region": region,
                            "img_width": img.width,
                            "img_height": img.height,
                            "scale_x": scale_x,
                            "scale_y": scale_y,
                            "pdf_x0": region.x0,
                            "pdf_top": region.top,
                            "pdf_x1": region.x1,
                            "pdf_bottom": region.bottom,
                        }
                    )

            if not base_images:
                raise ValueError("Failed to render any images for FlowRegion")

            # Calculate final canvas size based on stacking direction
            if stack_direction == "vertical":
                final_width = max(img.width for img in base_images)
                final_height = (
                    sum(img.height for img in base_images) + (len(base_images) - 1) * stack_gap
                )
            else:  # horizontal
                final_width = (
                    sum(img.width for img in base_images) + (len(base_images) - 1) * stack_gap
                )
                final_height = max(img.height for img in base_images)

            # Create unified canvas
            canvas = Image.new("RGB", (final_width, final_height), stack_background_color)
            draw = ImageDraw.Draw(canvas)

            # Paste base images and track positions
            region_positions = []  # (region_info, paste_x, paste_y)

            if stack_direction == "vertical":
                current_y = 0
                for i, (img, info) in enumerate(zip(base_images, region_infos)):
                    paste_x = (final_width - img.width) // 2  # Center horizontally
                    canvas.paste(img, (paste_x, current_y))
                    region_positions.append((info, paste_x, current_y))
                    current_y += img.height + stack_gap
            else:  # horizontal
                current_x = 0
                for i, (img, info) in enumerate(zip(base_images, region_infos)):
                    paste_y = (final_height - img.height) // 2  # Center vertically
                    canvas.paste(img, (current_x, paste_y))
                    region_positions.append((info, current_x, paste_y))
                    current_x += img.width + stack_gap

            # Now draw guides on the unified canvas
            # Draw vertical guides (blue) - these extend through the full canvas height
            for v_coord in self.vertical:
                # Find which region(s) this guide intersects
                for info, paste_x, paste_y in region_positions:
                    if info["pdf_x0"] <= v_coord <= info["pdf_x1"]:
                        # This guide is within this region's x-bounds
                        # Convert PDF coordinate to pixel coordinate relative to the region
                        adjusted_x = v_coord - info["pdf_x0"]
                        pixel_x = adjusted_x * info["scale_x"] + paste_x

                        # Draw full-height line on canvas (not clipped to region)
                        if 0 <= pixel_x <= final_width:
                            x_pixel = int(pixel_x)
                            draw.line(
                                [(x_pixel, 0), (x_pixel, final_height - 1)],
                                fill=(0, 0, 255, 200),
                                width=2,
                            )
                        break  # Only draw once per guide

            # Draw horizontal guides (red) - these extend through the full canvas width
            for h_coord in self.horizontal:
                # Find which region(s) this guide intersects
                for info, paste_x, paste_y in region_positions:
                    if info["pdf_top"] <= h_coord <= info["pdf_bottom"]:
                        # This guide is within this region's y-bounds
                        # Convert PDF coordinate to pixel coordinate relative to the region
                        adjusted_y = h_coord - info["pdf_top"]
                        pixel_y = adjusted_y * info["scale_y"] + paste_y

                        # Draw full-width line on canvas (not clipped to region)
                        if 0 <= pixel_y <= final_height:
                            y_pixel = int(pixel_y)
                            draw.line(
                                [(0, y_pixel), (final_width - 1, y_pixel)],
                                fill=(255, 0, 0, 200),
                                width=2,
                            )
                        break  # Only draw once per guide

            return canvas

        # Original single-region logic follows...
        # Determine what to display guides on
        target = on if on is not None else self.context

        # Handle string shortcuts
        if isinstance(target, str):
            if target == "page":
                if self.context is None:
                    raise ValueError("Cannot resolve 'page' without a guides context")
                try:
                    target = _resolve_single_page(self.context)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        "Cannot resolve 'page' from the current guides context"
                    ) from exc
            else:
                raise ValueError(f"Unknown string target: {target}. Only 'page' is supported.")

        if target is None:
            raise ValueError("No target specified and no context available for guides display")

        # Prepare kwargs for image generation
        image_kwargs = {}

        # Extract only the parameters that the new render() method accepts
        if "resolution" in kwargs:
            image_kwargs["resolution"] = kwargs["resolution"]
        if "width" in kwargs:
            image_kwargs["width"] = kwargs["width"]
        if "crop" in kwargs:
            image_kwargs["crop"] = kwargs["crop"]

        # If target is a region-like object, crop to just that region
        try:
            _resolve_single_page(target)
            target_is_single_page = True
        except (TypeError, ValueError):
            target_is_single_page = False

        if hasattr(target, "bbox") and target_is_single_page:
            image_kwargs["crop"] = True

        # Get base image
        if hasattr(target, "render"):
            rendered = cast(Any, target).render(**image_kwargs)
            if rendered is None:
                raise ValueError("Failed to generate base image")
            img = cast(Image.Image, rendered)
        elif hasattr(target, "mode") and hasattr(target, "size"):
            # It's already a PIL Image
            img = cast(Image.Image, target)
        else:
            raise ValueError(f"Object {target} does not support render() and is not a PIL Image")

        # Create a copy to draw on
        img = cast(Image.Image, img.copy())
        draw = ImageDraw.Draw(img)

        # Determine scale factor for coordinate conversion
        if _has_size(target) and not (hasattr(target, "mode") and hasattr(target, "size")):
            # target is a PDF object (Page/Region) with PDF coordinates
            size_target = cast(_SupportsSize, target)
            scale_x = img.width / size_target.width
            scale_y = img.height / size_target.height

            # If we're showing guides on a region, we need to adjust coordinates
            # to be relative to the region's origin
            if hasattr(target, "bbox") and target_is_single_page:
                # This is a Region - adjust guide coordinates to be relative to region
                region_like = cast(Any, target)
                region_x0 = float(getattr(region_like, "x0", 0.0))
                region_top = float(getattr(region_like, "top", 0.0))
            else:
                # This is a Page - no adjustment needed
                region_x0, region_top = 0, 0
        else:
            # target is already an image, no scaling needed
            scale_x = 1.0
            scale_y = 1.0
            region_x0, region_top = 0, 0

        # Draw vertical guides (blue)
        for x_coord in self.vertical:
            # Adjust coordinate if we're showing on a region
            adjusted_x = x_coord - region_x0
            pixel_x = adjusted_x * scale_x
            # Ensure guides at the edge are still visible by clamping to valid range
            if 0 <= pixel_x <= img.width - 1:
                x_pixel = int(min(pixel_x, img.width - 1))
                draw.line([(x_pixel, 0), (x_pixel, img.height - 1)], fill=(0, 0, 255, 200), width=2)

        # Draw horizontal guides (red)
        for y_coord in self.horizontal:
            # Adjust coordinate if we're showing on a region
            adjusted_y = y_coord - region_top
            pixel_y = adjusted_y * scale_y
            # Ensure guides at the edge are still visible by clamping to valid range
            if 0 <= pixel_y <= img.height - 1:
                y_pixel = int(min(pixel_y, img.height - 1))
                draw.line([(0, y_pixel), (img.width - 1, y_pixel)], fill=(255, 0, 0, 200), width=2)

        return img

    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------

    def get_cells(self) -> List[Tuple[float, float, float, float]]:
        """
        Get all cell bounding boxes from guide intersections.

        Returns:
            List of (x0, y0, x1, y1) tuples for each cell
        """
        cells = []

        # Create cells from guide intersections
        for i in range(len(self.vertical) - 1):
            for j in range(len(self.horizontal) - 1):
                x0 = self.vertical[i]
                x1 = self.vertical[i + 1]
                y0 = self.horizontal[j]
                y1 = self.horizontal[j + 1]
                cells.append((x0, y0, x1, y1))

        return cells

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary format suitable for pdfplumber table_settings.

        Returns:
            Dictionary with explicit_vertical_lines and explicit_horizontal_lines
        """
        return {
            "explicit_vertical_lines": self.vertical,
            "explicit_horizontal_lines": self.horizontal,
        }

    def to_relative(self) -> "Guides":
        """
        Convert absolute coordinates to relative (0-1) coordinates.

        Returns:
            New Guides object with relative coordinates
        """
        if self.relative:
            return self  # Already relative

        if not self.bounds:
            raise ValueError("Cannot convert to relative without bounds")

        x0, y0, x1, y1 = self.bounds
        width = x1 - x0
        height = y1 - y0

        rel_verticals = [(x - x0) / width for x in self.vertical]
        rel_horizontals = [(y - y0) / height for y in self.horizontal]

        return Guides(
            verticals=rel_verticals,
            horizontals=rel_horizontals,
            context=self.context,
            bounds=(0, 0, 1, 1),
            relative=True,
        )

    def to_absolute(self, bounds: Tuple[float, float, float, float]) -> "Guides":
        """
        Convert relative coordinates to absolute coordinates.

        Args:
            bounds: Target bounding box (x0, y0, x1, y1)

        Returns:
            New Guides object with absolute coordinates
        """
        if not self.relative:
            return self  # Already absolute

        x0, y0, x1, y1 = bounds
        width = x1 - x0
        height = y1 - y0

        abs_verticals = [x0 + x * width for x in self.vertical]
        abs_horizontals = [y0 + y * height for y in self.horizontal]

        return Guides(
            verticals=abs_verticals,
            horizontals=abs_horizontals,
            context=self.context,
            bounds=bounds,
            relative=False,
        )

    @property
    def n_rows(self) -> int:
        """Number of rows defined by horizontal guides."""
        return max(0, len(self.horizontal) - 1)

    @property
    def n_cols(self) -> int:
        """Number of columns defined by vertical guides."""
        return max(0, len(self.vertical) - 1)

    def _handle_snap_failure(self, message: str):
        """Handle cases where snapping cannot be performed."""
        behavior = getattr(self, "snap_behavior", getattr(self, "on_no_snap", "warn"))
        if behavior == "warn":
            logger.warning(message)
        elif behavior == "raise":
            raise ValueError(message)
        # 'ignore' case: do nothing

    def _optimal_guide_assignment(
        self, guides: List[float], trough_ranges: List[Tuple[float, float]]
    ) -> Dict[int, int]:
        """
        Assign guides to trough ranges and snap them to trough centers.
        All internal guides (not boundary guides) get assigned to the closest trough
        and moved to its center.
        """
        if not guides or not trough_ranges:
            return {}

        assignments = {}

        # Identify boundary guides (first and last) - these don't get reassigned
        boundary_indices = {0, len(guides) - 1} if len(guides) >= 2 else set()

        # Find internal guides that should be snapped to troughs
        internal_guides = []
        for i, guide_pos in enumerate(guides):
            if i not in boundary_indices:
                internal_guides.append(i)
                logger.debug(
                    f"Guide {i} (pos {guide_pos:.1f}) is internal, will be assigned to trough"
                )

        # Assign each internal guide to closest available trough
        if internal_guides and trough_ranges:
            # Calculate distances for all combinations
            distances = []
            for guide_idx in internal_guides:
                guide_pos = guides[guide_idx]
                for trough_idx, (trough_start, trough_end) in enumerate(trough_ranges):
                    trough_center = (trough_start + trough_end) / 2
                    distance = abs(guide_pos - trough_center)
                    distances.append((distance, guide_idx, trough_idx))

            # Sort by distance and assign greedily
            distances.sort()
            used_troughs = set()

            for distance, guide_idx, trough_idx in distances:
                if guide_idx not in assignments and trough_idx not in used_troughs:
                    assignments[guide_idx] = trough_idx
                    used_troughs.add(trough_idx)
                    logger.debug(
                        f"Assigned guide {guide_idx} (pos {guides[guide_idx]:.1f}) to trough {trough_idx} (distance: {distance:.1f})"
                    )

        logger.debug(f"Final assignments: {assignments}")
        return assignments

    def _snap_guides_to_gaps(self, guides: List[float], gaps: List[Tuple[float, float]], axis: str):
        """
        Snap guides to nearby gaps using optimal assignment.
        Only moves guides that are NOT already in a trough.
        """
        if not guides or not gaps:
            return

        logger.debug(f"Snapping {len(guides)} {axis} guides to {len(gaps)} trough ranges")
        for i, (start, end) in enumerate(gaps):
            center = (start + end) / 2
            logger.debug(f"  Trough {i}: {start:.1f} to {end:.1f} (center: {center:.1f})")

        # Get optimal assignments
        assignments = self._optimal_guide_assignment(guides, gaps)

        # Apply assignments (modify guides list in-place)
        for guide_idx, trough_idx in assignments.items():
            trough_start, trough_end = gaps[trough_idx]
            new_pos = (trough_start + trough_end) / 2  # Move to trough center
            old_pos = guides[guide_idx]
            guides[guide_idx] = new_pos
            logger.info(f"Snapped {axis} guide from {old_pos:.1f} to {new_pos:.1f}")

    def build_grid(
        self,
        target: Optional[GuidesContext] = None,
        source: str = "guides",
        cell_padding: float = 0.5,
        include_outer_boundaries: bool = False,
        *,
        multi_page: Literal["auto", True, False] = "auto",
    ) -> Dict[str, Any]:
        """
        Create table structure (table, rows, columns, cells) from guide coordinates.

        Args:
            target: Page or Region to create regions on (uses self.context if None)
            source: Source label for created regions (for identification)
            cell_padding: Internal padding for cell regions in points
            include_outer_boundaries: Whether to add boundaries at edges if missing
            multi_page: Controls multi-region table creation for FlowRegions.
                - "auto": (default) Creates a unified grid if there are multiple regions or guides span pages.
                - True: Forces creation of a unified multi-region grid.
                - False: Creates separate grids for each region.

        Returns:
            Dictionary with 'counts' and 'regions' created.
        """
        if self.is_flow_region:
            has_multiple_regions = len(self._flow_constituent_regions()) > 1
            spans_pages = self._spans_pages()

            if multi_page is True or (
                multi_page == "auto" and (spans_pages or has_multiple_regions)
            ):
                return self._build_grid_multi_page(
                    source=source,
                    cell_padding=cell_padding,
                    include_outer_boundaries=include_outer_boundaries,
                )
            else:
                total_counts = GridBuildCounts()
                all_regions = {"table": [], "rows": [], "columns": [], "cells": []}

                for region in self._flow_constituent_regions():
                    if region in self._flow_guides:
                        verticals, horizontals = self._flow_guides[region]
                        result = build_single_page_grid(
                            target_obj=region,
                            verticals=verticals,
                            horizontals=horizontals,
                            source=source,
                            cell_padding=cell_padding,
                            include_outer_boundaries=include_outer_boundaries,
                        )

                        total_counts.add(result.counts)

                        if result.regions.table:
                            all_regions["table"].append(result.regions.table)
                        all_regions["rows"].extend(result.regions.rows)
                        all_regions["columns"].extend(result.regions.columns)
                        all_regions["cells"].extend(result.regions.cells)

                logger.info(
                    f"Created {total_counts.table} tables, {total_counts.rows} rows, "
                    f"{total_counts.columns} columns, and {total_counts.cells} cells "
                    f"from guides across {len(self._flow_guides)} regions"
                )

                return {"counts": total_counts.as_dict(), "regions": all_regions}

        return self._build_grid_single_page(
            target=target,
            source=source,
            cell_padding=cell_padding,
            include_outer_boundaries=include_outer_boundaries,
        )

    def _build_grid_multi_page(
        self,
        source: str,
        cell_padding: float,
        include_outer_boundaries: bool,
    ) -> Dict[str, Any]:
        """
        Builds a single, coherent grid across multiple regions of a FlowRegion.

        Creates physical Region objects for each constituent region with _fragment
        region types (e.g., table_column_fragment), then stitches them into logical
        FlowRegion objects. Both are registered with pages, but the fragment types
        allow easy differentiation:
        - find_all('table_column') returns only logical columns
        - find_all('table_column_fragment') returns only physical fragments
        """
        from natural_pdf.flows.region import FlowRegion

        if not self.is_flow_region:
            raise ValueError("Multi-page grid building requires a FlowRegion with a valid Flow.")
        flow_context = self._flow_context()
        if not getattr(flow_context, "flow", None):
            raise ValueError("Multi-page grid building requires a FlowRegion with a valid Flow.")

        orientation = self._get_flow_orientation()
        adapter = FlowGuideAdapter(self)
        region_grids = adapter.build_region_grids(
            source=source,
            cell_padding=cell_padding,
            include_outer_boundaries=include_outer_boundaries,
        )

        if not region_grids:
            return {
                "counts": {"table": 0, "rows": 0, "columns": 0, "cells": 0},
                "regions": {"table": None, "rows": [], "columns": [], "cells": []},
            }

        flow_region = self._flow_context()
        flow = flow_region.flow

        physical_tables: List[Any] = [grid.table for grid in region_grids if grid.table is not None]
        flattened_tables = adapter._flatten_region_likes(physical_tables)
        multi_page_table = FlowRegion(
            flow=flow, constituent_regions=flattened_tables, source_flow_element=None
        )
        multi_page_table.source = source
        multi_page_table.region_type = "table"
        multi_page_table.metadata.update(
            {"is_multi_page": True, "num_rows": self.n_rows, "num_cols": self.n_cols}
        )

        final_rows, final_cols, final_cells = adapter.stitch_region_results(
            region_grids, orientation, source
        )

        constituent_pages = collect_constituent_pages(self._flow_constituent_regions())
        register_regions_with_pages(
            constituent_pages,
            multi_page_table,
            final_rows,
            final_cols,
            final_cells,
            log=logger,
        )

        final_counts = {
            "table": 1,
            "rows": len(final_rows),
            "columns": len(final_cols),
            "cells": len(final_cells),
        }
        final_regions = {
            "table": multi_page_table,
            "rows": final_rows,
            "columns": final_cols,
            "cells": final_cells,
        }

        logger.info(
            f"Created 1 multi-page table, {final_counts['rows']} logical rows, "
            f"{final_counts['columns']} logical columns from guides and registered with all constituent pages"
        )

        return {"counts": final_counts, "regions": final_regions}

    def _build_grid_single_page(
        self,
        target: Optional[GuidesContext] = None,
        source: str = "guides",
        cell_padding: float = 0.5,
        include_outer_boundaries: bool = False,
    ) -> Dict[str, Any]:
        """
        Private method to create table structure on a single page or region.
        (Refactored from the original public build_grid method).
        """
        target_obj = target or self.context
        if not target_obj:
            raise ValueError("No target object available. Provide target parameter or context.")
        result = build_single_page_grid(
            target_obj=target_obj,
            verticals=self.vertical,
            horizontals=self.horizontal,
            source=source,
            cell_padding=cell_padding,
            include_outer_boundaries=include_outer_boundaries,
        )
        return result.as_dict()

    def __repr__(self) -> str:
        """String representation of the guides."""
        return (
            f"Guides(verticals={len(self.vertical)}, "
            f"horizontals={len(self.horizontal)}, "
            f"cells={len(self.get_cells())})"
        )

    def _spans_pages(self) -> bool:
        """Check if any guides are defined across multiple pages in a FlowRegion."""
        if not self.is_flow_region:
            return False

        # Check vertical guides
        v_guide_pages = {}
        for coord, region in self._unified_vertical:
            v_guide_pages.setdefault(coord, set()).add(region.page.page_number)

        for pages in v_guide_pages.values():
            if len(pages) > 1:
                return True

        # Check horizontal guides
        h_guide_pages = {}
        for coord, region in self._unified_horizontal:
            h_guide_pages.setdefault(coord, set()).add(region.page.page_number)

        for pages in h_guide_pages.values():
            if len(pages) > 1:
                return True

        return False

    # -------------------------------------------------------------------------
    # Instance methods for fluent chaining (avoid name conflicts with class methods)
    # -------------------------------------------------------------------------

    def add_content(
        self,
        axis: Literal["vertical", "horizontal"] = "vertical",
        markers: Union[str, List[str], "ElementCollection", None] = None,
        obj: Optional[Union["Page", "Region"]] = None,
        align: Literal["left", "right", "center", "between"] = "left",
        outer: OuterBoundaryMode = True,
        tolerance: float = 5,
        apply_exclusions: bool = True,
    ) -> "Guides":
        """
        Instance method: Add guides from content, allowing chaining.
        This allows: Guides.new(page).add_content(axis='vertical', markers=[...])

        Args:
            axis: Which axis to create guides for
            markers: Content to search for. Can be:
                - str: single selector or literal text
                - List[str]: list of selectors or literal text strings
                - ElementCollection: collection of elements to extract text from
                - None: no markers
            obj: Page or Region to search (uses self.context if None)
            align: How to align guides relative to found elements
            outer: Whether to add outer boundary guides. Can be:
                - bool: True/False to add/not add both
                - "first": To add boundary before the first element
                - "last": To add boundary before the last element
            tolerance: Tolerance for snapping to element edges
            apply_exclusions: Whether to apply exclusion zones when searching for text

        Returns:
            Self for method chaining
        """
        # Use provided object or fall back to stored context
        target_obj = obj or self.context
        if target_obj is None:
            raise ValueError("No object provided and no context available")

        if axis == "vertical":
            self.vertical.from_content(
                markers=markers,
                obj=target_obj,
                align=align,
                outer=outer,
                tolerance=tolerance,
                append=True,
                apply_exclusions=apply_exclusions,
            )
        else:
            self.horizontal.from_content(
                markers=markers,
                obj=target_obj,
                align=align,
                outer=outer,
                tolerance=tolerance,
                append=True,
                apply_exclusions=apply_exclusions,
            )

        return self

    def add_lines(
        self,
        axis: Literal["vertical", "horizontal", "both"] = "both",
        obj: Optional[Union["Page", "Region"]] = None,
        threshold: Union[float, str] = "auto",
        source_label: Optional[str] = None,
        max_lines_h: Optional[int] = None,
        max_lines_v: Optional[int] = None,
        outer: bool = False,
        detection_method: str = "auto",
        resolution: int = 192,
        **detect_kwargs,
    ) -> "Guides":
        """
        Instance method: Add guides from lines, allowing chaining.
        This allows: Guides.new(page).add_lines(axis='horizontal')

        Args:
            axis: Which axis to detect lines for
            obj: Page or Region to search (uses self.context if None)
            threshold: Line detection threshold ('auto' or float 0.0-1.0)
            source_label: Filter lines by source label (vector) or label for detected lines (pixels)
            max_lines_h: Maximum horizontal lines to use
            max_lines_v: Maximum vertical lines to use
            outer: Whether to add outer boundary guides
            detection_method: 'auto' (default), 'vector', or 'pixels'. 'auto' uses vector line
                information when available and falls back to pixel detection otherwise.
            resolution: DPI for pixel-based detection (default: 192)
            **detect_kwargs: Additional parameters for pixel detection (see from_lines)

        Returns:
            Self for method chaining
        """
        # Use provided object or fall back to stored context
        target_obj = obj or self.context
        if target_obj is None:
            raise ValueError("No object provided and no context available")

        if axis == "both":
            vertical_result, horizontal_result = generate_both_axis_coordinates(
                method="lines",
                context=target_obj,
                options=build_line_options(
                    "both",
                    threshold=threshold,
                    source_label=source_label,
                    max_lines_h=max_lines_h,
                    max_lines_v=max_lines_v,
                    outer=outer,
                    detection_method=detection_method,
                    resolution=resolution,
                    detect_kwargs=detect_kwargs,
                ),
            )
            apply_generation_result(self, vertical_result, append=True)
            apply_generation_result(self, horizontal_result, append=True)
            return self

        if axis == "vertical":
            self.vertical.from_lines(
                obj=target_obj,
                threshold=threshold,
                source_label=source_label,
                max_lines=max_lines_v,
                outer=outer,
                detection_method=detection_method,
                resolution=resolution,
                append=True,
                **detect_kwargs,
            )
        if axis == "horizontal":
            self.horizontal.from_lines(
                obj=target_obj,
                threshold=threshold,
                source_label=source_label,
                max_lines=max_lines_h,
                outer=outer,
                detection_method=detection_method,
                resolution=resolution,
                append=True,
                **detect_kwargs,
            )

        return self

    def add_whitespace(
        self,
        axis: Literal["vertical", "horizontal", "both"] = "both",
        obj: Optional[Union["Page", "Region"]] = None,
        min_gap: float = 10,
    ) -> "Guides":
        """
        Instance method: Add guides from whitespace, allowing chaining.
        This allows: Guides.new(page).add_whitespace(axis='both')

        Args:
            axis: Which axis to create guides for
            obj: Page or Region to search (uses self.context if None)
            min_gap: Minimum gap size to consider

        Returns:
            Self for method chaining
        """
        # Use provided object or fall back to stored context
        target_obj = obj or self.context
        if target_obj is None:
            raise ValueError("No object provided and no context available")

        if axis in ("vertical", "both"):
            self.vertical.from_whitespace(obj=target_obj, min_gap=min_gap, append=True)
        if axis in ("horizontal", "both"):
            self.horizontal.from_whitespace(obj=target_obj, min_gap=min_gap, append=True)

        return self

    def extract_table(
        self,
        target: Optional[
            Union[
                "Page",
                "Region",
                "PageCollection",
                "ElementCollection",
                List[Union["Page", "Region"]],
            ]
        ] = None,
        source: str = "guides_temp",
        cell_padding: float = 0.5,
        include_outer_boundaries: bool = False,
        method: Optional[str] = None,
        table_settings: Optional[dict] = None,
        use_ocr: bool = False,
        ocr_config: Optional[dict] = None,
        text_options: Optional[Dict] = None,
        cell_extraction_func: Optional[Callable[["Region"], Optional[str]]] = None,
        cell_extract: Literal["text", "words"] = "text",
        cell_overlap: Literal["center", "full", "partial"] = "center",
        cell_newlines: Union[bool, str] = True,
        show_progress: bool = False,
        content_filter: Optional[Union[str, Callable[[str], bool], List[str]]] = None,
        apply_exclusions: bool = True,
        *,
        multi_page: Literal["auto", True, False] = "auto",
        header: Union[str, List[str], None] = "first",
        skip_repeating_headers: Optional[bool] = None,
        structure_engine: Optional[str] = None,
    ) -> "TableResult":
        """
        Extract table data directly from guides without leaving temporary regions.

        This method:
        1. Creates table structure using build_grid()
        2. Extracts table data from the created table region
        3. Cleans up all temporary regions
        4. Returns the TableResult

        When passed a collection (PageCollection, ElementCollection, or list), this method
        will extract tables from each element and combine them into a single result.

        Args:
            target: Page, Region, or collection of Pages/Regions to extract from (uses self.context if None)
            source: Source label for temporary regions (will be cleaned up)
            cell_padding: Internal padding for cell regions in points
            include_outer_boundaries: Whether to add boundaries at edges if missing
            method: Table extraction method ('tatr', 'pdfplumber', 'text', etc.)
            table_settings: Settings for pdfplumber table extraction
            use_ocr: Whether to use OCR for text extraction
            ocr_config: OCR configuration parameters
            text_options: Dictionary of options for the 'text' method
            cell_extraction_func: Optional callable for custom cell text extraction
            cell_extract: Cell text mode. "text" preserves current behavior; "words"
                extracts word elements and can be batched for guide-built cells.
            cell_overlap: Word overlap mode for cell_extract="words": "center",
                "full", or "partial".
            cell_newlines: Newline handling for extracted cell text.
            show_progress: Controls progress bar for text method
            content_filter: Content filtering function or patterns
            apply_exclusions: Whether to apply exclusion regions during text extraction (default: True)
            multi_page: Controls multi-region table creation for FlowRegions
            header: How to handle headers when extracting from collections:
                - "first": Use first row of first element as headers (default)
                - "all": Expect headers on each element, use from first element
                - None: No headers, use numeric indices
                - List[str]: Custom column names
            skip_repeating_headers: Whether to remove duplicate header rows when extracting from collections.
                Defaults to True when header is "first" or "all", False otherwise.
            structure_engine: Optional structure detection engine name passed to the underlying
                region extraction to leverage provider-backed table structure results.

        Returns:
            TableResult: Extracted table data

        Raises:
            ValueError: If no table region is created from the guides

        Example:
            ```python
            from natural_pdf.analyzers import Guides

            # Single page extraction
            guides = Guides.from_lines(page, source_label="detected")
            table_data = guides.extract_table()
            df = table_data.to_df()

            # Multiple page extraction
            guides = Guides(pages[0])
            guides.vertical.from_content(['Column 1', 'Column 2'])
            table_result = guides.extract_table(pages, header=['Col1', 'Col2'])
            df = table_result.to_df()

            # Region collection extraction
            regions = pdf.find_all('region[type=table]')
            guides = Guides(regions[0])
            guides.vertical.from_lines(n=3)
            table_result = guides.extract_table(regions)

            # Tiny text where character-level cell extraction collapses spacing
            table_result = guides.extract_table(
                include_outer_boundaries=True,
                cell_extract="words",
                cell_overlap="partial",
                cell_newlines=False,
            )
            ```
        """
        if self._ocr_prefer_words and cell_extraction_func is None and not use_ocr:
            if cell_extract == "text":
                cell_extract = "words"

        return extract_table_from_guides(
            self,
            target=target,
            source=source,
            cell_padding=cell_padding,
            include_outer_boundaries=include_outer_boundaries,
            method=method,
            table_settings=table_settings,
            use_ocr=use_ocr,
            ocr_config=ocr_config,
            text_options=text_options,
            cell_extraction_func=cell_extraction_func,
            cell_extract=cell_extract,
            cell_overlap=cell_overlap,
            cell_newlines=cell_newlines,
            show_progress=show_progress,
            content_filter=content_filter,
            apply_exclusions=apply_exclusions,
            multi_page=multi_page,
            header=header,
            skip_repeating_headers=skip_repeating_headers,
            structure_engine=structure_engine,
        )

    def _extract_table_from_collection(
        self,
        elements: Union["PageCollection", "ElementCollection", List[Union["Page", "Region"]]],
        header: Union[str, List[str], None] = "first",
        skip_repeating_headers: Optional[bool] = None,
        method: Optional[str] = None,
        table_settings: Optional[dict] = None,
        use_ocr: bool = False,
        ocr_config: Optional[dict] = None,
        text_options: Optional[Dict] = None,
        cell_extraction_func: Optional[Callable[["Region"], Optional[str]]] = None,
        cell_extract: Literal["text", "words"] = "text",
        cell_overlap: Literal["center", "full", "partial"] = "center",
        cell_newlines: Union[bool, str] = True,
        show_progress: bool = True,
        content_filter: Optional[Union[str, Callable[[str], bool], List[str]]] = None,
        apply_exclusions: bool = True,
        structure_engine: Optional[str] = None,
    ) -> "TableResult":
        """
        Extract tables from multiple pages or regions using this guide pattern.

        This method applies the guide to each element, extracts tables, and combines
        them into a single TableResult. Dynamic guides (using lambdas) are evaluated
        for each element.

        Args:
            elements: PageCollection, ElementCollection, or list of Pages/Regions to extract from
            header: How to handle headers:
                - "first": Use first row of first element as headers (default)
                - "all": Expect headers on each element, use from first element
                - None: No headers, use numeric indices
                - List[str]: Custom column names
            skip_repeating_headers: Whether to remove duplicate header rows.
                Defaults to True when header is "first" or "all", False otherwise.
            method: Table extraction method (passed to extract_table)
            table_settings: Settings for pdfplumber table extraction
            use_ocr: Whether to use OCR for text extraction
            ocr_config: OCR configuration parameters
            text_options: Dictionary of options for the 'text' method
            cell_extraction_func: Optional callable for custom cell text extraction
            cell_extract: Cell text mode. "text" preserves current behavior; "words"
                uses word-level extraction.
            cell_overlap: Word overlap mode for cell_extract="words".
            cell_newlines: Newline handling for extracted cell text.
            show_progress: Show progress bar for multi-element extraction (default: True)
            content_filter: Content filtering function or patterns
            apply_exclusions: Whether to apply exclusion regions during extraction
            structure_engine: Optional structure engine forwarded to each element extraction.

        Returns:
            TableResult: Combined table data from all elements

        Example:
            ```python
            # Create guide with static vertical, dynamic horizontal
            guide = Guides(regions[0])
            guide.vertical.from_content(columns, outer="last")
            guide.horizontal.from_content(lambda r: r.find_all('text:starts-with(NF-)'))

            # Extract from all regions
            table_result = guide._extract_table_from_collection(regions, header=columns)
            df = table_result.to_df()
            ```
        """
        from natural_pdf.core.page_collection import PageCollection
        from natural_pdf.tables.result import TableResult

        # Convert to list if it's a collection
        if isinstance(elements, (PageCollection, ElementCollection)):
            element_list = list(elements)
        else:
            element_list = elements

        if not element_list:
            return TableResult([])

        # Determine header handling
        if skip_repeating_headers is None:
            skip_repeating_headers = header in ["first", "all"] or isinstance(header, list)

        all_rows = []
        header_row = None

        # Configure progress bar
        iterator = element_list
        if show_progress and len(element_list) > 1:
            from tqdm.auto import tqdm

            iterator = tqdm(element_list, desc="Extracting tables from elements", unit="element")

        for i, element in enumerate(iterator):
            # Create a new Guides object for this element
            element_guide = Guides(element)

            # Copy vertical guides (usually static)
            if hasattr(self.vertical, "_callable") and self.vertical._callable is not None:
                # If vertical is dynamic (lambda), evaluate it
                element_guide.vertical.from_content(self.vertical._callable(element))
            else:
                # Copy static vertical positions
                element_guide.vertical.data = self.vertical.data.copy()

            # Handle horizontal guides
            if hasattr(self.horizontal, "_callable") and self.horizontal._callable is not None:
                # If horizontal is dynamic (lambda), evaluate it
                element_guide.horizontal.from_content(self.horizontal._callable(element))
            else:
                # Copy static horizontal positions
                element_guide.horizontal.data = self.horizontal.data.copy()

            # Extract table from this element
            table_result = element_guide.extract_table(
                method=method,
                table_settings=table_settings,
                use_ocr=use_ocr,
                ocr_config=ocr_config,
                text_options=text_options,
                cell_extraction_func=cell_extraction_func,
                cell_extract=cell_extract,
                cell_overlap=cell_overlap,
                cell_newlines=cell_newlines,
                show_progress=False,  # Don't show nested progress
                content_filter=content_filter,
                apply_exclusions=apply_exclusions,
                structure_engine=structure_engine,
            )

            # Convert to list of rows
            rows = list(table_result)

            # Handle headers based on strategy
            if i == 0:  # First element
                if header == "first" or header == "all":
                    # Use first row as header
                    if rows:
                        header_row = rows[0]
                        rows = rows[1:]  # Remove header from data
                elif isinstance(header, list):
                    # Custom headers provided
                    header_row = header
            else:  # Subsequent elements
                if header == "all" and skip_repeating_headers and rows:
                    # Expect and remove header row
                    if rows and header_row and rows[0] == header_row:
                        rows = rows[1:]
                    elif rows:
                        # Still remove first row if it looks like a header
                        rows = rows[1:]

            # Add rows to combined result
            all_rows.extend(rows)

        # Create final TableResult
        if isinstance(header, list):
            final_result = TableResult(all_rows)
            final_result.headers = header
        elif header_row is not None:
            final_result = TableResult([header_row] + all_rows)
            final_result.headers = header_row
        else:
            final_result = TableResult(all_rows)
        self._assign_headers_from_rows(final_result, header)
        return final_result

    @staticmethod
    def _assign_headers_from_rows(
        table_result: TableResult, header: Union[str, int, List[str], None]
    ) -> None:
        """Normalize headers on a TableResult and drop empty leading rows."""

        def _row_is_blank(row: Sequence[Any]) -> bool:
            return all(cell is None or (isinstance(cell, str) and not cell.strip()) for cell in row)

        rows = getattr(table_result, "_rows", None)
        if not isinstance(rows, list) or not rows or header in (None, False):
            return

        if isinstance(header, list):
            table_result.headers = header
            return

        # Remove leading empty rows that can appear due to outer boundaries
        while rows and _row_is_blank(rows[0]):
            rows.pop(0)

        if not rows:
            return

        header_row: Optional[List[Any]] = None
        if header == "first":
            header_row = list(rows[0])
        elif isinstance(header, int):
            idx = max(0, min(len(rows) - 1, header))
            header_row = list(rows[idx])

        if header_row:
            table_result.headers = header_row

    def _get_flow_orientation(self) -> Literal["vertical", "horizontal", "unknown"]:
        """Determines if a FlowRegion's constituent parts are arranged vertically or horizontally."""
        if not self.is_flow_region or len(self._flow_constituent_regions()) < 2:
            return "unknown"

        r1 = self._flow_constituent_regions()[0]
        r2 = self._flow_constituent_regions()[1]  # Compare first two regions

        if not r1.bbox or not r2.bbox:
            return "unknown"

        # Calculate non-overlapping distances.
        # This determines the primary direction of separation.
        x_dist = max(0, max(r1.x0, r2.x0) - min(r1.x1, r2.x1))
        y_dist = max(0, max(r1.top, r2.top) - min(r1.bottom, r2.bottom))

        if y_dist > x_dist:
            return "vertical"
        else:
            return "horizontal"


# -------------------------------------------------------------------------
# Accessor classes for property-based access
# -------------------------------------------------------------------------


class _ColumnAccessor:
    """Provides indexed access to columns via guides.columns[index]."""

    def __init__(self, guides: "Guides"):
        self._guides = guides

    def __len__(self):
        """Return number of columns (vertical guides - 1)."""
        return max(0, len(self._guides.vertical) - 1)

    def __getitem__(self, index: Union[int, slice]) -> Union["Region", "ElementCollection"]:
        """Get column at the specified index or slice."""

        if isinstance(index, slice):
            # Handle slice notation - return multiple columns
            columns = []
            num_cols = len(self)

            # Convert slice to range of indices
            start, stop, step = index.indices(num_cols)
            for i in range(start, stop, step):
                columns.append(self._guides.column(i))

            return ElementCollection(columns)
        else:
            # Handle negative indexing
            if index < 0:
                index = len(self) + index
            return self._guides.column(index)


class _RowAccessor:
    """Provides indexed access to rows via guides.rows[index]."""

    def __init__(self, guides: "Guides"):
        self._guides = guides

    def __len__(self):
        """Return number of rows (horizontal guides - 1)."""
        return max(0, len(self._guides.horizontal) - 1)

    def __getitem__(self, index: Union[int, slice]) -> Union["Region", "ElementCollection"]:
        """Get row at the specified index or slice."""

        if isinstance(index, slice):
            # Handle slice notation - return multiple rows
            rows = []
            num_rows = len(self)

            # Convert slice to range of indices
            start, stop, step = index.indices(num_rows)
            for i in range(start, stop, step):
                rows.append(self._guides.row(i))

            return ElementCollection(rows)
        else:
            # Handle negative indexing
            if index < 0:
                index = len(self) + index
            return self._guides.row(index)


class _CellAccessor:
    """Provides indexed access to cells via guides.cells[row][col] or guides.cells[row, col]."""

    def __init__(self, guides: "Guides"):
        self._guides = guides

    def __getitem__(self, key) -> Union["Region", "_CellRowAccessor", "ElementCollection"]:
        """
        Get cell(s) at the specified position.

        Supports:
        - guides.cells[row, col] - single cell
        - guides.cells[row][col] - single cell (nested)
        - guides.cells[row, :] - all cells in a row
        - guides.cells[:, col] - all cells in a column
        - guides.cells[:, :] - all cells
        - guides.cells[row][:] - all cells in a row (nested)
        """

        if isinstance(key, tuple) and len(key) == 2:
            row, col = key

            # Handle slices for row and/or column
            if isinstance(row, slice) or isinstance(col, slice):
                cells = []
                num_rows = len(self._guides.rows)
                num_cols = len(self._guides.columns)

                # Convert slices to ranges
                if isinstance(row, slice):
                    row_indices = range(*row.indices(num_rows))
                else:
                    # Single row index
                    if row < 0:
                        row = num_rows + row
                    row_indices = [row]

                if isinstance(col, slice):
                    col_indices = range(*col.indices(num_cols))
                else:
                    # Single column index
                    if col < 0:
                        col = num_cols + col
                    col_indices = [col]

                # Collect all cells in the specified ranges
                for r in row_indices:
                    for c in col_indices:
                        cells.append(self._guides.cell(r, c))

                return ElementCollection(cells)
            else:
                # Both are integers - single cell access
                # Handle negative indexing for both row and col
                if row < 0:
                    row = len(self._guides.rows) + row
                if col < 0:
                    col = len(self._guides.columns) + col
                return self._guides.cell(row, col)
        elif isinstance(key, slice):
            # First level slice: guides.cells[:] - return all rows as accessors
            # For now, let's return all cells flattened
            cells = []
            num_rows = len(self._guides.rows)
            row_indices = range(*key.indices(num_rows))

            for r in row_indices:
                for c in range(len(self._guides.columns)):
                    cells.append(self._guides.cell(r, c))

            return ElementCollection(cells)
        elif isinstance(key, int):
            # First level of nested access: guides.cells[row]
            # Handle negative indexing for row
            if key < 0:
                key = len(self._guides.rows) + key
            # Return a row accessor that allows [col] or [:] indexing
            return _CellRowAccessor(self._guides, key)
        else:
            raise TypeError(
                f"Cell indices must be integers, slices, or tuple of two integers/slices, got {type(key)}"
            )


class _CellRowAccessor:
    """Provides column access for a specific row in nested cell indexing."""

    def __init__(self, guides: "Guides", row: int):
        self._guides = guides
        self._row = row

    def __getitem__(self, col: Union[int, slice]) -> Union["Region", "ElementCollection"]:
        """Get cell at [row][col] or all cells in row with [row][:]."""

        if isinstance(col, slice):
            # Handle slice notation - return all cells in this row
            cells = []
            num_cols = len(self._guides.columns)

            # Convert slice to range of indices
            start, stop, step = col.indices(num_cols)
            for c in range(start, stop, step):
                cells.append(self._guides.cell(self._row, c))

            return ElementCollection(cells)
        else:
            # Handle single column index
            # Handle negative indexing for column
            if col < 0:
                col = len(self._guides.columns) + col
            return self._guides.cell(self._row, col)
