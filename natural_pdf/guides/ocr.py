"""Lazy guide-region views and their OCR planning/result contracts."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from dataclasses import replace as dataclass_replace
from types import MappingProxyType
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
    cast,
    overload,
)

from natural_pdf.core.ocr_contracts import (
    OCRDetectionRequest,
    OCRFunctionRequest,
    OCRRequest,
    normalize_ocr_request,
)
from natural_pdf.core.ocr_execution import ocr_execution_session
from natural_pdf.core.ocr_mixin import OCRScopeMixin
from natural_pdf.elements.region import Region
from natural_pdf.ocr import (
    normalize_ocr_options,
    resolve_ocr_device,
    resolve_ocr_engine_name,
    resolve_ocr_languages,
    resolve_ocr_min_confidence,
)
from natural_pdf.services.base import resolve_service
from natural_pdf.tables.result import TableResult

from ._targets import resolve_page_for_materialization
from .helpers import Bounds, GuidesContext, _bounds_from_object, _is_flow_region, _require_bounds

if TYPE_CHECKING:
    from .base import Guides


_OCR_WINDOW_DEBUG_COLORS: Tuple[Tuple[int, int, int, int], ...] = (
    (37, 99, 235, 34),
    (22, 163, 74, 34),
    (124, 58, 237, 34),
    (8, 145, 178, 34),
    (202, 138, 4, 34),
    (15, 118, 110, 34),
    (79, 70, 229, 34),
    (101, 163, 13, 34),
)


@dataclass(frozen=True, slots=True)
class GuideOCRPlanningOptions:
    """Advanced window-planning controls used only by ``GuideCells.plan_ocr``."""

    window: Union[str, Tuple[int, int], List[int]] = "auto"
    target_cell_px: int = 40
    representative_percentile: float = 25.0
    min_resolution: int = 150
    max_resolution: int = 400
    max_side_px: Optional[int] = None
    max_area_px: Optional[int] = None
    vertical_ratio: float = 2.0
    mixed_cell_threshold: float = 0.5
    large_cell_ratio: float = 2.0

    def __post_init__(self) -> None:
        if isinstance(self.window, list):
            object.__setattr__(self, "window", tuple(self.window))
        if isinstance(self.window, str):
            if self.window not in {"auto", "table", "cell"}:
                raise ValueError("window must be 'auto', 'table', 'cell', or a (cols, rows) pair")
        elif (
            not isinstance(self.window, (tuple, list))
            or len(self.window) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 1
                for value in self.window
            )
        ):
            raise ValueError(
                "window must be 'auto', 'table', 'cell', or a positive (cols, rows) pair"
            )
        if self.target_cell_px < 1:
            raise ValueError("target_cell_px must be positive")
        if not 0 <= self.representative_percentile <= 100:
            raise ValueError("representative_percentile must be between 0 and 100")
        if self.min_resolution < 1 or self.max_resolution < self.min_resolution:
            raise ValueError("resolution bounds must be positive and ordered")
        if self.max_side_px is not None and self.max_side_px < 1:
            raise ValueError("max_side_px must be positive or None")
        if self.max_area_px is not None and self.max_area_px < 1:
            raise ValueError("max_area_px must be positive or None")


def _freeze_snapshot_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze_snapshot_value(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_snapshot_value(item) for item in value)
    if isinstance(value, set):
        return frozenset(_freeze_snapshot_value(item) for item in value)
    return value


def _copy_snapshot_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _copy_snapshot_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_copy_snapshot_value(item) for item in value)
    if isinstance(value, frozenset):
        return set(_copy_snapshot_value(item) for item in value)
    return value


@dataclass(frozen=True, slots=True)
class GuideOCRPlan:
    """An exact geometry snapshot for one guide-view OCR operation."""

    guides: "Guides" = field(repr=False)
    view: "_GuideRegionView" = field(repr=False)
    request: OCRRequest = field(repr=False)
    resolution: int
    window: Union[str, Tuple[int, int]]
    windows: Tuple[Mapping[str, Any], ...]
    max_side_px: int
    max_area_px: int
    target_cell_px: int
    representative_percentile: float
    min_confidence: Optional[float]
    target: GuidesContext = field(repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "window", _freeze_snapshot_value(self.window))
        object.__setattr__(
            self,
            "windows",
            tuple(_freeze_snapshot_value(window) for window in self.windows),
        )

    def __len__(self) -> int:
        return len(self.windows)

    def summary(self) -> Dict[str, Any]:
        return {
            "resolution": self.resolution,
            "window": self.window,
            "window_count": len(self.windows),
            "max_side_px": self.max_side_px,
            "max_area_px": self.max_area_px,
            "target_cell_px": self.target_cell_px,
            "representative_percentile": self.representative_percentile,
            "min_confidence": self.min_confidence,
        }

    def to_dict(self) -> Dict[str, Any]:
        data = self.summary()
        data["windows"] = [_copy_snapshot_value(window) for window in self.windows]
        return data

    def apply(self) -> "GuideOCRResult":
        """Execute this exact snapshot and return the execution result."""

        return self.view._apply_ocr_plan(self)

    def extract_table(self, *args: Any, **kwargs: Any) -> TableResult:
        return self.guides.extract_table(*args, **kwargs)

    def ocr_elements(self) -> List[Any]:
        elements = _scoped_text_elements(self.target) or []
        sources = {"ocr"}
        if isinstance(self.request, OCRFunctionRequest):
            sources.add(self.request.source_label)
        return [element for element in elements if getattr(element, "source", None) in sources]

    @staticmethod
    def _center_in(bbox: Bounds, window_bbox: Bounds, tolerance: float = 0.25) -> bool:
        x0, top, x1, bottom = bbox
        wx0, wtop, wx1, wbottom = window_bbox
        center_x = (x0 + x1) / 2.0
        center_y = (top + bottom) / 2.0
        return bool(
            wx0 - tolerance <= center_x <= wx1 + tolerance
            and wtop - tolerance <= center_y <= wbottom + tolerance
        )

    def window_text_elements(self, *, tolerance: float = 0.25, **_: Any) -> List[List[Any]]:
        buckets: List[List[Any]] = [[] for _ in self.windows]
        for element in self.ocr_elements():
            bbox = _bounds_from_object(element)
            if bbox is None:
                continue
            for index, window in enumerate(self.windows):
                window_bbox = _bounds_from_object(window.get("bbox"))
                if window_bbox is not None and self._center_in(bbox, window_bbox, tolerance):
                    buckets[index].append(element)
                    break
        return buckets

    def show(
        self,
        *,
        window: Optional[Union[int, Sequence[int]]] = None,
        include_windows: bool = True,
        include_text: bool = True,
        window_color: Optional[Any] = None,
        window_colors: Optional[Sequence[Any]] = None,
        text_color: Any = "red",
        labels: bool = True,
        **kwargs: Any,
    ) -> Any:
        """Render the snapped OCR windows and currently assigned OCR text."""

        source = self.target
        if not hasattr(source, "show"):
            source = resolve_page_for_materialization(source)
        if window is None:
            indices = list(range(len(self.windows)))
        elif isinstance(window, int):
            indices = [window]
        else:
            indices = list(window)
        if any(index < 0 or index >= len(self.windows) for index in indices):
            raise IndexError(f"window index out of range for {len(self.windows)} windows")

        palette = window_colors or _OCR_WINDOW_DEBUG_COLORS
        buckets = self.window_text_elements()
        highlights: List[Dict[str, Any]] = []
        for index in indices:
            if include_windows:
                color = window_color or palette[index % len(palette)]
                highlights.append(
                    {
                        "bbox": self.windows[index]["bbox"],
                        "color": color,
                        "label": f"window {index + 1}",
                        "fill": True,
                        "line_width": 2.0,
                        "vertices": False,
                    }
                )
            if include_text:
                highlights.extend(
                    {
                        "element": element,
                        "color": text_color,
                        "label": f"window {index + 1} OCR",
                    }
                    for element in buckets[index]
                )
        return source.show(highlights=highlights, labels=labels, **kwargs)


@dataclass
class GuideOCRResult:
    """Outcome of applying a :class:`GuideOCRPlan`."""

    plan: GuideOCRPlan
    counts: List[Optional[int]] = field(default_factory=list)

    @property
    def guides(self) -> "Guides":
        return self.plan.guides

    @property
    def view(self) -> "_GuideRegionView":
        return self.plan.view

    @property
    def target(self) -> GuidesContext:
        return self.plan.target

    @property
    def resolution(self) -> int:
        return self.plan.resolution

    @property
    def window(self) -> Union[str, Tuple[int, int]]:
        return self.plan.window

    @property
    def min_confidence(self) -> Optional[float]:
        return self.plan.min_confidence

    @property
    def windows(self) -> Tuple[Mapping[str, Any], ...]:
        return self.plan.windows

    @property
    def ran(self) -> bool:
        return True

    def __len__(self) -> int:
        return len(self.plan)

    @property
    def total_created(self) -> Optional[int]:
        known_counts = [count for count in self.counts if count is not None]
        return int(sum(known_counts)) if known_counts else None

    def summary(self) -> Dict[str, Any]:
        data = self.plan.summary()
        data.update({"ran": True, "total_created": self.total_created})
        return data

    def to_dict(self) -> Dict[str, Any]:
        data = self.summary()
        data.update(
            {
                "windows": [_copy_snapshot_value(window) for window in self.windows],
                "counts": list(self.counts),
            }
        )
        return data

    def extract_table(self, *args: Any, **kwargs: Any) -> TableResult:
        return self.plan.extract_table(*args, **kwargs)

    def show(self, **kwargs: Any) -> Any:
        return self.plan.show(**kwargs)

    def ocr_elements(self) -> List[Any]:
        return self.plan.ocr_elements()

    def window_text_elements(self, **kwargs: Any) -> List[List[Any]]:
        return self.plan.window_text_elements(**kwargs)


@dataclass(frozen=True, slots=True)
class _GuideGeometrySnapshot:
    target: GuidesContext
    verticals: Tuple[float, ...]
    horizontals: Tuple[float, ...]
    bounds: Bounds


def _normalize_sequence_index(index: int, size: int, *, label: str) -> int:
    if not isinstance(index, int):
        raise TypeError(f"{label} index must be an integer")
    normalized = index + size if index < 0 else index
    if normalized < 0 or normalized >= size:
        raise IndexError(f"{label} index {index} out of range for {size} {label}s")
    return normalized


def _snapshot_guide_geometry(guides: "Guides") -> _GuideGeometrySnapshot:
    target = guides.context
    if target is None:
        raise ValueError("Guide views require Guides initialized with a Page or Region context.")
    if _is_flow_region(target):
        raise ValueError(
            "Guide OCR views do not support FlowRegion contexts; use a Page or Region."
        )
    from natural_pdf.core.page import Page

    if not isinstance(target, (Page, Region)):
        raise TypeError(
            f"Guide views require a Page or Region context, got {type(target).__name__}."
        )
    return _GuideGeometrySnapshot(
        target=target,
        verticals=tuple(sorted(float(value) for value in guides.vertical)),
        horizontals=tuple(sorted(float(value) for value in guides.horizontal)),
        bounds=_require_bounds(target, context="guide view context"),
    )


def _region_for_target(
    target: GuidesContext,
    bbox: Bounds,
    *,
    bake_callable_exclusions: bool = False,
) -> Region:
    x0, top, x1, bottom = bbox
    region = Region(
        resolve_page_for_materialization(target),
        (float(x0), float(top), float(x1), float(bottom)),
    )
    if isinstance(target, Region):
        # Windows carved out of a Region host must behave like that host:
        # region-local exclusions keep masking content and region-scope config
        # keeps resolving (page-level state already flows in via ``page``).
        # Callables resolve against the host either eagerly (OCR request path
        # with exclusions applied) or lazily (everything else).
        host_exclusions = getattr(target, "_exclusions", None)
        if host_exclusions:
            from natural_pdf.services.exclusion_service import (
                bind_exclusion_entries_to_host,
            )

            region._exclusions = bind_exclusion_entries_to_host(
                target, host_exclusions, eager=bake_callable_exclusions
            )
        host_config = target.metadata.get("config") if isinstance(target.metadata, dict) else None
        if isinstance(host_config, dict) and host_config:
            region.metadata["config"] = dict(host_config)
    return region


def _region_for_bbox(snapshot: _GuideGeometrySnapshot, bbox: Bounds) -> Region:
    return _region_for_target(snapshot.target, bbox)


def _execute_region_ocr_request(region: Region, request: OCRRequest) -> None:
    region._execute_ocr_request(request)


def _ocr_scope_for_target(target: GuidesContext) -> str:
    scope_getter = getattr(target, "_ocr_scope", None)
    if callable(scope_getter):
        scope = scope_getter()
        if isinstance(scope, str) and scope:
            return scope
    return "region"


def _configured_ocr_resolution(target: GuidesContext) -> Optional[int]:
    """Return the host-configured OCR resolution, if any (no built-in fallback)."""

    service = resolve_service(target, "ocr")
    option_value = service._context.get_option(  # noqa: SLF001 - exact service lookup
        "ocr",
        "resolution",
        host=target,
        default=None,
        scope=_ocr_scope_for_target(target),
    )
    if option_value is None:
        return None
    try:
        return int(option_value)
    except (TypeError, ValueError):
        return None


def _prepare_effective_request(
    target: GuidesContext,
    request: OCRRequest,
    *,
    resolution: Optional[int] = None,
) -> OCRRequest:
    """Freeze all configured host-scope defaults into one built-in request.

    Every default must resolve against the ORIGINAL host (Page or Region) so
    region-local configuration is honored; a fresh probe Region would lose the
    host's config chain.
    """

    if isinstance(request, OCRFunctionRequest):
        return request
    scope = _ocr_scope_for_target(target)
    normalized_options = normalize_ocr_options(request.options)
    requested_engine = request.engine
    if requested_engine is None and (request.model is not None or request.client is not None):
        requested_engine = "vlm"
    engine = resolve_ocr_engine_name(
        context=target,
        requested=requested_engine,
        options=normalized_options,
        scope=scope,
    )
    normalized_options = normalize_ocr_options(request.options, engine_name=engine)
    languages = resolve_ocr_languages(
        target,
        list(request.languages) if request.languages is not None else None,
        scope=scope,
    )
    min_confidence = resolve_ocr_min_confidence(target, request.min_confidence, scope=scope)
    device = resolve_ocr_device(target, request.device, scope=scope)
    service = resolve_service(target, "ocr")
    final_resolution = service._resolve_resolution(  # noqa: SLF001 - exact service default
        target,
        resolution if resolution is not None else request.resolution,
        scope,
    )
    return dataclass_replace(
        request,
        engine=engine,
        options=normalized_options,
        languages=tuple(languages) if languages is not None else None,
        min_confidence=min_confidence,
        device=device,
        resolution=final_resolution,
    )


def _scoped_text_elements(target: GuidesContext) -> Optional[List[Any]]:
    finder = getattr(target, "find_all", None)
    if callable(finder):
        try:
            return list(finder("text", apply_exclusions=False))
        except Exception:
            pass
    return None


def _is_detection_element(element: Any) -> bool:
    if bool(getattr(element, "is_ocr_detection", False)):
        return True
    metadata = getattr(element, "metadata", None)
    if isinstance(metadata, dict) and bool(metadata.get("ocr_detection_only", False)):
        return True
    obj = getattr(element, "_obj", None)
    return isinstance(obj, dict) and bool(obj.get("ocr_detection_only", False))


def _created_text_count(
    before: Optional[Sequence[Any]],
    after: Optional[Sequence[Any]],
    *,
    detection: bool,
) -> Optional[int]:
    if after is None:
        return None
    if before is not None:
        before_ids = {id(element) for element in before}
        return sum(id(element) not in before_ids for element in after)
    if detection:
        return sum(_is_detection_element(element) for element in after)
    return None


class _GuideRegionView(OCRScopeMixin, Sequence[Region]):
    """Shared lazy Sequence and aggregate OCR behavior for guide regions."""

    def __init__(self, guides: "Guides", indices: Optional[Tuple[int, ...]] = None):
        self._guides = guides
        self._indices = indices

    @property
    def guides(self) -> "Guides":
        return self._guides

    @property
    def last_ocr_result(self) -> Optional[GuideOCRResult]:
        return cast(Optional[GuideOCRResult], self._guides.last_ocr_result)

    def _selection(self, snapshot: _GuideGeometrySnapshot) -> Tuple[Any, ...]:
        raise NotImplementedError

    def _bbox_for_item(self, snapshot: _GuideGeometrySnapshot, item: Any) -> Bounds:
        raise NotImplementedError

    def _literal_windows(
        self, snapshot: _GuideGeometrySnapshot, selection: Sequence[Any]
    ) -> List[Dict[str, Any]]:
        return [
            {"selection": item, "bbox": self._bbox_for_item(snapshot, item)} for item in selection
        ]

    def _build_plan(
        self,
        request: OCRRequest,
        planning_options: Optional[GuideOCRPlanningOptions] = None,
    ) -> GuideOCRPlan:
        snapshot = _snapshot_guide_geometry(self._guides)
        selection = self._selection(snapshot)
        options = planning_options or GuideOCRPlanningOptions(window="cell")
        effective_request = _prepare_effective_request(snapshot.target, request)
        resolution = int(getattr(effective_request, "resolution", None) or options.min_resolution)
        windows = self._literal_windows(snapshot, selection)
        self._guides._annotate_ocr_windows(windows, resolution=resolution)
        max_side_px, max_area_px = self._guides._ocr_window_budget(
            getattr(effective_request, "engine", None),
            max_side_px=options.max_side_px,
            max_area_px=options.max_area_px,
        )
        return GuideOCRPlan(
            guides=self._guides,
            view=self,
            request=effective_request,
            resolution=resolution,
            window="logical",
            windows=tuple(windows),
            max_side_px=max_side_px,
            max_area_px=max_area_px,
            target_cell_px=options.target_cell_px,
            representative_percentile=options.representative_percentile,
            min_confidence=getattr(effective_request, "min_confidence", None),
            target=snapshot.target,
        )

    def _iter_ocr_hosts(self, request: OCRRequest) -> Iterable[Any]:
        plan = self._build_plan(request)
        bake = bool(getattr(plan.request, "apply_exclusions", True))
        for window in plan.windows:
            yield _region_for_target(
                plan.target,
                cast(Bounds, window["bbox"]),
                bake_callable_exclusions=bake,
            )

    def _execute_ocr_request(self, request: OCRRequest) -> None:
        self._apply_ocr_plan(self._build_plan(request))

    def _apply_ocr_plan(self, plan: GuideOCRPlan) -> GuideOCRResult:
        counts: List[Optional[int]] = []
        # Resolve callable host exclusions eagerly only when this OCR run will
        # actually apply exclusions; with apply_exclusions=False the callables
        # must never be invoked.
        bake = bool(getattr(plan.request, "apply_exclusions", True))
        with ocr_execution_session():
            for planned in plan.windows:
                region = _region_for_target(
                    plan.target,
                    cast(Bounds, planned["bbox"]),
                    bake_callable_exclusions=bake,
                )
                before_elements = _scoped_text_elements(region)
                _execute_region_ocr_request(region, plan.request)
                after_elements = _scoped_text_elements(region)
                created_count = _created_text_count(
                    before_elements,
                    after_elements,
                    detection=isinstance(plan.request, OCRDetectionRequest),
                )
                counts.append(created_count)

        result = GuideOCRResult(plan=plan, counts=counts)
        self._guides._ocr_applied = True
        self._guides._ocr_prefer_words = True
        self._guides._last_ocr_plan = plan
        self._guides._last_ocr_result = result
        return result


class GuideColumns(_GuideRegionView):
    """Lazy typed Sequence view over logical guide columns."""

    def _selection(self, snapshot: _GuideGeometrySnapshot) -> Tuple[int, ...]:
        size = max(0, len(snapshot.verticals) - 1)
        if self._indices is None:
            return tuple(range(size))
        for index in self._indices:
            _normalize_sequence_index(index, size, label="column")
        return self._indices

    def _bbox_for_item(self, snapshot: _GuideGeometrySnapshot, item: Any) -> Bounds:
        index = cast(int, item)
        _, top, _, bottom = snapshot.bounds
        return (snapshot.verticals[index], top, snapshot.verticals[index + 1], bottom)

    def __len__(self) -> int:
        return len(self._selection(_snapshot_guide_geometry(self._guides)))

    @overload
    def __getitem__(self, index: int) -> Region: ...

    @overload
    def __getitem__(self, index: slice) -> "GuideColumns": ...

    def __getitem__(self, index: Union[int, slice]) -> Union[Region, "GuideColumns"]:
        snapshot = _snapshot_guide_geometry(self._guides)
        selection = self._selection(snapshot)
        if isinstance(index, slice):
            return GuideColumns(self._guides, selection[index])
        selected = selection[_normalize_sequence_index(index, len(selection), label="column")]
        return _region_for_bbox(snapshot, self._bbox_for_item(snapshot, selected))

    def __iter__(self) -> Iterator[Region]:
        snapshot = _snapshot_guide_geometry(self._guides)
        for selected in self._selection(snapshot):
            yield _region_for_bbox(snapshot, self._bbox_for_item(snapshot, selected))


class GuideRows(_GuideRegionView):
    """Lazy typed Sequence view over logical guide rows."""

    def _selection(self, snapshot: _GuideGeometrySnapshot) -> Tuple[int, ...]:
        size = max(0, len(snapshot.horizontals) - 1)
        if self._indices is None:
            return tuple(range(size))
        for index in self._indices:
            _normalize_sequence_index(index, size, label="row")
        return self._indices

    def _bbox_for_item(self, snapshot: _GuideGeometrySnapshot, item: Any) -> Bounds:
        index = cast(int, item)
        left, _, right, _ = snapshot.bounds
        return (left, snapshot.horizontals[index], right, snapshot.horizontals[index + 1])

    def __len__(self) -> int:
        return len(self._selection(_snapshot_guide_geometry(self._guides)))

    @overload
    def __getitem__(self, index: int) -> Region: ...

    @overload
    def __getitem__(self, index: slice) -> "GuideRows": ...

    def __getitem__(self, index: Union[int, slice]) -> Union[Region, "GuideRows"]:
        snapshot = _snapshot_guide_geometry(self._guides)
        selection = self._selection(snapshot)
        if isinstance(index, slice):
            return GuideRows(self._guides, selection[index])
        selected = selection[_normalize_sequence_index(index, len(selection), label="row")]
        return _region_for_bbox(snapshot, self._bbox_for_item(snapshot, selected))

    def __iter__(self) -> Iterator[Region]:
        snapshot = _snapshot_guide_geometry(self._guides)
        for selected in self._selection(snapshot):
            yield _region_for_bbox(snapshot, self._bbox_for_item(snapshot, selected))


class GuideCells(_GuideRegionView):
    """Lazy two-dimensional view over guide cells with aggregate OCR."""

    def __init__(
        self,
        guides: "Guides",
        cells: Optional[Tuple[Tuple[int, int], ...]] = None,
        *,
        matrix: bool = True,
    ):
        super().__init__(guides)
        self._cells = cells
        self._matrix = matrix

    def _selection(self, snapshot: _GuideGeometrySnapshot) -> Tuple[Tuple[int, int], ...]:
        num_rows = max(0, len(snapshot.horizontals) - 1)
        num_cols = max(0, len(snapshot.verticals) - 1)
        if self._cells is None:
            return tuple((row, col) for row in range(num_rows) for col in range(num_cols))
        for row, col in self._cells:
            _normalize_sequence_index(row, num_rows, label="row")
            _normalize_sequence_index(col, num_cols, label="column")
        return self._cells

    def _bbox_for_item(self, snapshot: _GuideGeometrySnapshot, item: Any) -> Bounds:
        row, col = cast(Tuple[int, int], item)
        return (
            snapshot.verticals[col],
            snapshot.horizontals[row],
            snapshot.verticals[col + 1],
            snapshot.horizontals[row + 1],
        )

    def __len__(self) -> int:
        return len(self._selection(_snapshot_guide_geometry(self._guides)))

    @staticmethod
    def _axis_selection(value: Union[int, slice], size: int, *, label: str) -> Tuple[int, ...]:
        if isinstance(value, slice):
            return tuple(range(*value.indices(size)))
        return (_normalize_sequence_index(value, size, label=label),)

    def __getitem__(self, key: Any) -> Union[Region, "GuideCells"]:  # type: ignore[override]
        # GuideCells deliberately preserves the existing 2-D accessor: an int
        # selects a row view in matrix mode, while (row, column) selects a cell.
        snapshot = _snapshot_guide_geometry(self._guides)
        selection = self._selection(snapshot)
        num_rows = max(0, len(snapshot.horizontals) - 1)
        num_cols = max(0, len(snapshot.verticals) - 1)
        if isinstance(key, tuple):
            if len(key) != 2:
                raise TypeError("Cell tuple indices must contain exactly (row, column).")
            row_key, col_key = key
            if not isinstance(row_key, (int, slice)) or not isinstance(col_key, (int, slice)):
                raise TypeError("Cell row and column indices must be integers or slices.")
            rows = self._axis_selection(row_key, num_rows, label="row")
            cols = self._axis_selection(col_key, num_cols, label="column")
            selected = tuple((row, col) for row in rows for col in cols)
            if self._cells is not None:
                selected = tuple(cell for cell in selected if cell in set(selection))
            if isinstance(row_key, int) and isinstance(col_key, int):
                if not selected:
                    raise IndexError("Requested cell is not part of this guide view.")
                return _region_for_bbox(snapshot, self._bbox_for_item(snapshot, selected[0]))
            return GuideCells(self._guides, selected, matrix=False)
        if isinstance(key, slice):
            if self._matrix:
                rows = tuple(range(*key.indices(num_rows)))
                selected = tuple((row, col) for row in rows for col in range(num_cols))
                if self._cells is not None:
                    selected = tuple(cell for cell in selected if cell in set(selection))
                return GuideCells(self._guides, selected, matrix=False)
            return GuideCells(self._guides, selection[key], matrix=False)
        if isinstance(key, int):
            if self._matrix:
                row = _normalize_sequence_index(key, num_rows, label="row")
                selected = tuple((row, col) for col in range(num_cols))
                if self._cells is not None:
                    selected = tuple(cell for cell in selected if cell in set(selection))
                return GuideCells(self._guides, selected, matrix=False)
            selected_cell = selection[_normalize_sequence_index(key, len(selection), label="cell")]
            return _region_for_bbox(snapshot, self._bbox_for_item(snapshot, selected_cell))
        raise TypeError("Cell indices must be integers, slices, or a (row, column) tuple.")

    def __iter__(self) -> Iterator[Region]:
        snapshot = _snapshot_guide_geometry(self._guides)
        for selected in self._selection(snapshot):
            yield _region_for_bbox(snapshot, self._bbox_for_item(snapshot, selected))

    @staticmethod
    def _is_contiguous_rectangle(selection: Sequence[Tuple[int, int]]) -> bool:
        if not selection:
            return False
        rows = sorted({row for row, _ in selection})
        cols = sorted({col for _, col in selection})
        return (
            rows == list(range(rows[0], rows[-1] + 1))
            and cols == list(range(cols[0], cols[-1] + 1))
            and set(selection) == {(row, col) for row in rows for col in cols}
        )

    def _build_plan(
        self,
        request: OCRRequest,
        planning_options: Optional[GuideOCRPlanningOptions] = None,
    ) -> GuideOCRPlan:
        snapshot = _snapshot_guide_geometry(self._guides)
        selection = self._selection(snapshot)
        options = planning_options or GuideOCRPlanningOptions()
        resolved_request: OCRRequest
        if isinstance(request, OCRFunctionRequest):
            resolution = options.min_resolution
            windows = self._literal_windows(snapshot, selection)
            window_mode: Union[str, Tuple[int, int]] = "cell"
            resolved_request = request
            resolved_min_confidence = None
        else:
            resolution = self._guides._resolve_ocr_resolution_for_guides(
                snapshot.verticals,
                snapshot.horizontals,
                resolution=(
                    request.resolution
                    if request.resolution is not None
                    else _configured_ocr_resolution(snapshot.target)
                ),
                target_cell_px=options.target_cell_px,
                representative_percentile=options.representative_percentile,
                min_resolution=options.min_resolution,
                max_resolution=options.max_resolution,
            )
            engine_request = _prepare_effective_request(
                snapshot.target,
                request,
                resolution=resolution,
            )
            if isinstance(engine_request, OCRFunctionRequest):  # pragma: no cover - type guard
                raise TypeError("guide engine OCR unexpectedly resolved to function mode")
            resolved_request = engine_request
            resolution = cast(int, engine_request.resolution)
            resolved_min_confidence = engine_request.min_confidence
            window_mode = cast(Union[str, Tuple[int, int]], options.window)
            max_side_px, max_area_px = self._guides._ocr_window_budget(
                engine_request.engine,
                max_side_px=options.max_side_px,
                max_area_px=options.max_area_px,
            )
            if self._is_contiguous_rectangle(selection):
                rows = sorted({row for row, _ in selection})
                cols = sorted({col for _, col in selection})
                windows = self._guides._plan_ocr_windows(
                    snapshot.verticals[cols[0] : cols[-1] + 2],
                    snapshot.horizontals[rows[0] : rows[-1] + 2],
                    window=options.window,
                    resolution=resolution,
                    max_side_px=max_side_px,
                    max_area_px=max_area_px,
                    vertical_ratio=options.vertical_ratio,
                    mixed_cell_threshold=options.mixed_cell_threshold,
                    large_cell_ratio=options.large_cell_ratio,
                )
                for planned in windows:
                    col_start, col_stop = planned["cols"]
                    row_start, row_stop = planned["rows"]
                    planned["cols"] = (col_start + cols[0], col_stop + cols[0])
                    planned["rows"] = (row_start + rows[0], row_stop + rows[0])
            else:
                windows = self._literal_windows(snapshot, selection)
                for planned, (row, col) in zip(windows, selection):
                    planned.update({"rows": (row, row + 1), "cols": (col, col + 1)})
        max_side_px, max_area_px = self._guides._ocr_window_budget(
            getattr(resolved_request, "engine", None),
            max_side_px=options.max_side_px,
            max_area_px=options.max_area_px,
        )
        self._guides._annotate_ocr_windows(windows, resolution=resolution)
        return GuideOCRPlan(
            guides=self._guides,
            view=self,
            request=resolved_request,
            resolution=resolution,
            window=window_mode,
            windows=tuple(windows),
            max_side_px=max_side_px,
            max_area_px=max_area_px,
            target_cell_px=options.target_cell_px,
            representative_percentile=options.representative_percentile,
            min_confidence=resolved_min_confidence,
            target=snapshot.target,
        )

    def plan_ocr(
        self,
        engine: Optional[str] = None,
        *,
        options: Optional[Any] = None,
        languages: Optional[List[str]] = None,
        min_confidence: Optional[float] = None,
        device: Optional[str] = None,
        resolution: Optional[int] = None,
        detect_only: bool = False,
        apply_exclusions: bool = True,
        replace: Literal["ocr", "all", "none"] = "ocr",
        use_cache: bool = True,
        model: Optional[str] = None,
        client: Optional[Any] = None,
        prompt: Optional[str] = None,
        instructions: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        layout: Optional[bool | str] = None,
        preserve_markup: bool = False,
        function: Optional[Callable[[Any], Optional[str]]] = None,
        source_label: str = "custom-ocr",
        confidence: Optional[float] = None,
        planning_options: Optional[GuideOCRPlanningOptions] = None,
    ) -> GuideOCRPlan:
        """Snapshot an exact cells OCR plan without running OCR."""

        if planning_options is not None and not isinstance(
            planning_options, GuideOCRPlanningOptions
        ):
            raise TypeError("planning_options must be GuideOCRPlanningOptions or None")
        request = normalize_ocr_request(
            engine=engine,
            options=options,
            languages=languages,
            min_confidence=min_confidence,
            device=device,
            resolution=resolution,
            detect_only=detect_only,
            apply_exclusions=apply_exclusions,
            replace=replace,
            use_cache=use_cache,
            model=model,
            client=client,
            prompt=prompt,
            instructions=instructions,
            max_new_tokens=max_new_tokens,
            layout=layout,
            preserve_markup=preserve_markup,
            function=function,
            source_label=source_label,
            confidence=confidence,
        )
        plan = self._build_plan(request, planning_options)
        self._guides._last_ocr_plan = plan
        return plan


# Deprecated alias: ``GuidesOcrResult`` was the pre-refactor name of
# ``GuideOCRResult``. Kept so historical import paths (including the
# ``natural_pdf.analyzers.guides`` shim) continue to work.
GuidesOcrResult = GuideOCRResult

__all__ = [
    "GuideCells",
    "GuideColumns",
    "GuideOCRPlan",
    "GuideOCRPlanningOptions",
    "GuideOCRResult",
    "GuideRows",
    "GuidesOcrResult",
]
