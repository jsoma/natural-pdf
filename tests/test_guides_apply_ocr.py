from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

import natural_pdf.analyzers.guides.ocr as guides_ocr
from natural_pdf.analyzers.guides import (
    GuideCells,
    GuideColumns,
    GuideOCRPlan,
    GuideOCRPlanningOptions,
    GuideOCRResult,
    GuideRows,
    Guides,
)
from natural_pdf.core.ocr_contracts import OCRFunctionRequest, OCRRecognitionRequest
from natural_pdf.core.ocr_mixin import OCRScopeMixin
from natural_pdf.elements.region import Region
from natural_pdf.flows.region import FlowRegion


def _guides(page) -> Guides:
    return Guides(verticals=[0, 10, 20], horizontals=[0, 10, 20], context=page)


def test_guides_has_no_top_level_apply_ocr(practice_pdf):
    assert not hasattr(_guides(practice_pdf.pages[0]), "apply_ocr")
    assert GuideCells.apply_ocr is OCRScopeMixin.apply_ocr
    assert GuideRows.apply_ocr is OCRScopeMixin.apply_ocr
    assert GuideColumns.apply_ocr is OCRScopeMixin.apply_ocr


def test_public_slices_remain_guide_views(practice_pdf):
    guides = _guides(practice_pdf.pages[0])

    assert isinstance(guides.columns[:], GuideColumns)
    assert isinstance(guides.rows[:], GuideRows)
    assert isinstance(guides.cells[:, :], GuideCells)
    assert isinstance(guides.cells[0][:], GuideCells)
    assert guides.cells[-1, -1].bbox == guides.cells[1][1].bbox


def test_builtin_cells_group_while_custom_ocr_visits_every_cell(practice_pdf, monkeypatch):
    guides = _guides(practice_pdf.pages[0])
    calls = []

    monkeypatch.setattr(
        Region,
        "_execute_ocr_request",
        lambda region, request: calls.append((region.bbox, request)),
        raising=False,
    )

    view = guides.cells
    assert view.apply_ocr(engine="rapidocr", resolution=72) is view
    assert len(calls) < len(view)
    assert all(isinstance(request, OCRRecognitionRequest) for _, request in calls)
    assert isinstance(view.last_ocr_result, GuideOCRResult)
    assert guides.last_ocr_result is view.last_ocr_result

    calls.clear()
    assert view.apply_ocr(function=lambda region: "text") is view
    assert len(calls) == len(view)
    assert all(isinstance(request, OCRFunctionRequest) for _, request in calls)


def test_disjoint_cell_selection_never_covers_unselected_cells(practice_pdf, monkeypatch):
    guides = Guides(
        verticals=[0, 10, 20, 30],
        horizontals=[0, 10, 20],
        context=practice_pdf.pages[0],
    )
    selected = guides.cells[:, ::2]
    calls = []
    monkeypatch.setattr(
        Region,
        "_execute_ocr_request",
        lambda region, request: calls.append(region.bbox),
        raising=False,
    )

    selected.apply_ocr(engine="rapidocr", resolution=72)

    assert calls == [region.bbox for region in selected]


def test_detection_groups_cells_but_rows_and_columns_are_literal(practice_pdf, monkeypatch):
    guides = _guides(practice_pdf.pages[0])
    calls = []
    monkeypatch.setattr(
        Region,
        "_execute_ocr_request",
        lambda region, request: calls.append(region.bbox),
        raising=False,
    )

    guides.cells.apply_ocr(engine="rapidocr", resolution=72, detect_only=True)
    assert len(calls) < len(guides.cells)

    calls.clear()
    selected_rows = guides.rows[:1]
    selected_rows.apply_ocr(engine="rapidocr", resolution=72)
    assert calls == [region.bbox for region in selected_rows]

    calls.clear()
    selected_columns = guides.columns[-1:]
    selected_columns.apply_ocr(engine="rapidocr", resolution=72)
    assert calls == [region.bbox for region in selected_columns]


def test_rows_freeze_configured_effective_request_defaults(practice_pdf, monkeypatch):
    page = practice_pdf.pages[0]
    guides = _guides(page)
    requests = []
    monkeypatch.setitem(
        page._context._options,
        "ocr",
        {
            "ocr_engine": "rapidocr",
            "ocr_languages": ["fr"],
            "ocr_min_confidence": 0.2,
            "ocr_device": "cpu",
            "resolution": 333,
        },
    )
    monkeypatch.setattr(
        Region,
        "_execute_ocr_request",
        lambda region, request: requests.append(request),
        raising=False,
    )

    guides.rows[:1].apply_ocr()

    request = requests[0]
    assert request.engine == "rapidocr"
    assert request.languages == ("fr",)
    assert request.min_confidence == 0.2
    assert request.device == "cpu"
    assert request.resolution == 333
    assert guides.last_ocr_result.resolution == 333


def test_region_host_config_and_exclusions_flow_into_guide_ocr_windows(practice_pdf, monkeypatch):
    """Windowed guide OCR must resolve defaults against the ORIGINAL host region
    and keep honoring the host's region-local exclusions (regression: fresh
    parentless window Regions dropped both)."""
    from natural_pdf.services.base import resolve_service

    page = practice_pdf.pages[0]
    host = Region(page, (0.0, 0.0, 200.0, 200.0))
    host.metadata["config"] = {
        "ocr_languages": ["fr"],
        "ocr_min_confidence": 0.73,
        "ocr_device": "cpu",
        "resolution": 333,
    }
    mask = Region(page, (0.0, 0.0, 50.0, 50.0))
    host._exclusions = [(mask, "mask", "region")]
    guides = Guides(verticals=[0, 100, 200], horizontals=[0, 100, 200], context=host)

    captured = []
    monkeypatch.setattr(
        Region,
        "_execute_ocr_request",
        lambda region, request: captured.append((region, request)),
        raising=False,
    )

    guides.rows[:1].apply_ocr(engine="rapidocr")

    window, request = captured[0]
    assert request.engine == "rapidocr"
    assert request.languages == ("fr",)
    assert request.min_confidence == 0.73
    assert request.device == "cpu"
    assert request.resolution == 333

    # The exact mask geometry the OCR render path will use for this window.
    service = resolve_service(window, "ocr")
    render_kwargs = service._render_kwargs(window, apply_exclusions=True)
    assert (0.0, 0.0, 50.0, 50.0) in render_kwargs["_ocr_exclusion_bboxes"]


def test_callable_host_exclusions_resolve_against_host_not_window(practice_pdf, monkeypatch):
    """Callable exclusions are defined against the ORIGINAL host region.
    Windows must receive the host-resolved static geometry — the callable
    must never be invoked with a window object (which would change its
    meaning: unmasking intended content or masking unrelated content)."""
    from natural_pdf.services.base import resolve_service

    page = practice_pdf.pages[0]
    host = Region(page, (0.0, 0.0, 200.0, 200.0))
    mask = Region(page, (0.0, 0.0, 50.0, 50.0))

    callable_args = []

    def dynamic_mask(target):
        callable_args.append(target)
        return mask

    host._exclusions = [(dynamic_mask, "mask", "region")]
    guides = Guides(verticals=[0, 100, 200], horizontals=[0, 100, 200], context=host)

    captured = []
    monkeypatch.setattr(
        Region,
        "_execute_ocr_request",
        lambda region, request: captured.append(region),
        raising=False,
    )

    guides.rows[:1].apply_ocr(engine="rapidocr")

    window = captured[0]
    assert window is not host

    # The callable was resolved, and only ever against the original host.
    assert callable_args, "callable exclusion was never resolved"
    assert all(arg is host for arg in callable_args)

    # The window carries baked geometry, not the callable itself.
    assert all(not callable(entry[0]) for entry in window._exclusions)

    # Effective exclusion geometry equals the host-resolved region.
    service = resolve_service(window, "ocr")
    render_kwargs = service._render_kwargs(window, apply_exclusions=True)
    assert (0.0, 0.0, 50.0, 50.0) in render_kwargs["_ocr_exclusion_bboxes"]
    # Rendering the window did not re-invoke the callable with the window.
    assert all(arg is host for arg in callable_args)


def test_plain_guide_region_access_never_invokes_callable_exclusions(practice_pdf):
    """Materializing guide regions (cells/rows/columns access and iteration)
    must not run callable host exclusions: they may be expensive or raise, and
    plain access is not a read of excluded content."""
    page = practice_pdf.pages[0]
    host = Region(page, (0.0, 0.0, 200.0, 200.0))

    calls = []

    def raising_mask(target):
        calls.append(target)
        raise RuntimeError("callable exclusion must not run on plain access")

    host._exclusions = [(raising_mask, "mask", "region")]
    guides = Guides(verticals=[0, 100, 200], horizontals=[0, 100, 200], context=host)

    cell = guides.cells[0, 0]
    assert cell.bbox == (0.0, 0.0, 100.0, 100.0)
    list(guides.cells)
    list(guides.rows)
    list(guides.columns)

    assert calls == []


def test_apply_ocr_without_exclusions_never_invokes_callable_exclusions(practice_pdf, monkeypatch):
    """apply_ocr(apply_exclusions=False) must not execute callable host
    exclusions: the run will not apply exclusions, so resolving them is pure
    side effect."""
    page = practice_pdf.pages[0]
    host = Region(page, (0.0, 0.0, 200.0, 200.0))
    mask = Region(page, (0.0, 0.0, 50.0, 50.0))

    calls = []

    def dynamic_mask(target):
        calls.append(target)
        return mask

    host._exclusions = [(dynamic_mask, "mask", "region")]
    guides = Guides(verticals=[0, 100, 200], horizontals=[0, 100, 200], context=host)

    captured = []
    monkeypatch.setattr(
        Region,
        "_execute_ocr_request",
        lambda region, request: captured.append(region),
        raising=False,
    )

    guides.rows[:1].apply_ocr(engine="rapidocr", apply_exclusions=False)

    assert captured, "OCR was never dispatched"
    assert calls == [], "callable exclusion ran despite apply_exclusions=False"


def test_cells_view_prefers_host_configured_resolution_over_auto(practice_pdf, monkeypatch):
    page = practice_pdf.pages[0]
    host = Region(page, (0.0, 0.0, 200.0, 200.0))
    host.metadata["config"] = {"resolution": 333}
    guides = Guides(verticals=[0, 100, 200], horizontals=[0, 100, 200], context=host)

    requests = []
    monkeypatch.setattr(
        Region,
        "_execute_ocr_request",
        lambda region, request: requests.append(request),
        raising=False,
    )

    guides.cells.apply_ocr(engine="rapidocr")

    assert all(request.resolution == 333 for request in requests)
    assert guides.last_ocr_result.resolution == 333


def test_detection_result_counts_refreshed_artifacts(practice_pdf, monkeypatch):
    guides = _guides(practice_pdf.pages[0])
    selected = guides.cells[0][:1]
    old_detection = object()
    new_detection = object()
    snapshots = iter([[old_detection], [new_detection]])
    monkeypatch.setattr(guides_ocr, "_scoped_text_elements", lambda region: next(snapshots))
    monkeypatch.setattr(Region, "_execute_ocr_request", lambda *args: None, raising=False)

    selected.apply_ocr(engine="rapidocr", resolution=72, detect_only=True)

    assert selected.last_ocr_result.counts == [1]
    assert selected.last_ocr_result.total_created == 1


def test_view_ocr_does_not_grow_page_or_parent_region_trees(practice_pdf, monkeypatch):
    page = practice_pdf.pages[0]
    parent = Region(page, (0, 0, 20, 20))
    guides = Guides(verticals=[0, 10, 20], horizontals=[0, 10, 20], context=parent)
    detected_before = len(page._regions["detected"])
    children_before = len(parent.child_regions)
    monkeypatch.setattr(Region, "_execute_ocr_request", lambda *args: None, raising=False)

    guides.cells.apply_ocr(engine="rapidocr", resolution=72)
    list(guides.rows)
    list(guides.columns)

    assert len(page._regions["detected"]) == detected_before
    assert len(parent.child_regions) == children_before


def test_flow_region_guide_views_raise_an_explicit_error(practice_pdf):
    constituent = Region(practice_pdf.pages[0], (0, 0, 10, 10))
    guides = Guides(
        verticals=[0, 10],
        horizontals=[0, 10],
        context=FlowRegion(None, [constituent]),
    )

    with pytest.raises(ValueError, match="FlowRegion"):
        guides.cells.apply_ocr(engine="rapidocr")


def test_plan_snapshots_geometry_and_apply_returns_result(practice_pdf, monkeypatch):
    guides = _guides(practice_pdf.pages[0])
    plan = guides.cells.plan_ocr(
        engine="rapidocr",
        resolution=72,
        planning_options=GuideOCRPlanningOptions(window="table"),
    )
    assert isinstance(plan, GuideOCRPlan)
    assert plan.summary()["window_count"] == 1
    bbox = plan.windows[0]["bbox"]
    with pytest.raises(FrozenInstanceError):
        plan.resolution = 300
    with pytest.raises(TypeError):
        plan.windows[0]["bbox"] = (1, 1, 2, 2)
    exported = plan.to_dict()
    exported["windows"][0]["bbox"] = (1, 1, 2, 2)
    assert plan.windows[0]["bbox"] == bbox
    guides.vertical[1] = 15
    calls = []
    monkeypatch.setattr(
        Region,
        "_execute_ocr_request",
        lambda region, request: calls.append(region.bbox),
        raising=False,
    )

    result = plan.apply()

    assert isinstance(result, GuideOCRResult)
    assert result.plan is plan
    assert calls == [bbox]
    assert guides.last_ocr_result is result


def test_custom_function_result_counts_and_exposes_custom_text(practice_pdf_fresh):
    guides = _guides(practice_pdf_fresh.pages[0])
    selected = guides.cells[0][:1]

    selected.apply_ocr(function=lambda region: "custom cell")

    assert selected.last_ocr_result.counts == [1]
    assert selected.last_ocr_result.total_created == 1
    assert [element.text for element in selected.last_ocr_result.ocr_elements()] == ["custom cell"]


def test_guides_auto_ocr_resolution_uses_trimmed_cell_box_percentile():
    guides = Guides(verticals=[0, 10, 30, 1030], horizontals=[0, 10, 20, 1000])

    resolution = guides._resolve_ocr_resolution_for_guides(
        [0, 10, 30, 1030],
        [0, 10, 20, 1000],
        resolution=None,
        target_cell_px=40,
        representative_percentile=25,
        min_resolution=150,
        max_resolution=400,
    )

    assert resolution == 288


def test_guides_auto_ocr_windows_separate_verticalish_and_horizontal_rows():
    guides = Guides(verticals=[0, 10, 20, 30], horizontals=[0, 40, 50])

    windows = guides._plan_ocr_windows(
        [0, 10, 20, 30],
        [0, 40, 50],
        window="auto",
        resolution=72,
        max_side_px=10_000,
        max_area_px=10_000_000,
        vertical_ratio=2,
        mixed_cell_threshold=0.5,
        large_cell_ratio=2,
    )

    assert [window["rows"] for window in windows] == [(0, 1), (1, 2)]


def test_guides_auto_ocr_windows_separate_unusually_large_horizontal_rows():
    guides = Guides(verticals=[0, 100, 200], horizontals=[0, 20, 100, 120])

    windows = guides._plan_ocr_windows(
        [0, 100, 200],
        [0, 20, 100, 120],
        window="auto",
        resolution=72,
        max_side_px=10_000,
        max_area_px=10_000_000,
        vertical_ratio=2,
        mixed_cell_threshold=0.5,
        large_cell_ratio=2,
    )

    assert [window["rows"] for window in windows] == [(0, 1), (1, 2), (2, 3)]


def test_guides_auto_ocr_windows_split_columns_by_rendered_width():
    guides = Guides(verticals=[0, 100, 200, 300], horizontals=[0, 20])

    windows = guides._plan_ocr_windows(
        [0, 100, 200, 300],
        [0, 20],
        window="auto",
        resolution=72,
        max_side_px=150,
        max_area_px=10_000_000,
        vertical_ratio=2,
        mixed_cell_threshold=0.5,
        large_cell_ratio=2,
    )

    assert [window["cols"] for window in windows] == [(0, 1), (1, 2), (2, 3)]


def test_guides_auto_ocr_windows_enforce_max_area_px_on_initial_row_window():
    """Four compliant 50k-px cells must not merge into a 200k-px window under a
    60k-px budget (regression: only max_side_px gated the column merge)."""
    guides = Guides(verticals=[0, 250, 500, 750, 1000], horizontals=[0, 200])

    windows = guides._plan_ocr_windows(
        [0, 250, 500, 750, 1000],
        [0, 200],
        window="auto",
        resolution=72,
        max_side_px=1000,
        max_area_px=60_000,
        vertical_ratio=2,
        mixed_cell_threshold=0.5,
        large_cell_ratio=2,
    )

    assert [window["cols"] for window in windows] == [(0, 1), (1, 2), (2, 3), (3, 4)]
    for window in windows:
        x0, top, x1, bottom = window["bbox"]
        assert (x1 - x0) * (bottom - top) <= 60_000


def test_guides_auto_ocr_windows_split_back_to_column_groups_under_area_budget():
    guides = Guides(verticals=[0, 250, 500, 750, 1000], horizontals=[0, 200])

    windows = guides._plan_ocr_windows(
        [0, 250, 500, 750, 1000],
        [0, 200],
        window="auto",
        resolution=72,
        max_side_px=1000,
        max_area_px=120_000,
        vertical_ratio=2,
        mixed_cell_threshold=0.5,
        large_cell_ratio=2,
    )

    assert [window["cols"] for window in windows] == [(0, 2), (2, 4)]
    for window in windows:
        x0, top, x1, bottom = window["bbox"]
        assert (x1 - x0) * (bottom - top) <= 120_000
