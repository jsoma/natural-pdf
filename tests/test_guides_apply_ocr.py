from __future__ import annotations

import tqdm.auto

from natural_pdf.analyzers.guides import Guides, GuidesOcrResult


class FakeOcrRegion:
    def __init__(self, bbox, context):
        self.bbox = bbox
        self.context = context
        self.calls = []

    def apply_ocr(self, **kwargs):
        self.calls.append(kwargs)
        self.context.ocr_count += 2
        return self


class FakeOcrText:
    def __init__(self, bbox, text=""):
        self.bbox = bbox
        self.text = text
        self.confidence = 0.9


class FakeOcrContext:
    def __init__(self):
        self.bbox = (0, 0, 30, 20)
        self.created = []
        self.removed_ocr = 0
        self.ocr_count = 0
        self.ocr_elements = []
        self.show_calls = []

    def create_region(self, x0, top, x1, bottom):
        region = FakeOcrRegion((x0, top, x1, bottom), self)
        self.created.append(region)
        return region

    def remove_ocr_elements(self):
        self.removed_ocr += 1
        self.ocr_count = 0
        return 0

    def find_all(self, selector, **kwargs):
        assert selector == "text[source=ocr]"
        if self.ocr_elements:
            return self.ocr_elements
        return [object()] * self.ocr_count

    def show(self, **kwargs):
        self.show_calls.append(kwargs)
        return "preview"


def test_guides_apply_ocr_uses_cell_windows_and_clears_once():
    context = FakeOcrContext()
    guides = Guides(verticals=[0, 10, 30], horizontals=[0, 10, 20], context=context)

    result = guides.apply_ocr(
        resolution=72,
        window=(1, 1),
        engine="rapidocr",
        show_progress=False,
    )

    assert isinstance(result, GuidesOcrResult)
    assert result.guides is guides
    assert result.ran is True
    assert result.summary()["window_count"] == 4
    assert result.summary()["min_confidence"] == 0.5
    assert result.windows[0]["image_size"] == (10, 10)
    assert result.counts == [2, 2, 2, 2]
    assert result.total_created == 8
    assert context.removed_ocr == 1
    assert [region.bbox for region in context.created] == [
        (0.0, 0.0, 10.0, 10.0),
        (10.0, 0.0, 30.0, 10.0),
        (0.0, 10.0, 10.0, 20.0),
        (10.0, 10.0, 30.0, 20.0),
    ]
    assert all(region.calls[0]["replace"] is False for region in context.created)
    assert all(region.calls[0]["resolution"] == 72 for region in context.created)
    assert all(region.calls[0]["min_confidence"] == 0.5 for region in context.created)
    assert guides._ocr_prefer_words is True
    assert guides.last_ocr_result is result


def test_guides_apply_ocr_uses_tqdm_for_multiple_windows_by_default(monkeypatch):
    context = FakeOcrContext()
    guides = Guides(verticals=[0, 10, 30], horizontals=[0, 10, 20], context=context)
    calls = []

    def fake_tqdm(iterable, **kwargs):
        calls.append(kwargs)
        return iterable

    monkeypatch.setattr(tqdm.auto, "tqdm", fake_tqdm)

    result = guides.apply_ocr(resolution=72, window=(1, 1), engine="rapidocr")

    assert len(result.windows) == 4
    assert calls == [{"desc": "Applying guide-window OCR", "unit": "window"}]


def test_guides_apply_ocr_dry_run_returns_windows_without_running_ocr():
    context = FakeOcrContext()
    guides = Guides(verticals=[0, 10, 30], horizontals=[0, 10, 20], context=context)

    result = guides.apply_ocr(resolution=72, window=(1, 2), dry_run=True)

    assert result.ran is False
    assert len(result) == 2
    assert context.removed_ocr == 0
    assert context.created == []
    assert guides._ocr_prefer_words is False


def test_guides_ocr_result_extract_table_delegates_to_guides():
    context = FakeOcrContext()
    guides = Guides(verticals=[0, 10], horizontals=[0, 10], context=context)
    guides.extract_table = lambda *args, **kwargs: ("table", args, kwargs)

    result = guides.apply_ocr(resolution=72, window="table", dry_run=True)

    assert result.extract_table(header=None) == ("table", (), {"header": None})


def test_guides_ocr_result_show_highlights_windows_and_assigned_text():
    context = FakeOcrContext()
    context.ocr_elements = [
        FakeOcrText((1, 1, 2, 2), "left"),
        FakeOcrText((15, 1, 20, 2), "right"),
        FakeOcrText((100, 100, 110, 110), "outside"),
    ]
    guides = Guides(verticals=[0, 10, 30], horizontals=[0, 10], context=context)

    result = guides.apply_ocr(resolution=72, window=(1, 1), dry_run=True)
    preview = result.show(labels=False)

    assert preview == "preview"
    assert result.target is context
    assert len(context.show_calls) == 1
    call = context.show_calls[0]
    assert call["labels"] is False
    highlights = call["highlights"]
    window_highlights = [
        item
        for item in highlights
        if item.get("label", "").startswith("window ") and "bbox" in item
    ]
    text_highlights = [item for item in highlights if "element" in item]
    assert [item["bbox"] for item in window_highlights] == [
        (0.0, 0.0, 10.0, 10.0),
        (10.0, 0.0, 30.0, 10.0),
    ]
    assert [item["color"] for item in window_highlights] == [
        (37, 99, 235, 34),
        (22, 163, 74, 34),
    ]
    assert all(item["fill"] is True for item in window_highlights)
    assert all(item["line_width"] == 2.0 for item in window_highlights)
    assert all(item["vertices"] is False for item in window_highlights)
    assert [item["element"].text for item in text_highlights] == ["left", "right"]
    assert all(item["color"] == "red" for item in text_highlights)


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
