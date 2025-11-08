from __future__ import annotations

import pytest

import natural_pdf.analyzers.guides.base as guides_base
from natural_pdf.analyzers.guides import Guides


class DummyText:
    def __init__(self, x0, x1, top=0, bottom=10):
        self.x0 = x0
        self.x1 = x1
        self.top = top
        self.bottom = bottom


class DummyRegion:
    def __init__(self, elements):
        self._elements = elements
        self.bbox = (0.0, 0.0, 100.0, 100.0)

    def find_all(self, selector, **kwargs):
        assert selector == "text"
        return list(self._elements)


def test_snap_to_whitespace_uses_text_detection(monkeypatch):
    region = DummyRegion([DummyText(0, 10), DummyText(20, 30)])
    guides = Guides(verticals=[0, 50, 100], context=region)

    text_calls = {"count": 0}

    def fake_vertical_element(bounds, elements, min_gap):
        text_calls["count"] += 1
        return [(5, 15)]

    def explode(*args, **kwargs):
        raise AssertionError("pixel path should not be used")

    monkeypatch.setattr(guides_base, "find_vertical_element_gaps", fake_vertical_element)
    monkeypatch.setattr(guides_base, "find_vertical_whitespace_gaps", explode)

    captured = {}

    def fake_snap(self, guides_list, gaps, axis):
        captured["axis"] = axis
        captured["gaps"] = gaps

    monkeypatch.setattr(Guides, "_snap_guides_to_gaps", fake_snap)

    guides.snap_to_whitespace(axis="vertical", detection_method="text")

    assert text_calls["count"] == 1
    assert captured["axis"] == "vertical"
    assert captured["gaps"] == [(5, 15)]


def test_snap_to_whitespace_uses_pixel_detection(monkeypatch):
    region = DummyRegion([DummyText(0, 10), DummyText(20, 30)])
    guides = Guides(verticals=[0, 50, 100], context=region)

    pixel_calls = {"count": 0}

    def fake_vertical_pixels(bounds, elements, min_gap, threshold, guide_positions=None):
        pixel_calls["count"] += 1
        return [(10, 20)]

    def explode(*args, **kwargs):
        raise AssertionError("text path should not be used")

    monkeypatch.setattr(guides_base, "find_vertical_whitespace_gaps", fake_vertical_pixels)
    monkeypatch.setattr(guides_base, "find_vertical_element_gaps", explode)

    captured = {}

    def fake_snap(self, guides_list, gaps, axis):
        captured["axis"] = axis
        captured["gaps"] = gaps

    monkeypatch.setattr(Guides, "_snap_guides_to_gaps", fake_snap)

    guides.snap_to_whitespace(axis="vertical", detection_method="pixels")

    assert pixel_calls["count"] == 1
    assert captured["gaps"] == [(10, 20)]


def test_snap_to_whitespace_rejects_unknown_detection_method():
    region = DummyRegion([DummyText(0, 10)])
    guides = Guides(verticals=[0, 20], context=region)

    with pytest.raises(ValueError):
        guides.snap_to_whitespace(detection_method="unknown")
