from __future__ import annotations

from types import SimpleNamespace

import pytest

from natural_pdf.analyzers.guides import Guides
from natural_pdf.flows.flow import Flow
from natural_pdf.flows.region import FlowRegion


@pytest.mark.parametrize(
    ("label", "axis", "expected", "class_factory", "list_factory"),
    [
        (
            "lines",
            "vertical",
            [10.0, 30.0],
            lambda obj: Guides.from_lines(obj, axis="vertical"),
            lambda guides: guides.vertical.from_lines(),
        ),
        (
            "content",
            "vertical",
            [15.0, 25.0],
            lambda obj: Guides.from_content(obj, axis="vertical", markers=["A"]),
            lambda guides: guides.vertical.from_content(markers=["A"]),
        ),
        (
            "whitespace",
            "horizontal",
            [20.0, 40.0],
            lambda obj: Guides.from_whitespace(obj, axis="horizontal", min_gap=12),
            lambda guides: guides.horizontal.from_whitespace(min_gap=12),
        ),
        (
            "headers",
            "vertical",
            [12.0, 42.0],
            lambda obj: Guides.from_headers(obj, headers=["H1", "H2"]),
            lambda guides: guides.vertical.from_headers(["H1", "H2"]),
        ),
        (
            "stripes",
            "horizontal",
            [50.0, 70.0],
            lambda obj: Guides.from_stripes(obj, axis="horizontal", color="#00ffff"),
            lambda guides: guides.horizontal.from_stripes(color="#00ffff"),
        ),
    ],
)
def test_guides_and_guideslist_generation_match(
    monkeypatch, label, axis, expected, class_factory, list_factory
):
    page = SimpleNamespace(bbox=(0, 0, 100, 100))
    mapping = {
        ("lines", "vertical"): [10.0, 30.0],
        ("content", "vertical"): [15.0, 25.0],
        ("whitespace", "horizontal"): [20.0, 40.0],
        ("headers", "vertical"): [12.0, 42.0],
        ("stripes", "horizontal"): [50.0, 70.0],
    }

    def _fake_run_guides_detect(*, axis, method, context, options):
        return SimpleNamespace(coordinates=mapping[(method, axis)])

    monkeypatch.setattr(
        "natural_pdf.analyzers.guides._generation.run_guides_detect",
        _fake_run_guides_detect,
    )

    class_guides = class_factory(page)
    list_guides = Guides(context=page)
    result = list_factory(list_guides)

    assert result is list_guides
    if axis == "vertical":
        assert list(class_guides.vertical) == expected, label
        assert list(list_guides.vertical) == expected, label
        assert list(class_guides.horizontal) == []
        assert list(list_guides.horizontal) == []
    else:
        assert list(class_guides.horizontal) == expected, label
        assert list(list_guides.horizontal) == expected, label
        assert list(class_guides.vertical) == []
        assert list(list_guides.vertical) == []


def test_axis_updates_dedupe_and_sort_across_guides_and_guideslist():
    guides = Guides(context=SimpleNamespace(bbox=(0, 0, 100, 100)))

    guides.vertical = [30, 10, 10]
    assert guides.vertical.data == [10.0, 30.0]

    guides.vertical.extend([20, 10, 30])
    assert guides.vertical.data == [10.0, 20.0, 30.0]

    guides.add_vertical(25)
    assert guides.vertical.data == [10.0, 20.0, 25.0, 30.0]

    guides.horizontal.data = [40, 20, 20]
    assert guides.horizontal.data == [20.0, 40.0]

    guides.horizontal.append(30)
    assert guides.horizontal.data == [20.0, 30.0, 40.0]


def test_flow_axis_updates_refresh_flow_guides_and_cache(practice_pdf_fresh):
    page = practice_pdf_fresh.pages[0]
    region_a = page.region(0, 0, 50, 50)
    region_b = page.region(60, 0, 100, 50)
    flow = Flow(segments=[page], arrangement="vertical")
    flow_region = FlowRegion(
        flow=flow, constituent_regions=[region_a, region_b], source_flow_element=None
    )

    guides = Guides(context=flow_region)
    guides.vertical.data = [10, 70]

    _ = guides.vertical.data
    assert guides._vertical_cache == [10.0, 70.0]
    assert guides._flow_guides[region_a][0] == [10.0]
    assert guides._flow_guides[region_b][0] == [70.0]

    guides.vertical.append(20)

    assert guides._vertical_cache is None
    assert guides._flow_guides[region_a][0] == [10.0, 20.0]
    assert guides._flow_guides[region_b][0] == [70.0]
    assert guides.vertical.data == [10.0, 20.0, 70.0]
