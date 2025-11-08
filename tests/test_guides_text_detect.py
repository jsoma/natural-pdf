from __future__ import annotations

from natural_pdf.analyzers.guides.text_detect import (
    find_horizontal_element_gaps,
    find_vertical_element_gaps,
)


class DummyElement:
    def __init__(self, x0, x1, top, bottom):
        self.x0 = x0
        self.x1 = x1
        self.top = top
        self.bottom = bottom


def test_find_vertical_element_gaps_respects_min_gap():
    elements = [
        DummyElement(0, 20, 0, 10),
        DummyElement(40, 60, 0, 10),
        DummyElement(80, 90, 0, 10),
    ]
    bounds = (0, 0, 100, 100)

    gaps = find_vertical_element_gaps(bounds, elements, min_gap=15)

    assert gaps == [(20, 40), (60, 80)]


def test_find_horizontal_element_gaps_ignores_non_overlapping_elements():
    elements = [
        DummyElement(0, 20, 0, 15),
        DummyElement(0, 20, 40, 60),
        DummyElement(0, 20, 70, 80),
        DummyElement(80, 120, 10, 90),  # Outside horizontal bounds, ignored
    ]
    bounds = (0, 0, 40, 100)

    gaps = find_horizontal_element_gaps(bounds, elements, min_gap=10)

    assert gaps == [(15, 40), (60, 70)]
