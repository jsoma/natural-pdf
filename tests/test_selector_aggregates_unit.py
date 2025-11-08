"""Unit tests for aggregate helper functions in selectors parser."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from natural_pdf.selectors.parser import _calculate_aggregates


class DummyElement(SimpleNamespace):
    """Simple object whose attributes mimic selector targets."""


def _selector(attr_name: str, func: str, args=None):
    return {
        "attributes": [
            {
                "name": attr_name,
                "value": {"type": "aggregate", "func": func, "args": args},
            }
        ]
    }


def test_numeric_aggregates():
    elems = [DummyElement(size=value) for value in (10, 12, 14)]

    assert _calculate_aggregates(elems, _selector("size", "min"))["size"] == 10
    assert _calculate_aggregates(elems, _selector("size", "max"))["size"] == 14
    assert _calculate_aggregates(elems, _selector("size", "avg"))["size"] == pytest.approx(12)
    assert _calculate_aggregates(elems, _selector("size", "median"))["size"] == 12


def test_avg_with_non_numeric_values_returns_none():
    elems = [DummyElement(size="large"), DummyElement(size="small")]
    assert _calculate_aggregates(elems, _selector("size", "avg"))["size"] is None


def test_bbox_attributes_respected():
    elems = [DummyElement(_bbox=(idx, 0, idx + 5, 10)) for idx in (0, 7, 15)]
    assert _calculate_aggregates(elems, _selector("x0", "max"))["x0"] == 15


def test_mode_on_strings():
    elems = [DummyElement(fontname=font) for font in ("Arial", "Arial", "Courier")]
    assert _calculate_aggregates(elems, _selector("fontname", "mode"))["fontname"] == "Arial"


def test_closest_color():
    elems = [DummyElement(color=value) for value in ("#F00", "#0F0", "#1000FF")]
    # Closest to pure blue should pick the third entry
    result = _calculate_aggregates(elems, _selector("color", "closest", "blue"))
    assert result["color"] == "#1000FF"


def test_missing_values_yield_none():
    elems = []
    assert _calculate_aggregates(elems, _selector("size", "min"))["size"] is None
