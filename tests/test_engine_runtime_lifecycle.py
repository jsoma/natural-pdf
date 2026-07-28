"""Focused lifecycle coverage for public provider-backed runtime paths."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from PIL import Image

import natural_pdf.engine_provider as provider_module
from natural_pdf.checkbox.checkbox_analyzer import CheckboxAnalyzer
from natural_pdf.checkbox.checkbox_options import BaseCheckboxOptions
from natural_pdf.deskew.deskew_provider import DeskewApplyResult, run_deskew_detect
from natural_pdf.engine_provider import EngineProvider
from natural_pdf.guides.guides_provider import GuidesDetectionResult, run_guides_detect
from natural_pdf.layout.layout_analyzer import LayoutAnalyzer
from natural_pdf.layout.layout_options import BaseLayoutOptions
from natural_pdf.selectors.selector_provider import run_selector_engine
from natural_pdf.tables.structure_provider import (
    StructureDetectionResult,
    run_table_structure_engine,
)
from natural_pdf.tables.table_provider import run_table_engine


@pytest.fixture
def provider(monkeypatch):
    instance = EngineProvider()
    instance._entry_points_loaded = True
    monkeypatch.setattr(provider_module, "_PROVIDER", instance)
    yield instance
    instance.clear()


def test_table_and_structure_runtime_cleanup_transient_engines(provider):
    events = []

    class TableEngine:
        def extract_tables(self, **kwargs):
            events.append("table.run")
            return [[[]]]

        def cleanup(self):
            events.append("table.cleanup")

    class StructureEngine:
        def detect(self, **kwargs):
            events.append("structure.run")
            return StructureDetectionResult()

        def cleanup(self):
            events.append("structure.cleanup")

    provider.register("tables", "temporary", lambda **_: TableEngine(), lifetime="transient")
    provider.register(
        "tables.detect_structure",
        "temporary",
        lambda **_: StructureEngine(),
        lifetime="transient",
    )

    context = object()
    assert run_table_engine(context=context, region=context, engine_name="temporary") == [[[]]]
    assert isinstance(
        run_table_structure_engine(context=context, region=context, engine_name="temporary"),
        StructureDetectionResult,
    )
    assert events == [
        "table.run",
        "table.cleanup",
        "structure.run",
        "structure.cleanup",
    ]


def test_guides_deskew_and_selector_runtime_cleanup_transient_engines(provider):
    events = []

    class GuidesEngine:
        def detect(self, **kwargs):
            events.append("guides.run")
            return GuidesDetectionResult(coordinates=[3.0])

        def cleanup(self):
            events.append("guides.cleanup")

    class DeskewEngine:
        def detect(self, **kwargs):
            events.append("deskew.run")
            return 1.5

        def apply(self, **kwargs):
            return DeskewApplyResult(image=Image.new("RGB", (1, 1)), angle=1.5)

        def cleanup(self):
            events.append("deskew.cleanup")

    class SelectorEngine:
        def query(self, **kwargs):
            events.append("selector.run")
            return SimpleNamespace(elements=["selected"])

        def cleanup(self):
            events.append("selector.cleanup")

    class SelectorHost:
        def selector_page(self):
            return self

        def selector_region(self):
            return None

        def selector_flow(self):
            return None

    provider.register(
        "guides.detect", "temporary", lambda **_: GuidesEngine(), lifetime="transient"
    )
    provider.register(
        "deskew.detect", "temporary", lambda **_: DeskewEngine(), lifetime="transient"
    )
    provider.register("selectors", "temporary", lambda **_: SelectorEngine(), lifetime="transient")

    context = object()
    assert run_guides_detect(
        axis="vertical", method="content", context=context, engine_name="temporary"
    ).coordinates == [3.0]
    assert run_deskew_detect(target=context, context=context, engine_name="temporary") == 1.5
    assert run_selector_engine(SelectorHost(), "text", engine_name="temporary") == ["selected"]
    assert events == [
        "guides.run",
        "guides.cleanup",
        "deskew.run",
        "deskew.cleanup",
        "selector.run",
        "selector.cleanup",
    ]


def test_checkbox_runtime_cleanup_follows_detection(provider):
    events = []

    class Page:
        width = 20
        height = 10

        def render(self, resolution=72):
            return Image.new("RGB", (20, 10))

    class Engine:
        def detect(self, image, options, context):
            events.append("detect")
            return []

        def cleanup(self):
            events.append("cleanup")

    provider.register("checkbox", "temporary", lambda **_: Engine(), lifetime="transient")
    analyzer = CheckboxAnalyzer(Page())

    assert analyzer._try_engine("temporary", BaseCheckboxOptions()) == []
    assert events == ["detect", "cleanup"]


def test_layout_transient_engine_stays_live_through_post_processing(provider):
    events = []

    class Page:
        number = 1
        width = 20
        height = 10

        def get_config(self, key, default=None, **kwargs):
            return default

        def render(self, resolution=72):
            return Image.new("RGB", (20, 10))

        def clear_detected_layout_regions(self):
            events.append("clear")

        def add_region(self, region, source=None):  # pragma: no cover - no detections
            raise AssertionError("No regions should be added")

    class Engine:
        def detect(self, image, options, context=None):
            events.append("detect")
            return []

        def post_process_regions(self, regions, options):
            events.append("post_process")

        def cleanup(self):
            events.append("cleanup")

    provider.register("layout", "temporary", lambda **_: Engine(), lifetime="transient")
    result = LayoutAnalyzer(Page()).analyze_layout(engine="temporary", options=BaseLayoutOptions())

    assert result == []
    assert events == ["detect", "clear", "post_process", "cleanup"]
