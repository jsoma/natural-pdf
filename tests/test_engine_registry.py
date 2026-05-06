import uuid

import pytest
from PIL import Image

from natural_pdf.engine_provider import get_provider
from natural_pdf.engine_registry import (
    register_classification_engine,
    register_deskew_engine,
    register_guides_engine,
    register_layout_engine,
    register_ocr_engine,
    register_selector_engine,
    register_table_function,
)
from natural_pdf.guides.guides_provider import GuidesDetectionResult
from natural_pdf.ocr import infer_engine_from_options
from natural_pdf.ocr.ocr_options import BaseOCROptions
from natural_pdf.ocr.unified_dispatch import get_registry, run_ocr
from natural_pdf.tables.result import TableResult
from natural_pdf.tables.table_provider import run_table_engine


class DummyRegion:
    def __init__(self):
        self.page = None


def test_register_table_function_accepts_tableresult():
    name = "table_func_tableresult"

    def engine(**kwargs):
        return TableResult([["A", "B"]])

    register_table_function(name, engine, replace=True)

    region = DummyRegion()
    tables = run_table_engine(context=region, region=region, engine_name=name)
    assert tables == [[["A", "B"]]]


def test_register_table_function_accepts_list_of_rows():
    name = "table_func_rows"

    def engine(**kwargs):
        return [["1", "2"], ["3", "4"]]

    register_table_function(name, engine, replace=True)

    region = DummyRegion()
    tables = run_table_engine(context=region, region=region, engine_name=name)
    assert tables == [[["1", "2"], ["3", "4"]]]


def test_register_table_function_accepts_multiple_tables():
    name = "table_func_multiple"

    def engine(**kwargs):
        return [
            [["a1"]],
            TableResult([["b1"]]),
        ]

    register_table_function(name, engine, replace=True)

    region = DummyRegion()
    tables = run_table_engine(context=region, region=region, engine_name=name)
    assert tables == [[["a1"]], [["b1"]]]


def test_register_table_function_invalid_return_type():
    name = "table_func_invalid"

    def engine(**kwargs):
        return "not a table"

    register_table_function(name, engine, replace=True)

    region = DummyRegion()
    with pytest.raises(TypeError):
        run_table_engine(context=region, region=region, engine_name=name)


def test_register_guides_engine_round_trip():
    name = f"guides.test.{uuid.uuid4().hex}"

    class DummyGuidesEngine:
        def detect(self, *, axis, method, context, options):
            return GuidesDetectionResult(coordinates=[1.0])

    register_guides_engine(name, lambda **_: DummyGuidesEngine())

    provider = get_provider()
    engine = provider.get("guides.detect", context=None, name=name)
    result = engine.detect(axis="vertical", method="dummy", context=None, options={})
    assert result.coordinates == [1.0]


def test_register_ocr_engine_registers_all_capabilities():
    name = f"ocr.test.{uuid.uuid4().hex}"

    class DummyOCREngine:
        pass

    register_ocr_engine(name, lambda **_: DummyOCREngine())

    provider = get_provider()
    for capability in ("ocr", "ocr.apply", "ocr.extract"):
        engine = provider.get(capability, context=None, name=name)
        assert isinstance(engine, DummyOCREngine)

    assert name in get_registry()


def test_register_ocr_engine_classic_runs_through_unified_dispatch():
    name = f"ocr.classic.{uuid.uuid4().hex}"

    class DummyOCREngine:
        def process_image(self, image, **kwargs):
            return [{"bbox": [0, 0, 10, 10], "text": "hello", "confidence": 0.99}]

    register_ocr_engine(name, lambda **_: DummyOCREngine(), install_hint="pip install dummy")

    class Target:
        def render(self, resolution=72, **kwargs):
            return Image.new("RGB", (20, 20), "white")

    result = run_ocr(target=Target(), engine_name=name, resolution=72)
    assert result.results[0]["text"] == "hello"
    assert result.image_size == (20, 20)


def test_register_ocr_engine_options_class_infers_engine():
    name = f"ocr.options.{uuid.uuid4().hex}"

    class DummyOptions(BaseOCROptions):
        pass

    class DummyOCREngine:
        def process_image(self, image, **kwargs):
            return []

    register_ocr_engine(
        name,
        lambda **_: DummyOCREngine(),
        options_class=DummyOptions,
    )

    assert infer_engine_from_options(DummyOptions()) == name


def test_register_ocr_engine_missing_dependency_mentions_install_hint():
    name = f"ocr.missing.{uuid.uuid4().hex}"

    class MissingOCREngine:
        def is_available(self):
            return False

        def process_image(self, image, **kwargs):  # pragma: no cover - should not run
            return []

    register_ocr_engine(name, lambda **_: MissingOCREngine(), install_hint="pip install missing")

    class Target:
        def render(self, resolution=72, **kwargs):
            return Image.new("RGB", (20, 20), "white")

    with pytest.raises(RuntimeError, match="Install it with: pip install missing"):
        run_ocr(target=Target(), engine_name=name, resolution=72)


def test_register_ocr_engine_vlm_shorthand(monkeypatch):
    name = f"ocr.vlm.{uuid.uuid4().hex}"
    calls = {}

    def fake_run_vlm_ocr_on_image(image, **kwargs):
        calls.update(kwargs)
        return ([{"bbox": [0, 0, 10, 10], "text": "vlm", "confidence": 0.9}], image.size)

    monkeypatch.setattr(
        "natural_pdf.ocr.vlm_ocr.run_vlm_ocr_on_image",
        fake_run_vlm_ocr_on_image,
    )
    register_ocr_engine(
        name,
        kind="vlm",
        model_resolver=lambda: "custom-model",
        vlm_family="glm_ocr",
    )

    class Target:
        def render(self, resolution=72, **kwargs):
            return Image.new("RGB", (20, 20), "white")

    result = run_ocr(target=Target(), engine_name=name, resolution=72)
    assert result.results[0]["text"] == "vlm"
    assert calls["model"] == "custom-model"
    assert calls["family"] == "glm_ocr"


def test_register_layout_engine_round_trip():
    name = f"layout.test.{uuid.uuid4().hex}"

    class DummyLayoutEngine:
        pass

    register_layout_engine(name, lambda **_: DummyLayoutEngine())
    provider = get_provider()
    engine = provider.get("layout", context=None, name=name)
    assert isinstance(engine, DummyLayoutEngine)


def test_register_classification_engine_round_trip():
    name = f"classification.test.{uuid.uuid4().hex}"

    class DummyClassificationEngine:
        def classify_item(self, **kwargs):
            return "ok"

    register_classification_engine(name, lambda **_: DummyClassificationEngine())
    provider = get_provider()
    engine = provider.get("classification", context=None, name=name)
    assert isinstance(engine, DummyClassificationEngine)


def test_register_deskew_engine_registers_all_capabilities():
    name = f"deskew.test.{uuid.uuid4().hex}"

    class DummyDeskew:
        pass

    register_deskew_engine(name, lambda **_: DummyDeskew())

    provider = get_provider()
    for capability in ("deskew", "deskew.detect", "deskew.apply"):
        engine = provider.get(capability, context=None, name=name)
        assert isinstance(engine, DummyDeskew)


def test_register_selector_engine_round_trip():
    name = f"selector.test.{uuid.uuid4().hex}"

    class DummySelectorEngine:
        pass

    register_selector_engine(name, lambda **_: DummySelectorEngine())
    provider = get_provider()
    engine = provider.get("selectors", context=None, name=name)
    assert isinstance(engine, DummySelectorEngine)
