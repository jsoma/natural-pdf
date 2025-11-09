import uuid

import pytest

from natural_pdf.engine_provider import get_provider
from natural_pdf.engine_registry import (
    register_classification_engine,
    register_deskew_engine,
    register_guides_engine,
    register_layout_engine,
    register_ocr_engine,
    register_qa_engine,
    register_selector_engine,
    register_table_function,
)
from natural_pdf.guides.guides_provider import GuidesDetectionResult
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


def test_register_qa_engine_round_trip():
    name = f"qa.test.{uuid.uuid4().hex}"

    class DummyQAEngine:
        def ask_region(self, **kwargs):
            return "answer"

    register_qa_engine(name, lambda **_: DummyQAEngine())
    provider = get_provider()
    engine = provider.get("qa.document", context=None, name=name)
    assert isinstance(engine, DummyQAEngine)


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
