import natural_pdf.engine_provider as provider_module
from natural_pdf.engine_provider import EngineProvider
from natural_pdf.tables.structure_engines.tatr import TATRStructureEngine
from natural_pdf.tables.structure_provider import (
    StructureDetectionResult,
    resolve_structure_engine_name,
    run_table_structure_engine,
)


class _FakeElement:
    def __init__(self, x0, top, x1, bottom):
        self.x0 = x0
        self.x1 = x1
        self.top = top
        self.bottom = bottom
        self.width = x1 - x0
        self.height = bottom - top
        self.metadata = {}


class _FakeRegion:
    def __init__(self):
        self.x0 = 0
        self.y0 = 0
        self.x1 = 100
        self.y1 = 100
        self.top = 0
        self.bottom = 100
        self.page = self
        self._data = {
            "region[type=table-row][model=tatr]": [
                _FakeElement(0, 0, 100, 10),
                _FakeElement(0, 10, 100, 20),
            ],
            "region[type=table-column][model=tatr]": [
                _FakeElement(0, 0, 50, 20),
                _FakeElement(50, 0, 100, 20),
            ],
            "region[type=table-column-header][model=tatr]": [_FakeElement(0, 0, 100, 5)],
            "region[type=table_cell][model=tatr]": [
                _FakeElement(0, 0, 50, 10),
                _FakeElement(50, 0, 100, 10),
            ],
        }

    @property
    def bbox(self):
        return (self.x0, self.top, self.x1, self.bottom)

    def find_all(self, selector, apply_exclusions=False):
        return list(self._data.get(selector, []))


def test_tatr_structure_engine_returns_capabilities():
    region = _FakeRegion()
    engine = TATRStructureEngine()
    result = engine.detect(context=region, region=region)

    assert result.capabilities == {"rows", "columns", "headers", "cells"}
    assert len(result.cells) == 2


def test_resolve_structure_engine_name_prefers_region_model():
    class Dummy:
        model = "tatr"

    assert resolve_structure_engine_name(Dummy()) == "tatr"


def test_run_structure_engine_with_custom_provider(monkeypatch):
    provider = EngineProvider()
    provider._entry_points_loaded = True
    monkeypatch.setattr(provider_module, "_PROVIDER", provider)

    class _StubStructureEngine:
        def detect(self, *, context, region, options=None):
            return StructureDetectionResult(
                capabilities={"cells"}, cells=[_FakeElement(0, 0, 1, 1)]
            )

    provider.register(
        "tables.detect_structure",
        "stub",
        lambda **_: _StubStructureEngine(),
        replace=True,
    )

    fake_region = _FakeRegion()
    result = run_table_structure_engine(context=fake_region, region=fake_region, engine_name="stub")
    assert result is not None
    assert "cells" in result.capabilities
