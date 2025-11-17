import pytest

import natural_pdf as npdf
from natural_pdf.tables.structure_provider import StructureDetectionResult


class FakeCell:
    def __init__(self, x0, top, x1, bottom, value):
        self.x0 = x0
        self.x1 = x1
        self.top = top
        self.bottom = bottom
        self.width = x1 - x0
        self.height = bottom - top
        self.metadata = {}
        self._value = value

    def extract_text(self, layout=False, apply_exclusions=True, content_filter=None):
        return self._value


@pytest.mark.parametrize("structure_engine", [None, "stub"])
def test_region_extract_table_uses_structure_engine(monkeypatch, structure_engine):
    pdf = npdf.PDF("pdfs/01-practice.pdf")
    region = pdf.pages[0].to_region()
    region.model = "tatr"

    cells = [
        FakeCell(0, 0, 50, 10, "A"),
        FakeCell(50, 0, 100, 10, "B"),
        FakeCell(0, 10, 50, 20, "C"),
        FakeCell(50, 10, 100, 20, "D"),
    ]
    structure_result = StructureDetectionResult(capabilities={"cells"}, cells=cells)

    monkeypatch.setattr(
        "natural_pdf.services.table_service.resolve_table_structure_engine_name",
        lambda ctx, requested=None, scope="region": structure_engine or "tatr",
    )
    monkeypatch.setattr(
        "natural_pdf.services.table_service.run_table_structure_engine",
        lambda **kwargs: structure_result,
    )
    monkeypatch.setattr(
        "natural_pdf.services.table_service.run_table_engine",
        lambda **_: (_ for _ in ()).throw(AssertionError("table engine should not run")),
    )

    # Ensure there are no cached table cells so the structure engine path is used
    monkeypatch.setattr(region.page, "find_all", lambda *args, **kwargs: [])

    result = region.extract_table(structure_engine=structure_engine)
    assert list(result) == [["A", "B"], ["C", "D"]]
    pdf.close()


def test_region_extract_table_strict_structure_failure(monkeypatch):
    pdf = npdf.PDF("pdfs/01-practice.pdf")
    region = pdf.pages[0].to_region()

    # Ensure no cached cells divert the flow
    monkeypatch.setattr(region.page, "find_all", lambda *args, **kwargs: [])

    monkeypatch.setattr(
        "natural_pdf.services.table_service.resolve_table_structure_engine_name",
        lambda ctx, requested=None, scope="region": requested,
    )
    monkeypatch.setattr(
        "natural_pdf.services.table_service.run_table_structure_engine",
        lambda **kwargs: None,
    )

    with pytest.raises(ValueError, match="Structure engine 'stub' returned no structure"):
        region.extract_table(structure_engine="stub")

    pdf.close()
