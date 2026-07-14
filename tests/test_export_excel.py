"""Excel export smoke coverage for the declared export extra."""

from __future__ import annotations

from pathlib import Path

import pytest

from natural_pdf.export.mixin import ExportMixin

pytestmark = pytest.mark.optional_deps


class _ExcelExportHost(ExportMixin):
    def _gather_analysis_data(self, **_kwargs):
        return [{"name": "Ada", "score": 1}]


def test_excel_export_works_when_export_dependencies_are_installed(tmp_path: Path) -> None:
    pytest.importorskip("openpyxl")
    output_path = tmp_path / "analysis.xlsx"

    result = _ExcelExportHost().export_analyses(
        output_path,
        analysis_keys="example",
        format="excel",
    )

    assert result == str(output_path)
    assert output_path.is_file()
    assert output_path.stat().st_size > 0
