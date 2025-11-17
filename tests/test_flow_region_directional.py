import math

import pytest

from natural_pdf import PDF
from natural_pdf.flows import Flow
from natural_pdf.flows.collections import FlowRegionCollection
from natural_pdf.qa.qa_result import QAResult


@pytest.mark.parametrize("pdf_path", ["pdfs/multicolumn.pdf"])
def test_flow_region_directional_methods(pdf_path):
    """Validate .above(), .below() and .expand() on FlowRegion with multi-column flow."""

    pdf = PDF(pdf_path)
    page = pdf.pages[0]

    # Split page into three equal-width column regions
    col_width = page.width / 3
    columns = [page.region(left=i * col_width, width=col_width) for i in range(3)]

    # Construct a vertical flow over the columns (reading order: col0 → col1 → col2)
    flow = Flow(columns, arrangement="vertical")

    # Pick the section between the first two bold headings
    region = flow.find("text:bold").below(until="text:bold")

    # Helper to round bboxes for comparison
    def round_bbox(bbox):
        return tuple(round(v, 1) if isinstance(v, float) else v for v in bbox)

    def get_bboxes(fr):
        return [round_bbox(r.bbox) for r in fr.constituent_regions]

    # Expected reference bboxes (manually measured once, allow tolerance)
    expected_region = [
        (0.0, 287.3, 204.0, 792.0),
        (204.0, 0.0, 408.0, 334.1),
    ]
    expected_above = [(0.0, 0.0, 204.0, 286.3)]
    expected_below = [(204.0, 335.1, 408.0, 792.0)]
    expected_expanded = [
        (0.0, 187.3, 204.0, 792.0),
        (204.0, 0.0, 408.0, 334.1),
    ]

    # Compare helpers ---------------------------------------------------
    def assert_bboxes_close(result, expected, tol=1.5):
        assert len(result) == len(expected), f"Expected {len(expected)} boxes, got {len(result)}"
        for got, exp in zip(result, expected):
            assert all(
                math.isclose(g, e, abs_tol=tol) for g, e in zip(got, exp)
            ), f"BBox {got} differs from expected {exp} (tol={tol})"

    # Assertions --------------------------------------------------------
    assert_bboxes_close(get_bboxes(region), expected_region)
    assert_bboxes_close(get_bboxes(region.above()), expected_above)
    assert_bboxes_close(get_bboxes(region.below()), expected_below)
    assert_bboxes_close(get_bboxes(region.expand(top=100)), expected_expanded)


def test_flow_region_collection_directional_methods():
    pdf = PDF("pdfs/multicolumn.pdf")
    page = pdf.pages[0]

    col_width = page.width / 3
    columns = [page.region(left=i * col_width, width=col_width) for i in range(3)]
    flow = Flow(columns, arrangement="vertical")

    region = flow.find("text:bold").below(until="text:bold")
    collection = FlowRegionCollection([region])

    def collect_bboxes(flow_region):
        return [tuple(round(v, 2) for v in reg.bbox) for reg in flow_region.constituent_regions]

    expected_below = region.below(height=120.0, include_source=False)
    result_below = collection.below(height=120.0, include_source=False)
    assert len(result_below.flow_regions) == 1
    assert collect_bboxes(result_below.first) == collect_bboxes(expected_below)

    expected_above = region.above(height=150.0, include_source=False)
    result_above = collection.above(height=150.0, include_source=False)
    assert len(result_above.flow_regions) == 1
    assert collect_bboxes(result_above.first) == collect_bboxes(expected_above)


def test_flow_region_collection_within_requires_region():
    pdf = PDF("pdfs/multicolumn.pdf")
    flow = Flow([pdf.pages[0]], arrangement="vertical")
    region = flow.find("text:bold").below(until="text:bold")
    collection = FlowRegionCollection([region])

    with pytest.raises(TypeError, match="expects a Region"):
        collection.below(within=region)


def test_flow_region_collection_qa(monkeypatch):
    pdf = PDF("pdfs/multicolumn.pdf")
    try:
        page = pdf.pages[0]
        col_width = page.width / 3
        columns = [page.region(left=i * col_width, width=col_width) for i in range(3)]
        flow = Flow(columns, arrangement="vertical")

        anchor = flow.find("text:bold")
        assert anchor is not None
        primary_region = anchor.below(until="text:bold")
        secondary_region = primary_region.below(height=120.0, include_source=False)
        collection = FlowRegionCollection([primary_region, secondary_region])

        for region in primary_region.constituent_regions:
            setattr(region, "_qa_test_tag", "first")
        for region in secondary_region.constituent_regions:
            setattr(region, "_qa_test_tag", "second")

        def fake_run_document_qa(*, region, question, **kwargs):
            tag = getattr(region, "_qa_test_tag", "first")
            confidence = 0.25 if tag == "first" else 0.9
            return QAResult(
                question=question,
                answer=f"{tag}-answer",
                confidence=confidence,
                found=True,
            )

        monkeypatch.setattr(
            "natural_pdf.services.qa_service.run_document_qa",
            fake_run_document_qa,
        )

        result = collection.ask("Which region?", min_confidence=0.0)
        assert isinstance(result, dict)
        assert result["answer"] == "second-answer"
        assert pytest.approx(result["confidence"], rel=0.01) == 0.9
    finally:
        pdf.close()
