import math

import pytest

from natural_pdf import PDF
from natural_pdf.flows import Flow


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
