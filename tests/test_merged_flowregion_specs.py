"""Test that overlapping FlowRegions are properly merged in render specs."""

import pytest

from natural_pdf import PDF
from natural_pdf.elements.element_collection import ElementCollection
from natural_pdf.elements.region import Region
from natural_pdf.flows.flow import Flow
from natural_pdf.flows.region import FlowRegion


def test_overlapping_flowregions_merge_specs():
    """Test that multiple FlowRegions on the same page merge their specs."""
    pdf = PDF("pdfs/sections.pdf")

    # Create a flow
    flow = Flow(segments=[pdf.pages[0], pdf.pages[1]], arrangement="vertical")

    # Create FlowRegion 1: spans pages 1-2
    flowregion1 = FlowRegion(
        flow=flow,
        constituent_regions=[
            Region(pdf.pages[0], (0, 600, 612, 792)),  # Bottom of page 1
            Region(pdf.pages[1], (0, 0, 612, 200)),  # Top of page 2
        ],
        source_flow_element=None,
        boundary_element_found=None,
    )

    # Create FlowRegion 2: only on page 2
    flowregion2 = FlowRegion(
        flow=flow,
        constituent_regions=[Region(pdf.pages[1], (0, 150, 612, 400))],  # Middle of page 2
        source_flow_element=None,
        boundary_element_found=None,
    )

    # Create collection with overlapping FlowRegions
    collection = ElementCollection([flowregion1, flowregion2])

    # Get render specs
    specs = collection._get_render_specs(mode="show")

    # Should have exactly 2 specs (one for each page)
    assert len(specs) == 2, f"Expected 2 specs, got {len(specs)}"

    # Find the spec for page 2
    page2_spec = None
    for spec in specs:
        if spec.page.number == 2:
            page2_spec = spec
            break

    assert page2_spec is not None, "No spec found for page 2"

    # Page 2 should have 2 highlights (one from each FlowRegion)
    assert (
        len(page2_spec.highlights) == 2
    ), f"Expected 2 highlights on page 2, got {len(page2_spec.highlights)}"


def test_mixed_flowregions_and_regions():
    """Test that FlowRegions and regular Regions combine properly."""
    pdf = PDF("pdfs/sections.pdf")

    # Create a flow
    flow = Flow(segments=[pdf.pages[0], pdf.pages[1]], arrangement="vertical")

    # Create FlowRegion that spans pages
    flowregion = FlowRegion(
        flow=flow,
        constituent_regions=[
            Region(pdf.pages[0], (0, 700, 612, 792)),
            Region(pdf.pages[1], (0, 0, 612, 100)),
        ],
        source_flow_element=None,
        boundary_element_found=None,
    )

    # Create regular Region on page 2
    regular_region = Region(pdf.pages[1], (0, 200, 612, 300))

    # Create collection
    collection = ElementCollection([flowregion, regular_region])

    # Get render specs
    specs = collection._get_render_specs(mode="show")

    # Should have 2 specs (one for each page)
    assert len(specs) == 2, f"Expected 2 specs, got {len(specs)}"

    # Find page 2 spec
    page2_spec = None
    for spec in specs:
        if spec.page.number == 2:
            page2_spec = spec
            break

    assert page2_spec is not None, "No spec found for page 2"

    # Page 2 should have 2 highlights (FlowRegion part + regular Region)
    assert (
        len(page2_spec.highlights) == 2
    ), f"Expected 2 highlights on page 2, got {len(page2_spec.highlights)}"


def test_crop_bbox_merging():
    """Test that crop bboxes are properly merged when FlowRegions overlap."""
    pdf = PDF("pdfs/sections.pdf")

    # Create a flow
    flow = Flow(segments=[pdf.pages[0], pdf.pages[1]], arrangement="vertical")

    # Create FlowRegions with different regions on the same page
    flowregion1 = FlowRegion(
        flow=flow,
        constituent_regions=[Region(pdf.pages[0], (100, 100, 200, 200))],  # Small region
        source_flow_element=None,
        boundary_element_found=None,
    )

    flowregion2 = FlowRegion(
        flow=flow,
        constituent_regions=[Region(pdf.pages[0], (300, 300, 400, 400))],  # Different region
        source_flow_element=None,
        boundary_element_found=None,
    )

    # Create collection
    collection = ElementCollection([flowregion1, flowregion2])

    # Get render specs with cropping
    specs = collection._get_render_specs(mode="show", crop=True)

    # Should have 1 spec for page 1
    assert len(specs) == 1
    spec = specs[0]

    # Crop bbox should encompass both regions
    assert spec.crop_bbox is not None
    x0, y0, x1, y1 = spec.crop_bbox

    # Should include both regions (100-200 and 300-400)
    assert x0 <= 100 and x1 >= 400
    assert y0 <= 100 and y1 >= 400


if __name__ == "__main__":
    test_overlapping_flowregions_merge_specs()
    test_mixed_flowregions_and_regions()
    test_crop_bbox_merging()
    print("All tests passed!")
