"""Test that ElementCollection properly renders mixed Region/FlowRegion collections."""

import pytest

from natural_pdf import PDF
from natural_pdf.elements.element_collection import ElementCollection
from natural_pdf.flows.region import FlowRegion


def test_mixed_region_flowregion_rendering():
    """Test that collections with both Region and FlowRegion render correctly."""
    pdf = PDF("pdfs/sections.pdf")
    flow = pdf.pages.to_flow()

    # Get sections - this creates a mix of Region and FlowRegion objects
    sections = flow.get_sections("text:contains(Section)", include_boundaries="both")

    # Verify we have both types
    has_region = False
    has_flowregion = False

    for section in sections:
        if isinstance(section, FlowRegion):
            has_flowregion = True
        else:
            has_region = True

    assert has_region, "Expected at least one regular Region"
    assert has_flowregion, "Expected at least one FlowRegion"

    # Test that _get_render_specs works correctly
    specs = sections._get_render_specs(mode="show")

    # Should have exactly 2 specs (one per page, even with overlapping FlowRegions)
    assert len(specs) == 2, f"Expected 2 specs total, got {len(specs)}"

    pages_in_specs = set(spec.page for spec in specs)
    assert len(pages_in_specs) == 2, f"Expected specs for 2 pages, got {len(pages_in_specs)}"

    # FlowRegions should create proper highlights across pages
    flowregion_found = False
    for section in sections:
        if isinstance(section, FlowRegion):
            flowregion_found = True
            # FlowRegion should have constituent regions
            assert len(section.constituent_regions) > 0

            # Get specs for this FlowRegion
            flowregion_specs = section._get_render_specs(mode="show")

            # Should have specs for each page it spans
            assert len(flowregion_specs) == len(set(r.page for r in section.constituent_regions))

    assert flowregion_found, "No FlowRegion found in sections"


def test_elementcollection_handles_flowregions():
    """Test that ElementCollection delegates to FlowRegion's own render specs."""
    pdf = PDF("pdfs/sections.pdf")
    flow = pdf.pages.to_flow()

    # Create a FlowRegion that spans pages
    sections = flow.get_sections("text:contains(Section)", include_boundaries="both")

    # Find a FlowRegion
    flow_region = None
    for s in sections:
        if isinstance(s, FlowRegion):
            flow_region = s
            break

    assert flow_region is not None, "No FlowRegion found"

    # Create collection with just the FlowRegion
    collection = ElementCollection([flow_region])

    # Get render specs
    specs = collection._get_render_specs(mode="show")

    # Should have specs for each page the FlowRegion spans
    flow_region_pages = set(r.page for r in flow_region.constituent_regions)
    spec_pages = set(spec.page for spec in specs)

    assert (
        spec_pages == flow_region_pages
    ), f"Spec pages {spec_pages} don't match FlowRegion pages {flow_region_pages}"

    # Each spec should have highlights for the FlowRegion parts
    for spec in specs:
        assert len(spec.highlights) > 0, f"No highlights on page {spec.page.number}"


if __name__ == "__main__":
    test_mixed_region_flowregion_rendering()
    test_elementcollection_handles_flowregions()
    print("All tests passed!")
