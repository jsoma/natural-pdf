"""
Simple tests to verify current behavior and test the highlighting protocol implementation.
"""

from pathlib import Path

import pytest

import natural_pdf as npdf
from natural_pdf.elements.element_collection import ElementCollection
from natural_pdf.flows.element import FlowElement
from natural_pdf.flows.flow import Flow
from natural_pdf.flows.region import FlowRegion


def test_current_flow_region_show_fails():
    """Verify that FlowRegions in collections currently fail to show."""
    pdf = npdf.PDF("pdfs/multipage-table-african-recipes.pdf")

    try:
        # Create a simple FlowRegion manually
        flow = Flow(segments=[pdf.pages[0], pdf.pages[1]], arrangement="vertical")

        # Create a FlowRegion that spans pages
        source_elem = pdf.pages[0].find("text")
        flow_elem = FlowElement(physical_object=source_elem, flow=flow)

        # Create regions on both pages
        region1 = pdf.pages[0].region(50, 100, 200, 300)
        region2 = pdf.pages[1].region(50, 50, 200, 250)

        flow_region = FlowRegion(
            flow=flow, constituent_regions=[region1, region2], source_flow_element=flow_elem
        )

        # Create a collection with both regular elements and FlowRegion
        regular_elements = pdf.pages[0].find_all("text")[:3]
        mixed_collection = ElementCollection(list(regular_elements) + [flow_region])

        print(f"\nMixed collection has {len(mixed_collection)} elements")
        print(f"Regular elements: {len(regular_elements)}")
        print(f"FlowRegion type: {type(flow_region)}")
        print(f"FlowRegion has page attr: {hasattr(flow_region, 'page')}")
        if hasattr(flow_region, "page"):
            print(f"FlowRegion.page: {flow_region.page}")

        # This should fail because FlowRegion spans multiple pages
        try:
            img = mixed_collection.show()
            print("ERROR: show() succeeded when it should have failed!")
            print(f"Generated image size: {img.size}")
        except Exception as e:
            print(f"Expected failure: {type(e).__name__}: {e}")
            # Check for the expected error - should be about multiple pages
            error_msg = str(e).lower()
            assert "page" in error_msg or "multiple pages" in error_msg

    finally:
        pdf.close()


def test_current_multipage_elements_show_fails():
    """Verify that elements from multiple pages fail to show."""
    pdf = npdf.PDF("pdfs/multipage-table-african-recipes.pdf")

    try:
        # Get elements from different pages
        page1_text = pdf.pages[0].find_all("text")[:3]
        page2_text = pdf.pages[1].find_all("text")[:3]

        # Combine them
        mixed = ElementCollection(list(page1_text) + list(page2_text))

        print(f"\nMixed collection has {len(mixed)} elements")
        print(f"From page 0: {len(page1_text)} elements")
        print(f"From page 1: {len(page2_text)} elements")

        # This should fail
        try:
            img = mixed.show()
            print("ERROR: show() succeeded when it should have failed!")
        except Exception as e:
            print(f"Expected failure: {type(e).__name__}: {e}")
            # The error message might vary
            error_msg = str(e).lower()
            assert "page" in error_msg or "elements are on the same page" in error_msg

    finally:
        pdf.close()


def test_flow_region_structure():
    """Verify FlowRegion structure for protocol implementation."""
    pdf = npdf.PDF("pdfs/multipage-table-african-recipes.pdf")

    try:
        # Create a simple FlowRegion
        flow = Flow(segments=[pdf.pages[0], pdf.pages[1]], arrangement="vertical")

        source_elem = pdf.pages[0].find("text")
        flow_elem = FlowElement(physical_object=source_elem, flow=flow)

        # Create regions on both pages
        region1 = pdf.pages[0].region(50, 100, 200, 300)
        region2 = pdf.pages[1].region(50, 50, 200, 250)

        flow_region = FlowRegion(
            flow=flow, constituent_regions=[region1, region2], source_flow_element=flow_elem
        )

        print(f"\nFlowRegion has {len(flow_region.constituent_regions)} constituent regions")
        for i, region in enumerate(flow_region.constituent_regions):
            print(
                f"  Region {i}: page={region.page.number if region.page else 'None'}, bbox={region.bbox}"
            )

        # Check what we need for the protocol
        print(f"\nFlowRegion attributes:")
        print(f"  has 'page': {hasattr(flow_region, 'page')}")
        print(f"  has 'constituent_regions': {hasattr(flow_region, 'constituent_regions')}")
        print(f"  has 'bbox': {hasattr(flow_region, 'bbox')}")
        if hasattr(flow_region, "bbox"):
            print(f"  bbox value: {flow_region.bbox}")

    finally:
        pdf.close()


def test_single_page_show_works():
    """Verify that single-page collections still work."""
    pdf = npdf.PDF("pdfs/01-practice.pdf")

    try:
        # Get elements from a single page
        elements = pdf.pages[0].find_all("text")

        print(f"\nSingle page collection has {len(elements)} elements")

        # This should work
        img = elements.show()
        assert img is not None
        print(f"Success: Generated image {img.size}")

    finally:
        pdf.close()


def test_flow_and_flow_region_show_work():
    """Verify that Flow.show() and FlowRegion.show() work correctly."""
    pdf = npdf.PDF("pdfs/multipage-table-african-recipes.pdf")

    try:
        # Test Flow.show()
        flow = Flow(segments=[pdf.pages[0], pdf.pages[1]], arrangement="vertical")

        print("\nTesting Flow.show()...")
        flow_img = flow.show()
        assert flow_img is not None
        print(f"Flow.show() success: Generated image {flow_img.size}")

        # Test FlowRegion.show()
        source_elem = pdf.pages[0].find("text")
        flow_elem = FlowElement(physical_object=source_elem, flow=flow)

        region1 = pdf.pages[0].region(50, 100, 200, 300)
        region2 = pdf.pages[1].region(50, 50, 200, 250)

        flow_region = FlowRegion(
            flow=flow, constituent_regions=[region1, region2], source_flow_element=flow_elem
        )

        print("\nTesting FlowRegion.show()...")
        fr_img = flow_region.show()
        assert fr_img is not None
        print(f"FlowRegion.show() success: Generated image {fr_img.size}")

    finally:
        pdf.close()


if __name__ == "__main__":
    print("Testing current behavior...")

    print("\n1. Testing FlowRegion in collection:")
    test_current_flow_region_show_fails()

    print("\n2. Testing multi-page elements:")
    test_current_multipage_elements_show_fails()

    print("\n3. Testing FlowRegion structure:")
    test_flow_region_structure()

    print("\n4. Testing single page (should work):")
    test_single_page_show_works()

    print("\n5. Testing Flow and FlowRegion show (should work):")
    test_flow_and_flow_region_show_work()

    print("\nAll tests completed!")
