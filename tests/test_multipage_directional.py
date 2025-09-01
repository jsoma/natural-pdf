"""Test multipage parameter for directional methods (.below(), .above(), etc.)"""

import pytest

from natural_pdf import PDF
from natural_pdf.elements.region import Region
from natural_pdf.flows.region import FlowRegion


def test_below_multipage_basic():
    """Test basic multipage behavior for .below()"""
    pdf = PDF("pdfs/sections.pdf")

    # Find Section 1 on page 1
    section1 = pdf.pages[0].find("text:contains(Section 1)")
    assert section1 is not None

    # Default behavior - should stop at page boundary
    result = section1.below()
    assert isinstance(result, Region)
    assert result.page.number == 1

    # With multipage=True - should continue to next page
    result = section1.below(multipage=True)
    assert isinstance(result, FlowRegion)
    assert len(result.constituent_regions) == 2  # Spans 2 pages


def test_below_multipage_with_until():
    """Test multipage with 'until' parameter"""
    pdf = PDF("pdfs/sections.pdf")

    # Find Section 1 on page 1
    section1 = pdf.pages[0].find("text:contains(Section 1)")

    # Try to reach Section 6 on page 2 - should fail without multipage
    result = section1.below(until="text:contains(Section 6)")
    # Should get the region up to bottom of page 1 (no Section 6 found)
    assert isinstance(result, Region)
    assert result.page.number == 1

    # With multipage=True - should reach Section 6 on page 2
    result = section1.below(until="text:contains(Section 6)", multipage=True)
    assert isinstance(result, FlowRegion)
    # Should find Section 6 and stop there
    assert len(result.constituent_regions) == 2  # Spans pages 1 and 2

    # Verify Section 6 is in the result
    text = result.extract_text()
    assert "Section 6" in text


def test_below_multipage_include_endpoint():
    """Test multipage with include_endpoint parameter"""
    pdf = PDF("pdfs/sections.pdf")

    section1 = pdf.pages[0].find("text:contains(Section 1)")

    # With include_endpoint=False
    result = section1.below(
        until="text:contains(Section 6)", include_endpoint=False, multipage=True
    )
    assert isinstance(result, FlowRegion)
    text = result.extract_text()
    assert "Section 5" in text
    assert "Section 6" not in text  # Should not include endpoint

    # With include_endpoint=True (default)
    result = section1.below(until="text:contains(Section 6)", multipage=True)
    text = result.extract_text()
    assert "Section 6" in text  # Should include endpoint


def test_above_multipage():
    """Test multipage behavior for .above()"""
    pdf = PDF("pdfs/sections.pdf")

    # Find Section 6 on page 2
    section6 = pdf.pages[1].find("text:contains(Section 6)")
    assert section6 is not None

    # Default behavior - should stop at page boundary
    result = section6.above()
    assert isinstance(result, Region)
    assert result.page.number == 2

    # With multipage=True - should continue to previous page
    result = section6.above(multipage=True)
    assert isinstance(result, FlowRegion)
    assert len(result.constituent_regions) == 2  # Spans 2 pages


def test_multipage_stays_on_single_page():
    """Test that multipage=True returns Region when result is on single page"""
    pdf = PDF("pdfs/sections.pdf")

    # Find Section 2 and Section 3 on same page
    section2 = pdf.pages[0].find("text:contains(Section 2)")

    # Even with multipage=True, should return Region if doesn't cross pages
    result = section2.below(until="text:contains(Section 3)", multipage=True)
    assert isinstance(result, Region), "Should return Region when staying on single page"
    assert result.page.number == 1


def test_multipage_with_element_collection():
    """Test multipage with ElementCollection (find_all)"""
    pdf = PDF("pdfs/sections.pdf")

    # Find all "Section" texts
    sections = pdf.pages[0].find_all("text:contains(Section)")
    assert len(sections) > 0

    # Get first section
    first_section = sections[0]

    # Should work with elements from collection
    result = first_section.below(until="text:contains(Section 6)", multipage=True)
    assert isinstance(result, FlowRegion)
    assert "Section 6" in result.extract_text()


def test_multipage_parameter_validation():
    """Test that multipage parameter only accepts boolean values"""
    pdf = PDF("pdfs/sections.pdf")
    section1 = pdf.pages[0].find("text:contains(Section 1)")

    # Should work with True/False
    result = section1.below(multipage=True)
    assert result is not None

    result = section1.below(multipage=False)
    assert result is not None

    # Should handle other truthy/falsy values
    result = section1.below(multipage=1)  # Truthy
    assert isinstance(result, FlowRegion)

    result = section1.below(multipage=0)  # Falsy
    assert isinstance(result, Region)


def test_left_right_multipage():
    """Test multipage for horizontal methods (left/right)"""
    # Note: This might be less common but should still work
    # Implementation depends on how horizontal flow works
    pass  # TODO: Implement if horizontal multipage is supported


if __name__ == "__main__":
    # Run specific test for development
    test_below_multipage_basic()
    test_below_multipage_with_until()
    test_below_multipage_include_endpoint()
    test_above_multipage()
    test_multipage_stays_on_single_page()
    test_multipage_with_element_collection()
    test_multipage_parameter_validation()
    print("All tests defined!")
