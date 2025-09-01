"""Test auto_multipage global option."""

import pytest

import natural_pdf as npdf
from natural_pdf import PDF
from natural_pdf.flows.region import FlowRegion


def test_auto_multipage_option():
    """Test that the auto_multipage option works as default."""
    pdf = PDF("pdfs/sections.pdf")

    # Save current setting
    original_setting = npdf.options.layout.auto_multipage

    try:
        # Default should be False
        assert npdf.options.layout.auto_multipage == False

        # Find Section 1 on page 1
        section1 = pdf.pages[0].find("text:contains(Section 1)")

        # Without auto_multipage, should not cross pages
        result = section1.below(until="text:contains(Section 6)")
        assert result.page.number == 1  # Stays on page 1

        # Enable auto_multipage
        npdf.set_option("layout.auto_multipage", True)
        assert npdf.options.layout.auto_multipage == True

        # Now it should cross pages automatically
        result = section1.below(until="text:contains(Section 6)")
        assert isinstance(result, FlowRegion)
        assert "Section 6" in result.extract_text()

        # Explicit multipage=False should override global setting
        result = section1.below(until="text:contains(Section 6)", multipage=False)
        assert result.page.number == 1  # Stays on page 1

    finally:
        # Restore original setting
        npdf.options.layout.auto_multipage = original_setting


def test_set_option_errors():
    """Test that set_option raises appropriate errors."""
    with pytest.raises(KeyError, match="Unknown option section"):
        npdf.set_option("invalid_section.option", True)

    with pytest.raises(KeyError, match="Unknown option"):
        npdf.set_option("layout.invalid_option", True)


# Removed test_pdf_level_multipage_option as PDF-level auto_multipage parameter
# was not part of the original design. Users can use the global option instead:
# npdf.set_option('layout.auto_multipage', True)


if __name__ == "__main__":
    test_auto_multipage_option()
    test_set_option_errors()
    print("All tests passed!")
