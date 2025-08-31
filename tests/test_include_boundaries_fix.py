"""Test include_boundaries parameter in get_sections across all implementations."""

import pytest

from natural_pdf import PDF


def test_include_boundaries_parameter():
    """Test that include_boundaries parameter works correctly for all implementations."""
    pdf = PDF("pdfs/sections.pdf")
    page = pdf.pages[0]
    region = page.region()
    flow = pdf.pages.to_flow()

    # Expected outputs for each boundary mode
    expected = {
        "both": "Section 1\nABC\nSection 2",  # Include both section headers
        "none": "ABC",  # Exclude both section headers
        "start": "Section 1\nABC",  # Include only starting header
        "end": "ABC\nSection 2",  # Include only ending header
    }

    # Test all implementations with all boundary modes
    implementations = [
        ("pdf", lambda: pdf),
        ("page", lambda: page),
        ("region", lambda: region),
        ("flow", lambda: flow),
    ]

    for impl_name, impl_getter in implementations:
        for boundary_mode, expected_text in expected.items():
            impl = impl_getter()
            sections = impl.get_sections("text:contains(Section)", include_boundaries=boundary_mode)

            # Check we got sections
            assert (
                len(sections) > 0
            ), f"{impl_name}.get_sections with boundaries={boundary_mode} returned no sections"

            # For all implementations, we just test the first section
            # Flow now properly returns all sections like PDF and PageCollection
            actual_text = sections[0].extract_text().strip()

            # Normalize whitespace for comparison
            actual_text = "\n".join(
                line.strip() for line in actual_text.split("\n") if line.strip()
            )
            expected_normalized = "\n".join(
                line.strip() for line in expected_text.split("\n") if line.strip()
            )

            assert actual_text == expected_normalized, (
                f"{impl_name}.get_sections with boundaries={boundary_mode} failed:\n"
                f"Expected: {repr(expected_normalized)}\n"
                f"Actual: {repr(actual_text)}"
            )


def test_get_sections_with_only_start_elements():
    """Test get_sections behavior when only start elements are provided."""
    pdf = PDF("pdfs/sections.pdf")
    page = pdf.pages[0]

    # When only start elements are provided, sections should go from one start to the next
    sections = page.get_sections("text:contains(Section)")

    # Page 1 has 4 sections: Section 1, 2, 3, and 4
    assert len(sections) == 4, f"Expected 4 sections on page 1, got {len(sections)}"

    # Test with different boundary modes
    for boundary_mode in ["both", "none", "start", "end"]:
        sections = page.get_sections("text:contains(Section)", include_boundaries=boundary_mode)
        assert (
            len(sections) == 4
        ), f"Expected 4 sections with boundaries={boundary_mode}, got {len(sections)}"

        # First section content varies by boundary mode
        first_section = sections[0].extract_text().strip()

        if boundary_mode == "both":
            # Should include "Section 1", "ABC", and "Section 2"
            assert "Section 1" in first_section
            assert "ABC" in first_section
            assert "Section 2" in first_section
        elif boundary_mode == "none":
            # Should only have "ABC" (content between sections)
            assert "Section 1" not in first_section
            assert "ABC" in first_section
            assert "Section 2" not in first_section
        elif boundary_mode == "start":
            # Should include "Section 1" and "ABC" but not "Section 2"
            assert "Section 1" in first_section
            assert "ABC" in first_section
            assert "Section 2" not in first_section
        elif boundary_mode == "end":
            # Should include "ABC" and "Section 2" but not "Section 1"
            assert "Section 1" not in first_section
            assert "ABC" in first_section
            assert "Section 2" in first_section


def test_visual_verification():
    """Generate visual output to verify sections are extracted correctly."""
    pdf = PDF("pdfs/sections.pdf")
    page = pdf.pages[0]

    print("\n=== Visual Verification of Section Extraction ===")

    for boundary_mode in ["both", "none", "start", "end"]:
        print(f"\n--- Boundary mode: {boundary_mode} ---")
        sections = page.get_sections("text:contains(Section)", include_boundaries=boundary_mode)

        for i, section in enumerate(sections):
            print(f"Section {i}: {repr(section.extract_text().strip())}")
            # Could also save visual representation
            # section.save(f"temp/section_{boundary_mode}_{i}.png")


if __name__ == "__main__":
    # Run visual test for debugging
    test_visual_verification()

    # Run main tests
    test_include_boundaries_parameter()
    test_get_sections_with_only_start_elements()
    print("\nAll tests completed!")
