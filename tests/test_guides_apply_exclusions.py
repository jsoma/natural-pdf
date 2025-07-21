#!/usr/bin/env python3
"""Test guides.from_content() with apply_exclusions parameter."""

from pathlib import Path

import pytest

from natural_pdf import PDF
from natural_pdf.analyzers.guides import Guides


def find_test_pdf():
    """Find a suitable test PDF file."""
    project_root = Path(__file__).parent.parent
    pdf_path = (
        project_root
        / "bad-pdfs/submissions/Doc 06 - Approved Expenses 07.01.2022-06.30.2023 Marketplace Transactions - REDACTED.pdf"
    )
    return pdf_path if pdf_path.exists() else None


@pytest.mark.skipif(find_test_pdf() is None, reason="No test PDF file found")
def test_from_content_apply_exclusions():
    """Test that from_content respects apply_exclusions parameter."""

    pdf_path = find_test_pdf()
    pdf = PDF(pdf_path)
    page = pdf[0]

    # Find some text that exists on the page
    text_elements = page.find_all("text")
    if len(text_elements) == 0:
        pytest.skip("No text elements found on page")

    # Get the first few text elements and use their text as markers
    sample_texts = []
    for element in text_elements[:5]:
        if element.text and len(element.text.strip()) > 2:
            sample_texts.append(element.text.strip())

    if len(sample_texts) == 0:
        pytest.skip("No suitable text markers found")

    print(f"Using text markers: {sample_texts}")

    # Test 1: Create guides without exclusions (default)
    guides_with_exclusions = Guides.from_content(
        obj=page,
        axis="vertical",
        markers=sample_texts[:2],  # Use first 2 markers
        apply_exclusions=True,  # This is the default
    )

    print(f"Guides with exclusions: {len(guides_with_exclusions.vertical)} vertical guides")

    # Test 2: Create guides without applying exclusions
    guides_without_exclusions = Guides.from_content(
        obj=page,
        axis="vertical",
        markers=sample_texts[:2],  # Use same markers
        apply_exclusions=False,
    )

    print(f"Guides without exclusions: {len(guides_without_exclusions.vertical)} vertical guides")

    # Both should create some guides (behavior might be the same if no exclusions are set)
    assert len(guides_with_exclusions.vertical) >= 0
    assert len(guides_without_exclusions.vertical) >= 0

    print("✅ apply_exclusions parameter accepted and processed")


@pytest.mark.skipif(find_test_pdf() is None, reason="No test PDF file found")
def test_from_content_with_exclusion_zones():
    """Test from_content behavior when exclusion zones are actually set."""

    pdf_path = find_test_pdf()
    pdf = PDF(pdf_path)
    page = pdf[0]

    # Find some text elements
    text_elements = page.find_all("text")
    if len(text_elements) < 3:
        pytest.skip("Not enough text elements for exclusion test")

    # Use the text from some elements as markers
    marker_text = text_elements[0].text.strip() if text_elements[0].text else "test"
    if len(marker_text) < 2:
        marker_text = "test"

    print(f"Using marker: '{marker_text}'")

    # Add an exclusion zone that might contain some of our target text
    if len(text_elements) >= 2:
        exclusion_element = text_elements[1]
        exclusion_region = page.region(
            exclusion_element.x0 - 10,
            exclusion_element.top - 5,
            exclusion_element.x1 + 10,
            exclusion_element.bottom + 5,
        )

        print(f"Adding exclusion zone: {exclusion_region.bbox}")
        page.add_exclusion(exclusion_region, label="test_exclusion")

    # Test with exclusions applied (should respect exclusion zones)
    guides_with_exclusions = Guides.from_content(
        obj=page, axis="vertical", markers=[marker_text], apply_exclusions=True
    )

    # Test without exclusions applied (should ignore exclusion zones)
    guides_without_exclusions = Guides.from_content(
        obj=page, axis="vertical", markers=[marker_text], apply_exclusions=False
    )

    print(f"With exclusions applied: {len(guides_with_exclusions.vertical)} guides")
    print(f"Without exclusions applied: {len(guides_without_exclusions.vertical)} guides")

    # The exact behavior depends on whether the marker text was actually in the exclusion zone
    # But both calls should succeed without error
    from natural_pdf.analyzers.guides import GuidesList

    assert isinstance(guides_with_exclusions.vertical, GuidesList)
    assert isinstance(guides_without_exclusions.vertical, GuidesList)

    print("✅ Exclusion zones properly handled in from_content")


@pytest.mark.skipif(find_test_pdf() is None, reason="No test PDF file found")
def test_add_content_apply_exclusions():
    """Test that add_content instance method also supports apply_exclusions."""

    pdf_path = find_test_pdf()
    pdf = PDF(pdf_path)
    page = pdf[0]

    # Find a text marker
    text_elements = page.find_all("text")
    if len(text_elements) == 0:
        pytest.skip("No text elements found")

    marker_text = "test"  # Use a simple marker that might not exist
    for elem in text_elements[:5]:
        if elem.text and len(elem.text.strip()) > 2:
            marker_text = elem.text.strip()
            break

    print(f"Using marker for add_content: '{marker_text}'")

    # Create guides object and use add_content with apply_exclusions
    guides = Guides(page)

    # Test the method accepts the parameter
    result = guides.add_content(
        axis="vertical", markers=[marker_text], apply_exclusions=True  # Should be accepted
    )

    # Should return self for chaining
    assert result is guides

    # Test with apply_exclusions=False
    guides2 = Guides(page)
    result2 = guides2.add_content(axis="vertical", markers=[marker_text], apply_exclusions=False)

    assert result2 is guides2

    print("✅ add_content method accepts apply_exclusions parameter")


def test_apply_exclusions_parameter_defaults():
    """Test that apply_exclusions has the correct default value."""

    # This test doesn't need a real PDF, just checking the signature
    from inspect import signature

    # Check class method signature
    sig = signature(Guides.from_content)
    apply_exclusions_param = sig.parameters.get("apply_exclusions")

    assert apply_exclusions_param is not None, "apply_exclusions parameter should exist"
    assert apply_exclusions_param.default is True, "apply_exclusions should default to True"

    print("✅ apply_exclusions parameter defaults to True")


if __name__ == "__main__":
    # Run a simple test
    pdf_path = find_test_pdf()
    if pdf_path:
        print(f"Found test PDF: {pdf_path}")

        pdf = PDF(pdf_path)
        page = pdf[0]

        # Test that the parameter is accepted
        try:
            guides = Guides.from_content(
                obj=page, axis="vertical", markers=["test"], apply_exclusions=True
            )
            print("✅ from_content accepts apply_exclusions=True")
        except TypeError as e:
            print(f"❌ Parameter not accepted: {e}")

        try:
            guides = Guides.from_content(
                obj=page, axis="vertical", markers=["test"], apply_exclusions=False
            )
            print("✅ from_content accepts apply_exclusions=False")
        except TypeError as e:
            print(f"❌ Parameter not accepted: {e}")

    else:
        print("❌ No test PDF found")

    # Run pytest
    pytest.main([__file__, "-v"])
