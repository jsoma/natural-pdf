"""Test ElementCollection slicing behavior."""

import pytest

import natural_pdf as npdf
from natural_pdf.elements.element_collection import ElementCollection


def test_element_collection_slicing():
    """Test that slicing ElementCollection returns ElementCollection objects."""
    pdf = npdf.PDF("pdfs/01-practice.pdf")
    page = pdf.pages[0]

    # Get a collection of text elements
    text_elements = page.find_all("text")

    # Test that we have some elements to work with
    assert len(text_elements) > 5, "Need at least 5 text elements for testing"

    # Test single index access (should return Element)
    first_element = text_elements[0]
    assert not isinstance(first_element, ElementCollection)
    assert hasattr(first_element, "text"), "Should be a text element"

    # Test slice access (should return ElementCollection)
    first_three = text_elements[:3]
    assert isinstance(first_three, ElementCollection)
    assert len(first_three) == 3

    # Test slice with start and end
    middle_elements = text_elements[2:5]
    assert isinstance(middle_elements, ElementCollection)
    assert len(middle_elements) == 3

    # Test slice with step
    every_other = text_elements[::2]
    assert isinstance(every_other, ElementCollection)
    assert len(every_other) == (len(text_elements) + 1) // 2

    # Test negative indexing in slice
    last_three = text_elements[-3:]
    assert isinstance(last_three, ElementCollection)
    assert len(last_three) == 3

    # Test that sliced collections have the same methods as original
    assert hasattr(first_three, "show")
    assert hasattr(first_three, "extract_text")
    assert hasattr(first_three, "filter")

    # Test that sliced collections can be further sliced
    first_of_first_three = first_three[:1]
    assert isinstance(first_of_first_three, ElementCollection)
    assert len(first_of_first_three) == 1

    # Test that sliced collections work with show() method
    try:
        img = first_three.show()
        # show() should work without error (img might be None for some edge cases)
        assert img is not None or len(first_three) == 0
    except Exception as e:
        pytest.fail(f"show() method failed on sliced collection: {e}")

    pdf.close()


def test_empty_slice():
    """Test that empty slices return empty ElementCollection objects."""
    pdf = npdf.PDF("pdfs/01-practice.pdf")
    page = pdf.pages[0]

    text_elements = page.find_all("text")

    # Test empty slice
    empty_slice = text_elements[10:10]  # Empty range
    assert isinstance(empty_slice, ElementCollection)
    assert len(empty_slice) == 0

    # Test out-of-bounds slice
    out_of_bounds = text_elements[1000:2000]
    assert isinstance(out_of_bounds, ElementCollection)
    assert len(out_of_bounds) == 0

    pdf.close()


def test_slice_preserves_element_types():
    """Test that slicing preserves the correct element types."""
    pdf = npdf.PDF("pdfs/01-practice.pdf")
    page = pdf.pages[0]

    # Get different types of elements
    text_elements = page.find_all("text")

    if len(text_elements) > 0:
        # Slice and verify elements are still the correct type
        sliced = text_elements[:3]
        for element in sliced:
            assert hasattr(element, "text"), "Should still be text elements"

    pdf.close()


if __name__ == "__main__":
    test_element_collection_slicing()
    test_empty_slice()
    test_slice_preserves_element_types()
    print("All tests passed!")
