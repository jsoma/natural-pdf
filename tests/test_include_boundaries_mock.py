"""
Test include_boundaries with a mock setup to verify the fix works.
"""

import natural_pdf as npdf
from natural_pdf.core.interfaces import SupportsSections
from natural_pdf.elements.element_collection import ElementCollection
from natural_pdf.elements.region import Region
from natural_pdf.elements.text import TextElement


class MockSectionsPage(SupportsSections):
    """Lightweight page stub that satisfies the SupportsSections contract."""

    def __init__(self, number: int = 1, index: int = 0, width: int = 612, height: int = 792):
        self.number = number
        self.index = index
        self.width = width
        self.height = height
        self.pdf = None
        self._elements: list[TextElement] = []

    def attach(self, pdf) -> None:
        self.pdf = pdf

    def add_elements(self, elements: list[TextElement]) -> None:
        self._elements.extend(elements)

    def find_all(
        self,
        selector: str | None = None,
        *,
        text=None,
        **kwargs,
    ) -> ElementCollection:
        if selector and "Section" in selector:
            matching = [el for el in self._elements if "Section" in getattr(el, "text", "")]
            return ElementCollection(matching)
        return ElementCollection(list(self._elements))

    def get_elements(self, apply_exclusions: bool = True) -> list[TextElement]:
        return list(self._elements)

    def get_section_between(
        self,
        start,
        end,
        include_boundaries: str = "both",
        orientation: str = "vertical",
    ) -> Region | None:
        if start is None and end is None:
            return None

        end_top = getattr(end, "top", self.height)
        end_bottom = getattr(end, "bottom", self.height)

        if include_boundaries == "both":
            top = getattr(start, "top", 0)
            bottom = end_bottom
        elif include_boundaries == "start":
            top = getattr(start, "top", 0)
            bottom = end_top
        elif include_boundaries == "end":
            top = getattr(start, "bottom", 0)
            bottom = end_bottom
        else:
            top = getattr(start, "bottom", 0)
            bottom = end_top

        if top > bottom:
            top, bottom = bottom, top

        return Region(self, (0, top, self.width, bottom))

    def get_sections(self, start_elements=None, end_elements=None, **kwargs) -> ElementCollection:
        from natural_pdf.core.page_collection import PageCollection

        return PageCollection([self]).get_sections(
            start_elements=start_elements,
            end_elements=end_elements,
            **kwargs,
        )

    def to_region(self) -> Region:
        return Region(self, (0, 0, self.width, self.height))


def create_mock_element(page, text, top, bottom, x0=0, x1=100):
    """Create a mock text element."""
    obj = {
        "text": text,
        "x0": x0,
        "top": top,
        "x1": x1,
        "bottom": bottom,
        "height": bottom - top,
        "page_number": page.number,
    }
    element = TextElement(obj, page)
    return element


def test_get_sections_include_boundaries():
    """Test that include_boundaries parameter works correctly in get_sections."""

    # Create mock PDF and pages
    class SimplePDF:
        def __init__(self):
            self.pages = []

    pdf = SimplePDF()

    # Create mock page
    page = MockSectionsPage()
    page.attach(pdf)
    pdf.pages.append(page)

    # Create mock elements on the page
    header_element = create_mock_element(page, "Section 1", top=100, bottom=120)

    # Content in middle
    content_elements = [
        create_mock_element(page, "Content line 1", top=150, bottom=170),
        create_mock_element(page, "Content line 2", top=200, bottom=220),
        create_mock_element(page, "Content line 3", top=250, bottom=270),
    ]

    # Next header (lower on page, higher Y value)
    next_header = create_mock_element(page, "Section 2", top=300, bottom=320)

    all_elements = [header_element] + content_elements + [next_header]
    page.add_elements(all_elements)

    # Create PageCollection with mocked pages
    pages = [page]

    # Import PageCollection and patch its initialization
    from natural_pdf.core.page_collection import PageCollection

    collection = PageCollection(pages)
    collection.pages = pages

    # Test get_sections with different include_boundaries settings
    print("\nTesting get_sections with mock data...")

    # Mock the find_all method on collection
    collection.find_all = lambda selector, **kwargs: ElementCollection(
        [header_element, next_header]
    )

    # Test each include_boundaries option
    for boundaries in ["both", "start", "end", "none"]:
        sections = collection.get_sections("text:contains(Section)", include_boundaries=boundaries)

        print(f"\ninclude_boundaries='{boundaries}':")
        print(f"  Number of sections: {len(sections)}")

        if len(sections) > 0:
            section = sections[0]
            print(f"  Section bbox: {section.bbox}")
            print(f"  Top: {section.bbox[1]}, Bottom: {section.bbox[3]}")

            if boundaries in ("both", "start"):
                assert (
                    section.bbox[1] == header_element.top
                ), f"'{boundaries}' should start at the header's top"
                assert (
                    section.bbox[3] >= header_element.bottom
                ), f"'{boundaries}' should include at least the header height"
            else:
                assert (
                    section.bbox[1] == header_element.bottom
                ), f"'{boundaries}' should start below the header"
                assert (
                    section.bbox[3] >= section.bbox[1]
                ), f"'{boundaries}' should produce a non-negative height"

    print("\n✅ All mock tests passed! include_boundaries parameter is working correctly.")


def test_real_pdf_simple():
    """Test with a real PDF using simple boundaries."""
    from pathlib import Path

    # Use the types PDF which is simpler
    pdf_path = Path(__file__).parent.parent / "pdfs" / "types-of-type.pdf"
    if not pdf_path.exists():
        print(f"Skipping real PDF test - {pdf_path} not found")
        return

    pdf = npdf.PDF(str(pdf_path))

    # Find any text elements
    all_text = pdf.find_all("text")
    if len(all_text) < 2:
        print("Not enough text elements for real PDF test")
        return

    # Use first and last text elements as boundaries
    first_text = all_text[0].extract_text().strip()[:20]

    print(f"\nTesting with real PDF using '{first_text}' as boundary...")

    # Get sections with different boundaries
    sections_both = pdf.get_sections(f"text:contains({first_text})", include_boundaries="both")
    sections_none = pdf.get_sections(f"text:contains({first_text})", include_boundaries="none")

    if len(sections_both) > 0 and len(sections_none) > 0:
        # Compare bounding boxes
        bbox_both = sections_both[0].bbox
        bbox_none = sections_none[0].bbox

        print(f"Section with 'both': {bbox_both}")
        print(f"Section with 'none': {bbox_none}")

        # Basic check - they should be different
        assert (
            bbox_both != bbox_none
        ), "Bounding boxes should be different with different include_boundaries"
        print("✅ Real PDF test passed!")


if __name__ == "__main__":
    test_get_sections_include_boundaries()
    test_real_pdf_simple()
