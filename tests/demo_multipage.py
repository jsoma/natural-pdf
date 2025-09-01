"""Demo script showing multipage directional navigation."""

import natural_pdf as npdf
from natural_pdf import PDF


def main():
    """Demonstrate multipage directional navigation."""
    pdf = PDF("pdfs/sections.pdf")

    print("=== Multipage Directional Navigation Demo ===\n")

    # Find Section 1 on page 1
    section1 = pdf.pages[0].find("text:contains(Section 1)")
    print(f"Found Section 1 on page {section1.page.number}")

    # Without multipage - stops at page boundary
    print("\n1. Without multipage=True:")
    result = section1.below(until="text:contains(Section 6)")
    print(f"   Result type: {type(result).__name__}")
    print(f"   Result on page: {result.page.number}")
    print(f"   Text excerpt: {result.extract_text()[:50]}...")

    # With multipage=True - crosses page boundary
    print("\n2. With multipage=True:")
    result = section1.below(until="text:contains(Section 6)", multipage=True)
    print(f"   Result type: {type(result).__name__}")
    if hasattr(result, "constituent_regions"):
        print(f"   Spans {len(result.constituent_regions)} pages")
    text = result.extract_text()
    print(f"   Contains 'Section 6': {'Section 6' in text}")

    # Using global option
    print("\n3. Using global auto_multipage option:")
    original = npdf.options.layout.auto_multipage
    npdf.set_option("layout.auto_multipage", True)

    result = section1.below(until="text:contains(Section 6)")  # No multipage param needed!
    print(f"   Result type: {type(result).__name__}")
    text = result.extract_text()
    print(f"   Contains 'Section 6': {'Section 6' in text}")

    # Restore original setting
    npdf.options.layout.auto_multipage = original

    # Example of above() with multipage
    print("\n4. Using above() with multipage:")
    section6 = pdf.pages[1].find("text:contains(Section 6)")
    result = section6.above(multipage=True)
    print(f"   Result type: {type(result).__name__}")
    if hasattr(result, "constituent_regions"):
        print(f"   Spans {len(result.constituent_regions)} pages")


if __name__ == "__main__":
    main()
