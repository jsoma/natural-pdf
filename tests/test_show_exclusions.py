#!/usr/bin/env python3
"""Test visualizing exclusions with show()."""

from pathlib import Path

from natural_pdf import PDF


def test_exclusion_visualization():
    """Test if we can visualize exclusions."""

    # Find test PDF
    project_root = Path(__file__).parent.parent
    pdf_path = project_root / "pdfs/practice.pdf"
    if not pdf_path.exists():
        pdf_path = (
            project_root
            / "bad-pdfs/submissions/Doc 06 - Approved Expenses 07.01.2022-06.30.2023 Marketplace Transactions - REDACTED.pdf"
        )

    if not pdf_path.exists():
        print("No test PDF found")
        return

    pdf = PDF(pdf_path)
    page = pdf[0]

    # Add some exclusions
    page.add_exclusion(page.region(0, 0, page.width, 50), label="header")

    # Get the exclusion regions
    exclusion_regions = page._get_exclusion_regions(include_callable=True)
    print(f"Found {len(exclusion_regions)} exclusion regions")

    # Try different ways to visualize exclusions
    print("\n1. Testing if show() accepts exclusions parameter...")
    try:
        # This might not work but let's try
        page.show(exclusions="red")
        print("   ✅ exclusions='red' parameter accepted!")
    except TypeError as e:
        print(f"   ❌ exclusions parameter not accepted: {e}")

    print("\n2. Testing with highlights parameter...")
    try:
        # Create highlights from exclusion regions
        exclusion_highlights = {
            "elements": exclusion_regions,
            "color": "red",
            "label": "Exclusions",
        }
        page.show(highlights=[exclusion_highlights])
        print("   ✅ Exclusions shown using highlights!")
    except Exception as e:
        print(f"   ❌ Highlights approach failed: {e}")

    print("\n3. Testing with manual region highlighting...")
    try:
        # Highlight exclusion regions manually
        for region in exclusion_regions:
            region.highlight(color="red")
        page.show()
        print("   ✅ Exclusions highlighted manually!")
    except Exception as e:
        print(f"   ❌ Manual highlighting failed: {e}")

    print("\n4. Checking if debug_exclusions shows visual output...")
    try:
        # Get elements with debug mode
        elements = page.get_elements(debug_exclusions=True)
        print(f"   ℹ️  Got {len(elements)} elements with debug_exclusions=True")
    except Exception as e:
        print(f"   ❌ debug_exclusions failed: {e}")


if __name__ == "__main__":
    test_exclusion_visualization()
