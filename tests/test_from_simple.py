"""Simple test for from= parameter self-exclusion."""

import natural_pdf as npdf

# Load a PDF
pdf = npdf.PDF("pdfs/01-practice.pdf")
page = pdf.pages[0]

# Find a text element
elem = page.find("text:contains('the')")
print(f"Source element: '{elem.extract_text()}'")
print(f"Source bbox: {elem.bbox}")
print()

# Test 1: from='start'
print("Test 1: from='start' (default)")
region = elem.below(until="text")
if region and hasattr(region, "boundary_element"):
    target = region.boundary_element
    print(f"Found target: '{target.extract_text()}'")
    print(f"Target bbox: {target.bbox}")
    print(f"Same as source? {target is elem}")
    assert target is not elem, "Should not find itself!"
    print("✓ Correctly excluded source element")
else:
    print("No boundary element found")

# Test 2: from='end'
print("\nTest 2: from='end'")
region2 = elem.below(until="text", anchor="end")
if region2 and hasattr(region2, "boundary_element"):
    target2 = region2.boundary_element
    print(f"Found target: '{target2.extract_text()}'")
    print(f"Target bbox: {target2.bbox}")
    print(f"Same as source? {target2 is elem}")
    assert target2 is not elem, "Should not find itself!"
    print("✓ Correctly excluded source element")
else:
    print("No boundary element found")

# Test 3: Check if from='start' and from='end' find different things
if (
    region
    and region2
    and hasattr(region, "boundary_element")
    and hasattr(region2, "boundary_element")
):
    print(
        f"\nDo from='start' and from='end' find the same target? {region.boundary_element is region2.boundary_element}"
    )
    if region.boundary_element is not region2.boundary_element:
        print("They found different targets! This suggests there might be overlapping text.")
        print(f"from='start' found at y={region.boundary_element.top}")
        print(f"from='end' found at y={region2.boundary_element.top}")

print("\nAll tests passed!")
