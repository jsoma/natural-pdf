"""Test that from='start' doesn't find itself."""

import natural_pdf as npdf

# Load a PDF
pdf = npdf.PDF("pdfs/01-practice.pdf")
page = pdf.pages[0]

# Find a text element
elem = page.find("text:contains('the')")
print(f"Starting element: '{elem.extract_text()}'")
print(f"Starting element bbox: {elem.bbox}")
print()

# Test 1: from='start' should NOT find itself
print("Test 1: from='start' (default)")
region = elem.below(until="text")
if region:
    print(f"Region created: {region.bbox}")
    # The target element that was found by 'until'
    if hasattr(region, "boundary_element"):
        target = region.boundary_element
        print(
            f"Target element found by until: '{target.extract_text() if hasattr(target, 'extract_text') else target}'"
        )
        print(f"Is target the same as source? {target is elem}")
        assert target is not elem, "from='start' should not find the source element itself!"
    else:
        print("No boundary element attribute")
else:
    print("No region found (until condition not met)")

# Test 2: from='end' should also not find itself
print("\nTest 2: from='end'")
region2 = elem.below(until="text", anchor="end")
if region2:
    print(f"Region created: {region2.bbox}")
    if hasattr(region2, "boundary_element"):
        target2 = region2.boundary_element
        print(
            f"Target element found by until: '{target2.extract_text() if hasattr(target2, 'extract_text') else target2}'"
        )
        print(f"Is target the same as source? {target2 is elem}")
        assert target2 is not elem, "from='end' should not find the source element itself!"
    else:
        print("No boundary element attribute")
else:
    print("No region found (until condition not met)")

# Test 3: Simple direct test
print("\nTest 3: Direct test - looking for next element below")
# Use a larger region to ensure we find something
next_below = elem.below(until="text", anchor="start", height=200)
if next_below and hasattr(next_below, "boundary_element"):
    target = next_below.boundary_element
    print(f"Source: '{elem.extract_text()[:50]}...'")
    print(f"Target: '{target.extract_text() if hasattr(target, 'extract_text') else target}'")
    print(f"Same element? {target is elem}")

print("\nDone!")
