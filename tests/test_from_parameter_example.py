"""Example demonstrating the from= parameter for handling overlapping elements."""

import natural_pdf as npdf

# Load a PDF
pdf = npdf.PDF("pdfs/01-practice.pdf")
page = pdf.pages[0]

# Find a text element
elem = page.find("text:contains('the')")
print(f"Starting element bbox: {elem.bbox}")
print(f"Element text: '{elem.extract_text()}'")
print()

# Example 1: Default behavior (from='start')
# This starts looking from the top of the element, so it will catch overlapping text
print("1. Default behavior (from='start'):")
region1 = elem.below(until="text", height=100)
if region1:
    found_text = region1.find("text")
    if found_text:
        print(f"   Found text: '{found_text.extract_text()}'")
        print(f"   Text bbox: {found_text.bbox}")
        print(f"   Text top position: {found_text.top}")

# Example 2: Strict behavior (from='end')
# This starts looking from the bottom of the element, so it skips overlapping text
print("\n2. Strict behavior (from='end'):")
region2 = elem.below(until="text", height=100, anchor="end")
if region2:
    found_text = region2.find("text")
    if found_text:
        print(f"   Found text: '{found_text.extract_text()}'")
        print(f"   Text bbox: {found_text.bbox}")
        print(f"   Text top position: {found_text.top}")

# Example 3: Center-based search (from='center')
# This starts looking from the center of the element
print("\n3. Center-based search (from='center'):")
region3 = elem.below(until="text", height=100, anchor="center")
if region3:
    found_text = region3.find("text")
    if found_text:
        print(f"   Found text: '{found_text.extract_text()}'")
        print(f"   Text bbox: {found_text.bbox}")
        print(f"   Text top position: {found_text.top}")

# Example 4: Using explicit edge names
print("\n4. Using explicit edges:")
print("   from='top' (same as 'start' for below):")
region4 = elem.below(until="text", height=100, anchor="top")
if region4:
    found_text = region4.find("text")
    if found_text:
        print(f"      Found text: '{found_text.extract_text()}'")

print("   from='bottom' (same as 'end' for below):")
region5 = elem.below(until="text", height=100, anchor="bottom")
if region5:
    found_text = region5.find("text")
    if found_text:
        print(f"      Found text: '{found_text.extract_text()}'")

# Summary
print("\nSummary:")
print("- from='start' (default): Includes overlapping elements")
print("- from='end': Excludes overlapping elements (strict)")
print("- from='center': Starts from element center")
print("- You can also use explicit edges: 'top', 'bottom', 'left', 'right'")
