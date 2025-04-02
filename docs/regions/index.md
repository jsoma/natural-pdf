# Working with Regions

Regions are rectangular areas on a page that define boundaries for operations like text extraction, element finding, or visualization. They're one of Natural PDF's most powerful features for working with specific parts of a document.

## Setup

Let's set up a PDF to experiment with regions.

```python
from natural_pdf import PDF
from pathlib import Path

# Path to sample PDF
pdf_path = Path("../tutorials/pdfs/01-practice.pdf")

# Load the PDF
pdf = PDF(pdf_path)

# Get the first page
page = pdf.pages[0]

# Display the page
page.show()
```

## Creating Regions

There are several ways to create regions in Natural PDF.

### Using `create_region()` with Coordinates

This is the most direct method - provide the coordinates directly.

```python
# Create a region by specifying (x0, top, x1, bottom) coordinates
# Let's create a region in the middle of the page
mid_region = page.create_region(
    x0=100,         # Left edge
    top=200,        # Top edge
    x1=500,         # Right edge
    bottom=400      # Bottom edge
)

# Highlight the region to see it
page.highlight(mid_region, color="blue", alpha=0.3).show()
```

### Using Element Methods: `above()`, `below()`, `left_of()`, `right_of()`

You can create regions relative to existing elements.

```python
# Find a heading-like element
heading = page.find('text[size>=12]:bold')

# Create a region below this heading element
if heading:
    region_below = heading.below()
    
    # Highlight the heading and the region below it
    page.copy().highlight(heading, color="red")\
              .highlight(region_below, color="blue", alpha=0.3)\
              .show()
```

```python
# Create a region with height limit
if heading:
    # Only include 100px below the heading
    small_region_below = heading.below(height=100)
    
    page.copy().highlight(heading, color="red")\
              .highlight(small_region_below, color="green", alpha=0.3)\
              .show()
```

```python
# Find a line or other element to create a region above
line = page.find('line')
if line:
    # Create a region above the line
    region_above = line.above()
    
    page.copy().highlight(line, color="black")\
              .highlight(region_above, color="purple", alpha=0.3)\
              .show()
```

### Creating a Region Between Elements with `until()`

```python
# Find two elements to use as boundaries
first_heading = page.find('text[size>=11]:bold')
next_heading = first_heading.next('text[size>=11]:bold') if first_heading else None

if first_heading and next_heading:
    # Create a region from the first heading until the next heading
    section = first_heading.below(until=next_heading, include_until=False)
    
    # Highlight both elements and the region between them
    page.copy().highlight(first_heading, color="red")\
              .highlight(next_heading, color="red")\
              .highlight(section, color="yellow", alpha=0.3)\
              .show()
```

## Using Regions

Once you have a region, here's what you can do with it.

### Extract Text from a Region

```python
# Find a region to work with (e.g., from a title to the next bold text)
title = page.find('text:contains("Jungle")')  # Adjust if needed
if title:
    # Create a region from title down to the next bold text
    content_region = title.below(until='text:bold', include_until=False)
    
    # Extract text from just this region
    region_text = content_region.extract_text()
    
    # Show the region and the extracted text
    page.highlight(content_region, color="green", alpha=0.3).show()
    
    # Displaying the text (first 300 chars if long)
    region_text[:300] + "..." if len(region_text) > 300 else region_text
```

### Find Elements Within a Region

You can use a region as a "filter" to only find elements within its boundaries.

```python
# Create a region in an interesting part of the page
test_region = page.create_region(
    x0=page.width * 0.1, 
    top=page.height * 0.25, 
    x1=page.width * 0.9, 
    bottom=page.height * 0.75
)

# Find all text elements ONLY within this region
text_in_region = test_region.find_all('text')

# Display result
page.copy().highlight(test_region, color="blue", alpha=0.2)\
          .highlight_all(text_in_region, color="red")\
          .show()

len(text_in_region)  # Number of text elements found in region
```

### Generate an Image of a Region

```python
# Find a specific region to capture
# (Could be a table, figure, or any significant area)
region_for_image = page.create_region(
    x0=100, 
    top=150,
    x1=page.width - 100,
    bottom=300
)

# Generate an image of just this region
region_for_image.show()  # Shows just the region
```

### Check Relationships Between Regions and Elements

You can determine if elements are within regions or if regions overlap.

```python
# Create two regions that may overlap
region_a = page.create_region(100, 200, 400, 400)
region_b = page.create_region(300, 300, 600, 500)

# Check if they intersect
does_intersect = region_a.intersects(region_b)

# Visualize the regions
img = page.copy()
img.highlight(region_a, color="red", alpha=0.3, label="Region A")
img.highlight(region_b, color="blue", alpha=0.3, label="Region B")
img.show()

f"Do the regions intersect? {does_intersect}"
```

```python
# Check if an element is within a region
element = page.find('text') # Just find any element

if element and region_a:
    # Check if the element is inside region_a
    is_inside = region_a.contains(element)
    
    # Visualize
    img = page.copy()
    img.highlight(region_a, color="green", alpha=0.3, label="Region A")
    img.highlight(element, color="red" if not is_inside else "blue")
    img.show()
    
    f"Is the element inside Region A? {is_inside}"
```

### Adjust and Expand Regions

```python
# Take an existing region and expand it
if 'region_a' in locals():
    # Expand by a certain number of points in each direction
    expanded = region_a.expand(left=20, right=20, top=20, bottom=20)
    
    # Visualize original and expanded regions
    img = page.copy()
    img.highlight(region_a, color="blue", alpha=0.3, label="Original")
    img.highlight(expanded, color="red", alpha=0.3, label="Expanded")
    img.show()
```

```python
# Expand by a factor (double the width and height)
if 'region_a' in locals():
    doubled = region_a.expand(width_factor=2, height_factor=2)
    
    # Visualize
    img = page.copy()
    img.highlight(region_a, color="blue", alpha=0.3, label="Original")
    img.highlight(doubled, color="purple", alpha=0.3, label="Doubled")
    img.show()
```

## Using Exclusion Zones with Regions

Exclusion zones are regions that you want to ignore during operations like text extraction.

```python
# Create a region for the whole page
full_page = page.create_region(0, 0, page.width, page.height)

# Extract text without exclusions as baseline
full_text = full_page.extract_text()
f"Full page text length: {len(full_text)} characters"
```

```python
# Define an area we want to exclude (like a header)
# Let's exclude the top 10% of the page
header_zone = page.create_region(0, 0, page.width, page.height * 0.1)

# Add this as an exclusion for the page
page.add_exclusion(header_zone)

# Visualize the exclusion
page.highlight(header_zone, color="red", alpha=0.3, label="Excluded").show()
```

```python
# Now extract text again - the header should be excluded
text_with_exclusion = full_page.extract_text() # Uses apply_exclusions=True by default

# Compare text lengths
f"Original text: {len(full_text)} chars\nText with exclusion: {len(text_with_exclusion)} chars"
f"Difference: {len(full_text) - len(text_with_exclusion)} chars excluded"
```

```python
# Temporarily bypass exclusions if needed
text_ignoring_exclusion = full_page.extract_text(apply_exclusions=False)
f"Text ignoring exclusions: {len(text_ignoring_exclusion)} chars (should match original)"
```

```python
# When done with this page, clear exclusions
page.clear_exclusions()
f"Page now has {len(page.exclusions)} exclusions"
```

## Document-Level Exclusions

PDF-level exclusions apply to all pages and use functions to adapt to each page.

```python
# Define a PDF-level exclusion for headers
# This will exclude the top 10% of every page
pdf.add_exclusion(
    lambda p: p.create_region(0, 0, p.width, p.height * 0.1),
    label="Header zone"
)

# Define a PDF-level exclusion for footers
# This will exclude the bottom 10% of every page
pdf.add_exclusion(
    lambda p: p.create_region(0, p.height * 0.9, p.width, p.height),
    label="Footer zone"
)

# PDF-level exclusions are used whenever you extract text
# Let's try on the first three pages
for i in range(min(3, len(pdf.pages))):
    page_i = pdf.pages[i]
    text = page_i.extract_text()
    print(f"Page {i+1}: {len(text)} characters after exclusions")
```

```python
# Clear PDF-level exclusions when done
pdf.clear_exclusions()
"Cleared all PDF-level exclusions"
```

## Working with Layout Analysis Regions

When you run layout analysis, the detected regions (tables, titles, etc.) are also Region objects.

```python
# First, run layout analysis to detect regions
page.analyze_layout()  # Uses 'yolo' engine by default

# Find all detected regions
detected_regions = page.find_all('region')
f"Found {len(detected_regions)} layout regions"
```

```python
# Highlight all detected regions by type
page.highlight_all(detected_regions, group_by='type', label_attribute='type').show()
```

```python
# Extract text from a specific region type (e.g., title)
title_regions = page.find_all('region[type=title]')
if title_regions:
    titles_text = title_regions.extract_text(delimiter="\n")
    f"Title text: {titles_text}"
```

## Next Steps

Now that you understand regions, you can:

- [Create multi-section documents](../section-extraction/index.md) by dividing a document into regions
- [Extract tables](../tables/index.md) from table regions
- [Ask questions](../document-qa/index.md) about specific regions
- [Exclude content](../text-extraction/index.md#filtering-out-headers-and-footers) from extraction