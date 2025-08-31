# Working with Regions

Regions are rectangular areas on a page that define boundaries for operations like text extraction, element finding, or visualization. They're one of Natural PDF's most powerful features for working with specific parts of a document.

## Setup

Let's set up a PDF to experiment with regions.

```python
from natural_pdf import PDF

# Load the PDF
pdf = PDF("https://github.com/jsoma/natural-pdf/raw/refs/heads/main/pdfs/01-practice.pdf")

# Get the first page
page = pdf.pages[0]

# Display the page
page.show(width=700)
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
mid_region.show(color="blue")
```

### Using Element Methods: `above()`, `below()`, `left()`, `right()`

You can create regions relative to existing elements. Natural PDF uses smart defaults for these directional methods:

- **`.left()` and `.right()`**: Default to `height='element'` (matches the element's height)
- **`.above()` and `.below()`**: Default to `width='full'` (full page width)

These defaults match common use cases - when looking sideways you usually want the same height as your reference element, while looking up/down typically needs the full page width.

```python
# Find a heading-like element
heading = page.find('text[size>=12]:bold')

# Create a region below this heading element
if heading:
    # Uses default width='full' - extends across full page width
    region_below = heading.below()

    # Highlight the heading and the region below it
    with page.highlights() as h:
        h.add(heading, color="red")
        h.add(region_below, color="blue")
        h.show()
```

```python
# Create regions to the left and right with smart defaults
if heading:
    # Default height='element' - matches heading height
    region_left = heading.left()
    region_right = heading.right()

    # Or specify custom dimensions
    region_left_tall = heading.left(height=200)  # 200px tall
    region_right_full = heading.right(height='full')  # Full page height

    with page.highlights() as h:
        h.add(heading, color="red")
        h.add(region_left, color="green", label="Left (element height)")
        h.add(region_right, color="blue", label="Right (element height)")
        h.show()
```

```python
# Create a region with height limit
if heading:
    # Only include 100px below the heading
    small_region_below = heading.below(height=100)

    with page.highlights() as h:
        h.add(heading, color="red")
        h.add(small_region_below, color="green")
        h.show()
```

```python
# Find a line or other element to create a region above
line = page.find('line')
if line:
    # Create a region above the line
    region_above = line.above()

    with page.highlights() as h:
        h.add(line, color="black")
        h.add(region_above, color="purple")
        h.show()
```

### Creating a Region Between Elements with `until()`

```python
# Find two elements to use as boundaries
first_heading = page.find('text[size>=11]:bold')
next_heading = first_heading.next('text[size>=11]:bold') if first_heading else None

if first_heading and next_heading:
    # Create a region from the first heading until the next heading
    section = first_heading.below(until=next_heading, include_endpoint=False)

    # Highlight both elements and the region between them
    with page.highlights() as h:
        h.add(first_heading, color="red")
        h.add(next_heading, color="red")
        h.add(section, color="yellow")
        h.show()
```

### Jump to the enclosing area with `parent()`

Need to know which *table*, *figure* or *coloured panel* an element sits inside?

The new `parent()` helper finds the **smallest element or detected region that
spatially encloses the current one**.

```python
# After running a layout model
page.analyze_layout('tatr')

text  = page.find(text="Hazardous Materials")
row       = text.parent('region[type=table-row]')
row.show(crop=True)
```

Parameters
```
parent(selector=None, *, mode="contains")

mode = "contains"  # candidate fully covers the element (default)
     | "center"    # candidate contains element's centroid
     | "overlap"   # any intersection > 0pt²
```

If no enclosing object matches the selector (or exists at all) it returns
`None`, so chaining is safe.

## Using Regions

Once you have a region, here's what you can do with it.

### Extract Text from a Region

```python
# Find a region to work with (e.g., from a title to the next bold text)
title = page.find('text:contains("Site")')  # Adjust if needed
if title:
    # Create a region from title down to the next bold text
    content_region = title.below(until='line:horizontal', include_endpoint=False)

    # Extract text from just this region
    region_text = content_region.extract_text()

    # Show the region and the extracted text
    content_region.show(color="green")

    # Displaying the text (first 300 chars if long)
    print(region_text[:300] + "..." if len(region_text) > 300 else region_text)
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
with page.highlights() as h:
    h.add(test_region, color="blue")
    h.add(text_in_region, color="red")
    h.show()

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
region_for_image.show(crop=True)  # Shows just the region
```

### Adjust and Expand Regions

```python
# Take an existing region and expand it
region_a = page.create_region(200, 200, 400, 400)

# Expand by a certain number of points in each direction
expanded = region_a.expand(left=20, right=20, top=20, bottom=20)

# Visualize original and expanded regions
with page.highlights() as h:
    h.add(region_a, color="blue", label="Original")
    h.add(expanded, color="red", label="Expanded")
    h.show()
```

### Global Offset Configuration

You can configure global offsets that will be applied to all regions created with directional methods. This is useful for consistently adding padding or margins:

```python
from natural_pdf import PDF

# Configure global offsets for all PDFs
PDF.configure_offsets(
    below_offset=5,     # Add 5px gap below elements
    above_offset=5,     # Add 5px gap above elements
    left_offset=2,      # Add 2px gap to the left
    right_offset=2      # Add 2px gap to the right
)

# Now all directional methods will include these offsets
heading = page.find('text:bold')
if heading:
    # This region will start 5px below the heading (not touching)
    content_below = heading.below()

    # This region will end 5px above the heading
    content_above = heading.above(height=100)
```

```python
# Reset to default offsets (all 0)
PDF.configure_offsets(
    below_offset=0,
    above_offset=0,
    left_offset=0,
    right_offset=0
)
```

These offsets are particularly useful when:
- Extracting text that might be too close to headers/footers
- Creating regions that need consistent spacing
- Working with documents that have tight layouts

## Using Exclusion Zones with Regions

Exclusion zones are regions that you want to ignore during operations like text extraction.

```python
# Create a region for the whole page
full_page_region = page.create_region(0, 0, page.width, page.height)

# Extract text without exclusions as baseline
full_text = full_page_region.extract_text()
print(f"Full page text length: {len(full_text)} characters")
```

```python
# Define an area we want to exclude (like a header)
# Let's exclude the top 10% of the page
header_zone = page.create_region(0, 0, page.width, page.height * 0.1)

# Add this as an exclusion for the page
page.add_exclusion(header_zone)

# Visualize the exclusion
header_zone.show(color="red", label="Excluded")
```

```python
# Now extract text again - the header should be excluded
text_with_exclusion = full_page_region.extract_text() # Uses apply_exclusions=True by default

# Compare text lengths
print(f"Original text: {len(full_text)} chars\nText with exclusion: {len(text_with_exclusion)} chars")
print(f"Difference: {len(full_text) - len(text_with_exclusion)} chars excluded")
```

```python
# When done with this page, clear exclusions
page.clear_exclusions()
```

## Document-Level Exclusions

PDF-level exclusions apply to all pages and use functions to adapt to each page.

```python
# Define a PDF-level exclusion for headers
# This will exclude the top 30% of every page
pdf.add_exclusion(
    lambda p: p.create_region(0, 0, p.width, p.height * 0.3),
    label="Header zone"
)

# Define a PDF-level exclusion for footers
# This will exclude the bottom 20% of every page
pdf.add_exclusion(
    lambda p: p.create_region(0, p.height * 0.8, p.width, p.height),
    label="Footer zone"
)

# PDF-level exclusions are used whenever you extract text
# Let's try on the first three pages
for page in pdf.pages[:3]:
    text = page.extract_text()
    text_original = page.extract_text(use_exclusions=False)
    print(f"Page {page.number} – Before: {len(text_original)} After: {len(text)}")
```

```python
# Clear PDF-level exclusions when done
pdf.clear_exclusions()
print("Cleared all PDF-level exclusions")
```

## Working with Layout Analysis Regions

When you run layout analysis, the detected regions (tables, titles, etc.) are also Region objects.

```python
# First, run layout analysis to detect regions
page.analyze_layout()  # Uses 'yolo' engine by default

# Find all detected regions
detected_regions = page.find_all('region')
print(f"Found {len(detected_regions)} layout regions")
```

```python
# Highlight all detected regions by type
detected_regions.show(group_by='region_type')
```

```python
# Extract text from a specific region type (e.g., title)
title_regions = page.find_all('region[type=title]')
if title_regions:
    titles_text = title_regions.extract_text()
    print(f"Title text: {titles_text}")
```

## Next Steps

Now that you understand regions, you can:

- [Extract tables](../tables/index.ipynb) from table regions
- [Ask questions](../document-qa/index.ipynb) about specific regions
- [Exclude content](../text-extraction/index.md#filtering-out-headers-and-footers) from extraction
