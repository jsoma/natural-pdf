# Working with Regions

Regions are one of the most powerful features in Natural PDF. A region is simply a rectangular area on a page that you can interact with - extract text from it, find elements within it, or generate images of it.

You can create regions in several ways:

1. **Manual creation**: `page.create_region(x0, y0, x1, y1)`
2. **Spatial relations**: `element.above()` or `element.below()`
3. **Content-based**: `element.select_until("another element")`
4. **Structural**: `page.get_sections(start_elements="text:bold")`

## Creating Regions

There are several ways to create regions:

### From an Element

The simplest way to create a region is based on an existing element:

```python
# Get a region below a heading
title = page.find('text:contains("Summary")')
summary_region = title.below()

# Get a region above an element
footer = page.find('text:contains("Page")')
content_region = footer.above()
```

### From One Element to Another

You can create a region spanning from one element to another:

```python
start = page.find('text:contains("Introduction")')
end = page.find('text:contains("Conclusion")')

# Create a region from start to end
content_region = start.until(end)
```

### Manually by Coordinates

You can create a region directly using coordinates:

```python
# Create a region from scratch
# (x0, top, x1, bottom)
region = page.create_region(100, 200, 400, 500)
```

### From Layout Analysis

You can get regions detected by document layout analysis:

```python
# Get regions detected by layout analysis
page.analyze_layout()
layout_regions = page.find_all('region')

# Get a specific type of region
tables = page.find_all('region[type=table]')
```

## Working with Regions

Once you have a region, you can:

### Extract Text

```python
# Extract all text from the region
text = region.extract_text()
```

### Find Elements Within the Region

```python
# Find elements within the region
bold_text = region.find_all('text:bold')
```

### Generate an Image of the Region

```python
# Generate an image of just this region
region_image = region.to_image(resolution=150)

# Save it to a file
region.save_image("region.png")

# Options for customization
region.save_image(
    "region_no_border.png", 
    crop_only=True           # Don't add a border
)

region.save_image(
    "region_high_res.png", 
    resolution=300           # Higher resolution
)
```

### Modify the Region

You can expand or adjust a region:

```python
# Expand a region in all directions
larger_region = region.expand(left=10, right=10, top=10, bottom=10)

# Expand by a factor
doubled_region = region.expand(width_factor=2, height_factor=2)
```

## Practical Examples

Here are some practical examples of using regions:

### Extract a Section Between Headings

```python
heading1 = page.find('text:contains("Introduction")')
heading2 = page.find('text:contains("Methods")')

# Get the content between the headings
content = heading1.below(until='text:contains("Methods")', include_endpoint=False)
text = content.extract_text()
```

### Extract a Table with its Caption

```python
# Find a table caption
caption = page.find('text:contains("Table 1")')

# Look for the table below the caption
table_region = caption.below(height=200)  # Approximate height

# Extract the table
table_text = table_region.extract_text()
```

### Exclude Headers and Footers

```python
# Define header and footer as exclusion zones
header = page.find('text:contains("CONFIDENTIAL")').above()
footer = page.find('text:contains("Page")').below()

page.add_exclusion(header)
page.add_exclusion(footer)

# Now extract text from a region without the header/footer content
region = page.create_region(50, 100, page.width - 50, page.height - 100)
clean_text = region.extract_text()  # Headers/footers automatically excluded
```

## Next Steps

Now that you know how to work with regions, check out:

- [Visual Debugging](../visual-debugging/index.md) to see what you're extracting
- [OCR](../ocr/index.md) for working with scanned documents
- [Layout Analysis](../layout-analysis/index.md) for automatically detecting regions