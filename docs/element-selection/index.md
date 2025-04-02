# Finding Elements with Selectors

Natural PDF uses CSS-like selectors to find elements (text, lines, images, etc.) within a PDF page or document. This guide demonstrates how to use these selectors effectively.

## Setup

Let's load a sample PDF to work with. We'll use `01-practice.pdf` which has various elements.

```python
from natural_pdf import PDF

# Load the PDF
pdf = PDF("https://github.com/jsoma/natural-pdf/raw/refs/heads/main/pdfs/01-practice.pdf")

# Select the first page
page = pdf.pages[0]

# Display the page
page.show()
```

## Basic Element Finding

The core methods are `find()` (returns the first match) and `find_all()` (returns all matches as an `ElementCollection`).

The basic selector structure is `element_type[attribute_filter]:pseudo_class`.

### Finding Text by Content

```python
# Find the first text element containing "Summary"
summary_text = page.find('text:contains("Summary")')
summary_text # Display the element object
```

```python
# Highlight the found element
if summary_text:
    page.highlight(summary_text, color="yellow").show()
```

```python
# Find all text elements containing "the"
all_the_text = page.find_all('text:contains("the")')
len(all_the_text) # Show how many were found
```

```python
# Highlight all occurrences of "the"
if all_the_text:
    page.highlight_all(all_the_text, color="lightblue").show()
```

## Selecting by Element Type

You can select specific types of elements found in PDFs.

```python
# Find all text elements
all_text = page.find_all('text')
len(all_text)
```

```python
# Find all rectangle elements
all_rects = page.find_all('rect')
len(all_rects)
```

```python
# Highlight the first rectangle found
if all_rects:
    page.highlight(all_rects[0], color="pink").show()
```

```python
# Find all line elements
all_lines = page.find_all('line')
len(all_lines)
```

```python
# Highlight all lines
if all_lines:
    page.highlight_all(all_lines, color="lightgreen").show()
```

*Note: `image`, `curve`, and `region` (from layout analysis) are other common types.*

## Filtering by Attributes

Use square brackets `[]` to filter elements by their properties (attributes).

### Common Attributes & Operators

| Attribute     | Example Usage          | Operators | Notes |
|---------------|------------------------|-----------|-------|
| `size` (text) | `text[size>=12]`       | `>`, `<`, `>=`, `<=` | Font size in points |
| `fontname`    | `text[fontname*=Bold]` | `=`, `*=`  | `*=` for contains substring |
| `color` (text)| `text[color~=red]`     | `~=`      | Approx. match (name, rgb, hex) |
| `width` (line)| `line[width>1]`        | `>`, `<`, `>=`, `<=` | Line thickness |
| `source`      | `text[source=ocr]`     | `=`       | `pdf`, `ocr`, `detected` |
| `type` (region)| `region[type=table]`  | `=`       | Layout analysis region type |

```python
# Find large text (size >= 11 points)
large_text = page.find_all('text[size>=11]')
large_text # Display the list
```

```python
# Highlight large text
if large_text:
    page.highlight_all(large_text, color="orange").show()
```

```python
# Find text with 'Helvetica' in the font name
helvetica_text = page.find_all('text[fontname*=Helvetica]')
helvetica_text
```

```python
# Find red text (using approximate color match)
# This PDF has text with color (0.8, 0.0, 0.0)
red_text = page.find_all('text[color~=red]')
red_text
```

```python
# Highlight the red text
if red_text:
    page.highlight(red_text[0], color="red").show() # Highlight just the first match
```

```python
# Find thick lines (width >= 2)
thick_lines = page.find_all('line[width>=2]')
thick_lines
```

```python
# Highlight the thick lines
if thick_lines:
    page.highlight_all(thick_lines, color="purple").show()
```

## Using Pseudo-Classes

Use colons `:` for special conditions (pseudo-classes).

### Common Pseudo-Classes

| Pseudo-Class          | Example Usage                           | Notes |
|-----------------------|-----------------------------------------|-------|
| `:contains('text')` | `text:contains('Report')`             | Finds elements containing specific text |
| `:bold`               | `text:bold`                             | Finds text heuristically identified as bold |
| `:italic`             | `text:italic`                           | Finds text heuristically identified as italic |
| `:below(selector)`    | `text:below('line[width>=2]')`         | Finds elements physically below the reference element |
| `:above(selector)`    | `text:above('text:contains("Summary")')`| Finds elements physically above the reference element |
| `:left-of(selector)`  | `line:left-of('rect')`                 | Finds elements physically left of the reference element |
| `:right-of(selector)` | `text:right-of('rect')`                | Finds elements physically right of the reference element |
| `:near(selector)`     | `text:near('image')`                   | Finds elements physically near the reference element |

*Note: Spatial pseudo-classes like `:below`, `:above` identify elements based on bounding box positions relative to the **first** element matched by the inner selector.*

```python
# Find bold text
bold_text = page.find_all('text:bold')
bold_text
```

```python
# Highlight bold text
if bold_text:
    page.highlight_all(bold_text, color="teal").show()
```

```python
# Combine attribute and pseudo-class: bold text size >= 11
large_bold_text = page.find_all('text[size>=11]:bold')
large_bold_text
```

```python
# Highlight large bold text
if large_bold_text:
    page.highlight_all(large_bold_text, color="lime").show()
```

### Spatial Pseudo-Classes Examples

```python
# Find the thick horizontal line first
ref_line = page.find('line[width>=2]')

# Find text elements strictly above that line
text_above_line = page.find_all('text:above("line[width>=2]")')
text_above_line
```

```python
# Highlight the line and the text found above it
if ref_line and text_above_line:
    page.copy().highlight(ref_line, color="red")\
               .highlight_all(text_above_line, color="blue")\
               .show()
```

```python
# Find text near the red text element
# (We found red_text earlier)
if red_text:
    nearby_text = page.find_all('text:near("text[color~=red]")')
    # Highlight the red text and nearby text
    page.copy().highlight(red_text[0], color="red")\
               .highlight_all(nearby_text, color="orange", alpha=0.3)\
               .show()
```

## Advanced Text Searching Options

Pass options to `find()` or `find_all()` for more control over text matching.

```python
# Case-insensitive search for "summary"
page.find_all('text:contains("summary")', case=False)
```

```python
# Regular expression search for the inspection ID (e.g., INS-XXX...)
# The ID is in the red text we found earlier
page.find_all('text:contains("INS-\\w+")', regex=True)
```

```python
# Combine regex and case-insensitivity
page.find_all('text:contains("jungle health")', regex=True, case=False)
```

## Working with ElementCollections

`find_all()` returns an `ElementCollection`, which is like a list but with extra PDF-specific methods.

```python
# Get all headings (using a selector for large, bold text)
headings = page.find_all('text[size>=11]:bold')
headings
```

```python
# Get the first and last heading in reading order
first = headings.first
last = headings.last
(first, last)
```

```python
# Get the physically highest/lowest element in the collection
highest = headings.highest()
lowest = headings.lowest()
(highest, lowest)
```

```python
# Filter the collection further: headings containing "Service"
service_headings = headings.filter('text:contains("Service")')
service_headings
```

```python
# Extract text from all elements in the collection
headings.extract_text(delimiter=" | ")
```

```python
# Highlight all elements in the collection (returns the collection)
headings.highlight(color="magenta", label="Headings")
page.show() # Show the page with persistent highlights
```

```python
# Clear highlights from the page
page.clear_highlights()
page.show()
```

*Remember: `.highest()`, `.lowest()`, `.leftmost()`, `.rightmost()` raise errors if the collection spans multiple pages.*

## Font Variants

Sometimes PDFs use font variants (prefixes like `AAAAAB+`) which can be useful for selection.

```python
# Find text elements with a specific font variant prefix (if any exist)
# This example PDF doesn't use variants heavily, but the selector works like this:
variant_text = page.find_all('text[font-variant=AAAAAB]')
# If found, you could highlight them:
# if variant_text: page.highlight_all(variant_text).show()
len(variant_text)
```

## Next Steps

Now that you can find elements, explore:

- [Text Extraction](../text-extraction/index.md): Get text content from found elements.
- [Spatial Navigation](../pdf-navigation/index.md): Use found elements as anchors to navigate (`.above()`, `.below()`, etc.).
- [Working with Regions](../regions/index.md): Define areas based on found elements.
- [Visual Debugging](../visual-debugging/index.md): Techniques for highlighting and visualizing elements.