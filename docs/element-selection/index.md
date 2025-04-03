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
summary_text
```

```python
# Find all text elements containing "Inadequate"
contains_inadequate = page.find_all('text:contains("Inadequate")')
len(contains_inadequate)
```

```python
summary_text.highlight(label='summary')
contains_inadequate.highlight(label="inadequate")
page.to_image(width=700)
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
# Find all line elements
all_lines = page.find_all('line')
len(all_lines)
```

```python
page.find_all('line').show()
```

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
page.find_all('text[size>=11]')
```

```python
# Find text with 'Helvetica' in the font name
page.find_all('text[fontname*=Helvetica]')
```

```python
# Find red text (using approximate color match)
# This PDF has text with color (0.8, 0.0, 0.0)
red_text = page.find_all('text[color~=red]')
```

```python
# Highlight the red text (ignoring existing highlights)
red_text.show()
```

```python
# Find thick lines (width >= 2)
page.find_all('line[width>=2]')
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
page.find_all('text:bold').show()
```

```python
# Combine attribute and pseudo-class: bold text size >= 11
page.find_all('text[size>=11]:bold')
```

### Spatial Pseudo-Classes Examples

```python
# Find the thick horizontal line first
ref_line = page.find('line[width>=2]')

# Find text elements strictly above that line
text_above_line = page.find_all('text:above("line[width>=2]")')
text_above_line
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
service_headings = headings.find_all('text:contains("Service")')
service_headings
```

```python
# Extract text from all elements in the collection
headings.extract_text()
```

*Remember: `.highest()`, `.lowest()`, `.leftmost()`, `.rightmost()` raise errors if the collection spans multiple pages.*

## Font Variants

Sometimes PDFs use font variants (prefixes like `AAAAAB+`) which can be useful for selection.

```python
# Find text elements with a specific font variant prefix (if any exist)
# This example PDF doesn't use variants, but the selector works like this:
page.find_all('text[font-variant=AAAAAB]')
```

## Next Steps

Now that you can find elements, explore:

- [Text Extraction](../text-extraction/index.ipynb): Get text content from found elements.
- [Spatial Navigation](../pdf-navigation/index.ipynb): Use found elements as anchors to navigate (`.above()`, `.below()`, etc.).
- [Working with Regions](../regions/index.ipynb): Define areas based on found elements.
- [Visual Debugging](../visual-debugging/index.ipynb): Techniques for highlighting and visualizing elements.