# Visual Debugging

Sometimes it's hard to understand what's happening when working with PDFs. Natural PDF provides powerful visual debugging tools to help you see what you're extracting.

## Adding Persistent Highlights

Use the `.highlight()` method on `Element` or `ElementCollection` objects to add persistent highlights to a page. These highlights are stored and will appear when viewing the page later.

```python
from natural_pdf import PDF

pdf = PDF('document.pdf')
page = pdf.pages[0]

# Find a specific element and add a persistent highlight
title = page.find('text:contains("Summary")')
title.highlight()

## Viewing Highlights

You can view persistent highlights in two ways:

1. **Interactive Viewer (Jupyter):** Use `.viewer()` on a `Page` object in a Jupyter environment (Notebook, Lab, VS Code). This requires installing `natural-pdf[interactive]`.
2. **Static Image:** Use `.save_image()` on a `Page` object to save a static PNG file.

```python
# Add some highlights
page.find('text:contains("Summary")').highlight(label="Summary Title")
page.find_all('rect').highlight(color="blue", label="Boxes")

# --- View Interactively (Jupyter) ---
# Requires: pip install natural-pdf[interactive]
page.viewer()

# --- Save Static Image ---
# Include a legend for labeled highlights
page.save_image("highlighted_page.png", labels=True)
```

## Customizing Persistent Highlights

Customize the appearance of persistent highlights added with `.highlight()`:

```python
# Highlight with a specific color (string name, hex, or RGB/RGBA tuple)
title.highlight(color=(1, 0, 0, 0.3))  # Red with 30% opacity
title.highlight(color="#FF0000")        # Hex color
title.highlight(color="red")           # Color name

# Add a label to the highlight (appears in legend)
title.highlight(label="Title")

# Combine color and label
table = page.find('rect[width>=400][height>=200]')
table.highlight(color=(0, 0, 1, 0.2), label="Table")

# Save with a legend that shows the labels
page.viewer() # Or view interactively
```

## Highlighting Multiple Elements

Highlighting an `ElementCollection` applies the highlight to all elements within it. By default, all elements in the collection get the same color and a label based on their type.

```python
# Find and highlight all headings with a single color/label
headings = page.find_all('text[size>=14]:bold')
headings.highlight(color=(0, 0.5, 0, 0.3), label="Headings")

# Find and highlight all tables
tables = page.find_all('region[type=table]')
tables.highlight(color=(0, 0, 1, 0.2), label="Tables")

# View the result
page.viewer()
```

## Highlighting Regions

You can highlight regions to see what area you're working with:

```python
# Find a title and create a region below it
title = page.find('text:contains("Introduction")')
content = title.below(height=200)

# Highlight the region
content.highlight(color=(0, 0.7, 0, 0.2), label="Introduction")

# Highlight region boundaries
content.highlight(label="Region Boundary")

# Extract a cropped image of just this region
region_image = content.to_image(resolution=150)
content.save_image("region.png")
```

## Working with Text Styles

Visualize text styles to understand the document structure:

```python
# Analyze and highlight text styles
styles = page.analyze_text_styles()
page.highlight_text_styles()
page.save_image("text_styles.png", labels=True)

# Work with a specific style
if "Text Style 1" in styles:
    title_style = styles["Text Style 1"]
    title_style.highlight(color=(1, 0, 0, 0.3), label="Title Style")
```

## Displaying Attributes

You can display element attributes directly on the highlights:

```python
# Show confidence scores for OCR text
ocr_text = page.find_all('text[source=ocr]')
ocr_text.highlight(include_attrs=['confidence'])

# Show region types and confidence for layout analysis
regions = page.find_all('region')
regions.highlight(include_attrs=['region_type', 'confidence'])

# Show font information for text
text = page.find_all('text[size>=12]')
text.highlight(include_attrs=['fontname', 'size'])
```

## Clearing Highlights

You can clear persistent highlights from a page:

```python
# Clear all highlights on the page
page.clear_highlights()

# Apply new highlights
page.find_all('text:bold').highlight(label="Bold Text")
page.viewer()
```

## Composite Highlighting

You can build up complex visualizations layer by layer:

```python
# Clear any existing highlights
page.clear_highlights()

# Highlight different elements with different colors
page.find_all('text:bold').highlight(color=(1, 0, 0, 0.3), label="Bold Text")
page.find_all('text:contains("Table")').highlight(color=(0, 0, 1, 0.3), label="Table References")
page.find_all('line').highlight(color=(0, 0.5, 0, 0.3), label="Lines")

# Highlight regions
title = page.find('text:contains("Summary")')
if title:
    title.below(height=200).highlight(color=(0.5, 0, 0.5, 0.1), label="Summary Section")

# Save the composite image
page.save_image("composite_highlight.png", labels=True)
```

## OCR Visualization

Visualize OCR results with confidence levels:

```python
# Apply OCR first
ocr_elements = page.apply_ocr(engine='easyocr')

# Highlight OCR elements by confidence level (using group_by)
# (This generates a temporary preview image)
ocr_confidence_preview = page.find_all('text[source=ocr]').show(group_by=lambda el: f"Conf >= {0.8 if el.confidence >= 0.8 else (0.5 if el.confidence >= 0.5 else 0.0):.1f}")
ocr_confidence_preview

# --- Alternatively, add persistent highlights by confidence ---
# page.clear_highlights() # Optional: Clear previous highlights
# high_conf = page.find_all('text[source=ocr][confidence>=0.8]')
# med_conf = page.find_all('text[source=ocr][confidence>=0.5][confidence<0.8]')
# low_conf = page.find_all('text[source=ocr][confidence<0.5]')
# high_conf.highlight(color=(0, 1, 0, 0.3), label="High Confidence")
# med_conf.highlight(color=(1, 1, 0, 0.3), label="Medium Confidence")
# low_conf.highlight(color=(1, 0, 0, 0.3), label="Low Confidence")
# page.viewer()

# Save the visualization (if using persistent highlights)
# page.save_image("ocr_confidence.png", labels=True)
```

## Document QA Visualization

Visualize document QA results:

```python
# Ask a question to the document
result = page.ask("What is the total revenue?")

if result.get("found", False):
    # Highlight the answer source elements
    if "source_elements" in result:
        for element in result["source_elements"]:
            element.highlight(color=(1, 0.5, 0, 0.3), label="Answer")
            
    # Add the question and answer as an annotation
    question = "What is the total revenue?"
    answer = result["answer"]
    confidence = result["confidence"]
    
    # Save the highlighted image
    page.save_image("qa_visualization.png", labels=True)
```

## Next Steps

Now that you know how to visualize PDF content, you might want to explore:

- [OCR capabilities](../ocr/index.md) for working with scanned documents
- [Layout analysis](../layout-analysis/index.md) for automatic structure detection
- [Document QA](../document-qa/index.md) for asking questions directly to your documents