# Visual Debugging

Sometimes it's hard to understand what's happening when working with PDFs. Natural PDF provides powerful visual debugging tools to help you see what you're extracting.

## Basic Highlighting

The simplest way to highlight elements is with the `highlight()` method:

```python
from natural_pdf import PDF

pdf = PDF('document.pdf')
page = pdf.pages[0]

# Find a specific element and highlight it
title = page.find('text:contains("Summary")')
title.highlight()

# Show or save the highlighted page
page.show()  # Returns a PIL Image that displays in notebooks
page.save_image("highlighted.png")
```

## Customizing Highlights

You can customize your highlights with colors and labels:

```python
# Highlight with a specific color (RGBA tuple)
title.highlight(color=(1, 0, 0, 0.3))  # Red with 30% opacity

# Add a label to the highlight
title.highlight(label="Title")

# Combine color and label
table = page.find('rect[width>=400][height>=200]')
table.highlight(color=(0, 0, 1, 0.2), label="Table")

# Save with a legend that shows the labels
page.save_image("highlighted_with_legend.png", labels=True)
```

## Highlighting Multiple Elements

You can highlight multiple elements at once:

```python
# Find and highlight all headings
headings = page.find_all('text[size>=14]:bold')
headings.highlight(color=(0, 0.5, 0, 0.3), label="Headings")

# Find and highlight all tables
tables = page.find_all('region[type=table]')
tables.highlight(color=(0, 0, 1, 0.2), label="Tables")

# Save the image with all highlights
page.save_image("multiple_highlights.png", labels=True)
```

## Highlight All Elements

The `highlight_all()` method is great for quickly seeing all elements on a page:

```python
# Highlight all elements on the page
page.highlight_all()

# Save the image
page.save_image("all_elements.png", labels=True)

# Highlight only specific types of elements
page.highlight_all(include_types=['text', 'line'])

# Include text styles in the highlighting
page.highlight_all(include_text_styles=True)

# Include layout regions in the highlighting
page.highlight_all(include_layout_regions=True)
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

You can clear highlights when needed:

```python
# Clear all highlights
page.clear_highlights()

# Apply new highlights
page.find_all('text:bold').highlight(label="Bold Text")
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
# Enable OCR
pdf = PDF('scanned_document.pdf', ocr=True)
page = pdf.pages[0]

# Apply OCR
ocr_elements = page.apply_ocr()

# Highlight OCR elements by confidence level
high_conf = page.find_all('text[source=ocr][confidence>=0.8]')
med_conf = page.find_all('text[source=ocr][confidence>=0.5][confidence<0.8]')
low_conf = page.find_all('text[source=ocr][confidence<0.5]')

high_conf.highlight(color=(0, 1, 0, 0.3), label="High Confidence")
med_conf.highlight(color=(1, 1, 0, 0.3), label="Medium Confidence")
low_conf.highlight(color=(1, 0, 0, 0.3), label="Low Confidence")

# Save the visualization
page.save_image("ocr_confidence.png", labels=True)
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