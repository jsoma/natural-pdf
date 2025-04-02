# Layout Analysis

Beyond simple text and lines, `natural-pdf` can use layout analysis models (like YOLO or DETR) to identify semantic regions within a page, such as paragraphs, tables, figures, headers, etc. This provides a higher-level understanding of the document structure.

Let's analyze the layout of our `01-practice.pdf`.

```python
#%pip install "natural-pdf[all]"
```

```python
from natural_pdf import PDF

# Load the PDF and get the page
pdf = PDF("https://github.com/jsoma/natural-pdf/raw/refs/heads/main/pdfs/01-practice.pdf")
page = pdf.pages[0]

# Analyze the layout using the default model
# This adds 'detected' Region objects to the page
# It returns an ElementCollection of the detected regions
page.analyze_layout()
detected_regions = page.find_all('region[source="detected"]')
```

```python
# Visualize all detected regions, using default colors based on type
page.clear_highlights() # Clear previous highlights
detected_regions.highlight(group_by='type', include_attrs=['confidence'])

# Show the image with region overlays
page.to_image(width=900)
```

```python
# Find and visualize only the detected table region(s)
tables = page.find_all('region[type=table]')
tables.show(color='lightgreen', label='Detected Table')
```

```python
# Extract text specifically from the detected table region
table_region = tables.first # Assuming only one table was detected
# Extract text preserving layout
table_text_layout = table_region.extract_text(layout=True)
table_text_layout
```

```python
# Layout-detected regions can also be used for table extraction
# This can be more robust than the basic page.extract_tables()
# especially for tables without clear lines.
table_data = table_region.extract_table()
table_data
```

Layout analysis provides structured `Region` objects. You can filter these regions by their predicted `type` and then perform actions like visualization or extracting text/tables specifically from those regions.

<div class="admonition note">
<p class="admonition-title">Layout Models and Configuration</p>

    *   Layout analysis requires external models. Ensure these are installed.
    *   You can specify different models (`engine='yolo'`, `engine='detr'`, `engine='paddle'`) or configurations (confidence thresholds, specific classes) via arguments to `page.analyze_layout()`. Different models may perform better on different document types.
    *   The detected regions are added to the page and can be found using selectors like `page.find_all('region[type=paragraph]')`.
</div>

``` 