# Document Layout Analysis

Natural PDF can automatically detect the structure of a document (titles, paragraphs, tables, figures) using layout analysis models. This guide shows how to use this feature.

## Setup

We'll use a sample PDF that includes various layout elements.

```python
from natural_pdf import PDF

pdf = PDF("https://github.com/jsoma/natural-pdf/raw/refs/heads/main/pdfs/01-practice.pdf")
page = pdf.pages[0]

page.show(width=700)
```

## Running Basic Layout Analysis

Use the `analyze_layout()` method. By default, it uses the YOLO model.

```python
# Analyze the layout using the default engine (YOLO)
# This adds 'region' elements to the page
page.analyze_layout()
```

```python
# Find all detected regions
regions = page.find_all('region')
len(regions) # Show how many regions were detected
```

```python
first_region = regions[0]
f"First region: type='{first_region.type}', confidence={first_region.confidence:.2f}"
```

## Visualizing Detected Layout

Use `show()` on the detected regions.

```python
# Highlight all detected regions, colored by type
regions.show(group_by='type', width=700)
```

## Finding Specific Region Types

Use attribute selectors to find regions of a specific type.

```python
# Find all detected titles
titles = page.find_all('region[type=title]')
titles
```

```python
titles.show()
```

```python
page.find_all('region[type=table]').show()
```

## Working with Layout Regions

Detected regions are like any other `Region` object. You can extract text, find elements within them, etc.

```python
page.find('region[type=table]').extract_text(layout=True)
```

## Using Different Layout Models

Natural PDF supports multiple engines (`yolo`, `paddle`, `tatr`). Specify the engine when calling `analyze_layout`.

*Note: Using different engines requires installing the corresponding extras (e.g., `natural-pdf[layout_paddle]`).* `yolo` is the default.

```python
page.clear_detected_layout_regions()
page.clear_highlights()

page.analyze_layout(engine="paddle")
page.find_all('region[model=paddle]').show(group_by='region_type', width=700)
```

```python
# Analyze using Table Transformer (TATR) - specialized for tables
page.clear_detected_layout_regions()
page.clear_highlights()

page.analyze_layout(engine="tatr")
page.find_all('region[model=tatr]').show(group_by='region_type', width=700)
```

```python
# Analyze using Docling
# https://docling-project.github.io/docling/

# Docling has been weird, it's not included at the moment

# page.clear_detected_layout_regions()
# page.clear_highlights()

# page.analyze_layout(engine="docling")
# page.find_all('region[model=docling]').show(group_by='region_type')
# page.render(width=700)
```

```python
# Analyze using Table Transformer (TATR) - specialized for tables
page.clear_detected_layout_regions()
page.clear_highlights()

page.analyze_layout(engine="surya")
page.find_all('region[model=surya]').show(group_by='region_type', width=700)
```

*Note: Calling `analyze_layout` multiple times (even with the same engine) can add duplicate regions. You might want to use `page.clear_detected_layout_regions()` first, or filter by model using `region[model=yolo]`.*

## Controlling Confidence Threshold

Filter detections by their confidence score.

```python
# Re-run YOLO analysis (clearing previous results might be good practice)
page.clear_detected_layout_regions()
page.analyze_layout(engine="yolo")

# Find only high-confidence regions (e.g., >= 0.8)
high_conf_regions = page.find_all('region[confidence>=0.8]')
len(high_conf_regions)
```

## Table Structure with TATR

The TATR engine provides detailed table structure elements (`table`, `table-row`, `table-column`, `table-column-header`). This is very useful for precise table extraction.

```python
# Ensure TATR analysis has been run
page.clear_detected_layout_regions()
page.clear_highlights()

page.analyze_layout(engine="tatr")
page.find_all('region[model=tatr]').show(group_by='region_type', width=700)
```

```python
# Find different structural elements from TATR
tables = page.find_all('region[type=table][model=tatr]')
rows = page.find_all('region[type=table-row][model=tatr]')
cols = page.find_all('region[type=table-column][model=tatr]')
hdrs = page.find_all('region[type=table-column-header][model=tatr]')

f"Found: {len(tables)} tables, {len(rows)} rows, {len(cols)} columns, {len(hdrs)} headers (from TATR)"
```

### Enhanced Table Extraction with TATR

When a `region[type=table]` comes from the TATR model, `extract_table()` can use the underlying row/column structure for more robust extraction.

```python
# Find the TATR table region again
tatr_table = page.find('region[type=table][model=tatr]')

# This extraction uses the detected rows/columns
tatr_table.extract_table()
```

if you'd like the normal approach instead of the "intelligent" one, you can ask for pdfplumber.

```python
# This extraction uses the detected rows/columns
tatr_table.extract_table(method='pdfplumber')
```

## Using Gemini for Layout Analysis (Advanced)

Natural PDF supports layout analysis using Google's Gemini models via an OpenAI-compatible API. This is an advanced feature and requires you to provide your own OpenAI client, API key, and endpoint.

**Example usage:**

```py
from openai import OpenAI
from natural_pdf import PDF
from natural_pdf.analyzers.layout.layout_options import GeminiLayoutOptions

# Create a compatible OpenAI client for Gemini
client = OpenAI(
    api_key="YOUR_GOOGLE_API_KEY",
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

options = GeminiLayoutOptions(
    model_name="gemini-2.0-flash",
    client=client,
    classes=["text", "title"]
)

pdf = PDF("https://github.com/jsoma/natural-pdf/raw/refs/heads/main/pdfs/01-practice.pdf")
page = pdf.pages[0]
regions = page.analyze_layout(engine="gemini", options=options)
```

- You must provide your own API key and endpoint for Gemini.
- The client must be compatible with the OpenAI API (see the `openai` Python package).
- This feature is intended for advanced users who need LLM-based layout analysis.

## Using Judge for Visual Classification

Natural PDF includes a `Judge` class that can learn to classify visual elements like checkboxes. This is particularly useful after layout detection when you need to determine the state of detected elements.

### Example: Checkbox Classification

```python
from natural_pdf import Judge

# Create a judge for checkbox classification
judge = Judge("form_checkboxes", labels=["checked", "unchecked"])

# Train with examples
checked_region = page.find("text=Acceptable").left(width=20)
unchecked_region = page.find("text=Deficient").left(width=20)

judge.add(checked_region, "checked")
judge.add(unchecked_region, "unchecked")

# Classify new checkboxes
new_checkbox = page.find("text=At-Risk").left(width=20)
result = judge.decide(new_checkbox)
print(f"Checkbox is: {result.label} (confidence: {result.score:.2f})")

# Find which checkbox is selected among multiple options
checkboxes = [
    page.find("text=Option A").left(width=20),
    page.find("text=Option B").left(width=20),
    page.find("text=Option C").left(width=20)
]
selected = judge.pick("checked", checkboxes, labels=["Option A", "Option B", "Option C"])
print(f"Selected: {selected.label}")
```

### Key Features of Judge

1. **Simple Training**: Requires minimal examples (even just one per class)
2. **Robust to Imbalance**: Uses Youden's J weights and prior correction
3. **Interactive Teaching**: Use `judge.teach()` in Jupyter for labeling
4. **Visual Inspection**: Use `judge.inspect()` to see predictions on training data
5. **Persistence**: Save/load trained judges with `judge.save()` and `Judge.load()`

### Advanced Usage

```python
# Adjust prior if you expect more checked boxes
judge = Judge("checkboxes", labels=["checked", "unchecked"], target_prior=0.7)

# Interactive teaching in Jupyter
judge.teach()  # Use arrow keys to label examples

# Visual inspection with previews
judge.inspect(preview=True)  # Shows HTML table with images

# Count checkboxes by type
checked_count = judge.count("checked", checkbox_regions)
```

## Next Steps

Layout analysis provides regions that you can use for:

- [Table Extraction](../tables/index.ipynb): Especially powerful with TATR regions.
- [Text Extraction](../text-extraction/index.ipynb): Extract text only from specific region types (e.g., paragraphs).
- [Document QA](../document-qa/index.ipynb): Focus question answering on specific detected regions.
- Visual Classification: Use Judge to classify detected elements (checkboxes, signatures, etc.)
