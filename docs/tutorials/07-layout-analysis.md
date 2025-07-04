# Layout Analysis

Beyond simple text and lines, `natural-pdf` can use layout analysis models (like YOLO or DETR) to identify semantic regions within a page, such as paragraphs, tables, figures, headers, etc. This provides a higher-level understanding of the document structure.

## Available Layout Engines

* **yolo** – YOLOv5 model trained on DocLayNet; fast and good at classic page objects (paragraph, table, figure, heading).  Install via `npdf install yolo`.
* **tatr** – Microsoft's Table Transformer (LayoutLM) specialised in tables; already included in the **ai** extra.
* **paddle** – PaddleOCR`s layout detector; lightweight and CPU-friendly.
* **surya** – Surya Layout Parser (DETR backbone) tuned for invoices and forms.
* **docling** – YOLOX model published by DocLING researchers; performs well on historical documents.
* **gemini** – Calls Google's Vision Gemini API (experimental, requires `OPENAI_API_KEY`).

`page.analyze_layout()` defaults to the first available engine (search order `yolo → paddle → tatr`), but you can pick one explicitly with `engine="..."`.

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
detected_regions.show(group_by='type', annotate=['confidence'])
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

## Switching Engines and Tuning Thresholds

```python
# Re-run layout with PaddleOCR detector
page.clear_detected_layout_regions()

paddle_regions = page.analyze_layout(engine="paddle", confidence=0.3)
#paddle_regions.show(group_by="type")

# Only keep detections the model tagged as "table" or "figure"
tables_and_figs = paddle_regions.filter(lambda r: r.normalized_type in {"table", "figure"})
#tables_and_figs.show(label_format="{normalized_type} ({confidence:.2f})")
```

The helper accepts these common kwargs (see `LayoutOptions` subclasses for full list):

* `confidence` – minimum score for retaining a prediction.
* `classes` / `exclude_classes` – whitelist or blacklist region types.
* `device` – "cuda" or "cpu"; defaults to GPU if available.

Each engine also exposes its own options class (e.g., `YOLOLayoutOptions`) for fine control over NMS thresholds, model sizes, etc. Pass an instance via the `options=` param.

Layout analysis provides structured `Region` objects. You can filter these regions by their predicted `type` and then perform actions like visualization or extracting text/tables specifically from those regions.

## TODO

* Add a speed/accuracy comparison snippet looping over all installed engines.
* Demonstrate multi-page batch: `pdf.pages[::2].analyze_layout(engine="yolo")`.
* Show `page.get_sections(start_elements=page.find_all('region[type=heading]'))` to split by detected headings.
* Include an example of exporting regions to COCO JSON for custom model fine-tuning.
* Document how to override the model path via `model_name` and how to plug a remote inference client (`client=`).

## Wish List (Future Enhancements)

* **Confidence palette** – Allow `show(color_by="confidence")` to auto-map scores to a red–green gradient.
* **`ElementCollection.to_json()`** – one-liner export of detected regions (and optionally `to_df()`).
* **Model cache override** – honor an env variable like `NATPDF_MODEL_DIR` so enterprises can redirect weight downloads.
* **Remote inference support** – make the `client=` hook forward images to a custom REST or gRPC service.
