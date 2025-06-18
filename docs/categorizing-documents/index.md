# Categorizing Pages and Regions

Natural PDF allows you to automatically categorize pages or specific regions within a page using machine learning models. This is incredibly useful for filtering large collections of documents or understanding the structure and content of individual PDFs.

## Installation

To use the classification features, you need to install the optional dependencies:

```bash
pip install "natural-pdf[ai]"
```

This installs necessary libraries like `torch`, `transformers`, and others.

## Core Concept: The `.classify()` Method

The primary way to perform categorization is using the `.classify()` method available on `Page` and `Region` objects.

```python
from natural_pdf import PDF

# Example: Classify a Page
pdf = PDF("https://github.com/jsoma/natural-pdf/raw/refs/heads/main/pdfs/01-practice.pdf")
page = pdf.pages[0]
labels = ["invoice", "letter", "report cover", "data table"]
page.classify(labels, using="text")

# Access the top result
print(f"Top Category: {page.category}")
print(f"Confidence: {page.category_confidence:.3f}")
```

**Key Arguments:**

*   `labels` (required): A list of strings representing the potential labels you want to classify the item into.
*   `using` (optional): Specifies which classification model or strategy to use. Defaults to `"text"`.
    *   `"text"`: Uses a text-based model (default: `facebook/bart-large-mnli`) suitable for classifying based on language content.
    *   `"vision"`: Uses a vision-based model (default: `openai/clip-vit-base-patch32`) suitable for classifying based on visual layout and appearance.
    *   Specific Model ID: You can provide a Hugging Face model ID (e.g., `"google/siglip-base-patch16-224"`, `"MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"`) compatible with zero-shot text or image classification. The library attempts to infer whether it's text or vision, but you might need `using`.
*   `model` (optional): Explicitly model ID (HuggingFace repo name)
*   `min_confidence` (optional): A float between 0.0 and 1.0. Only labels with a confidence score greater than or equal to this threshold will be included in the results (default: 0.0).

## Text vs. Vision Classification

Choosing the right model type depends on your goal:

### Text Classification (`using="text"`)

*   **How it works:** Extracts the text from the page or region and analyzes the language content.
*   **Best for:**
    *   **Topic Identification:** Determining what a page or section is *about* (e.g., "budget discussion," "environmental impact," "legal terms").
    *   **Content-Driven Document Types:** Identifying document types primarily defined by their text (e.g., emails, meeting minutes, news articles, reports).
*   **Data Journalism Example:** You have thousands of pages of government reports. You can use text classification to find all pages discussing "public health funding" or classify paragraphs within environmental impact statements to find mentions of specific endangered species.

```python
# Find pages related to finance
financial_labels = ["budget", "revenue", "expenditure", "forecast"]
pdf.classify_pages(financial_labels, using="text")
budget_pages = [p for p in pdf.pages if p.category == "budget"]
```

### Vision Classification (`using="vision"`)

*   **How it works:** Renders the page or region as an image and analyzes its visual layout, structure, and appearance.
*   **Best for:**
    *   **Layout-Driven Document Types:** Identifying documents recognizable by their structure (e.g., invoices, receipts, forms, presentation slides, title pages).
    *   **Identifying Visual Elements:** Distinguishing between pages dominated by text, tables, charts, or images.
*   **Data Journalism Example:** You have a scanned archive of campaign finance filings containing various document types. You can use vision classification to quickly isolate all the pages that look like donation receipts or expenditure forms, even if the OCR quality is poor.

```python
# Find pages that look like invoices or receipts
visual_labels = ["invoice", "receipt", "letter", "form"]
page.classify(visual_labels, using="vision")
if page.category in ["invoice", "receipt"]:
    print(f"Page {page.number} looks like an invoice or receipt.")
```

## Classifying Specific Objects

### Pages (`page.classify(...)`)

Classifying a whole page is useful for sorting documents or identifying the overall purpose of a page within a larger document.

```python
# Classify the first page
page = pdf.pages[0]
page_types = ["cover page", "table of contents", "chapter start", "appendix"]
page.classify(page_types, using="vision") # Vision often good for page structure
print(f"Page 1 Type: {page.category}")
```

### Regions (`region.classify(...)`)

Classifying a specific region allows for more granular analysis within a page. You might first detect regions using Layout Analysis and then classify those regions.

```python
# Assume layout analysis has run, find paragraphs
paragraphs = page.find_all("region[type=paragraph]")
if paragraphs:
    # Classify the topic of the first paragraph
    topic_labels = ["introduction", "methodology", "results", "conclusion"]
    # Use text model for topic
    paragraphs[0].classify(topic_labels, using="text")
    print(f"First paragraph category: {paragraphs[0].category}")
```

## Accessing Classification Results

After running `.classify()`, you can access the results:

*   `page.category` or `region.category`: Returns the string label of the category with the highest confidence score from the *last* classification run. Returns `None` if no classification has been run or no category met the threshold.
*   `page.category_confidence` or `region.category_confidence`: Returns the float confidence score (0.0-1.0) for the top category. Returns `None` otherwise.
*   `page.classification_results` or `region.classification_results`: Returns the full result dictionary stored in the object's `.metadata['classification']`, containing the model used, engine type, labels provided, timestamp, and a list of all scores above the threshold sorted by confidence. Returns `None` if no classification has been run.

```python
results = page.classify(["invoice", "letter"], using="text", min_confidence=0.5)

if page.category == "invoice":
    print(f"Found an invoice with confidence {page.category_confidence:.2f}")

# See all results above the threshold
# print(page.classification_results['scores'])
```

## Classifying Collections

For batch processing, use the `.classify_all()` method on `PDFCollection` or `ElementCollection` objects. This displays a progress bar tracking individual items (pages or elements).

### PDFCollection (`collection.classify_all(...)`)

Classifies pages across all PDFs in the collection. Use `max_workers` for parallel processing across different PDF files.

```python
collection = natural_pdf.PDFCollection.from_directory("./documents/")
labels = ["form", "datasheet", "image", "text document"]

# Classify all pages using vision model, processing 4 PDFs concurrently
collection.classify_all(labels, using="vision", max_workers=4)

# Filter PDFs containing forms
form_pdfs = []
for pdf in collection:
    if any(p.category == "form" for p in pdf.pages if p.category):
        form_pdfs.append(pdf.path)
    pdf.close() # Remember to close PDFs

print(f"Found forms in: {form_pdfs}")
```

### ElementCollection (`element_collection.classify_all(...)`)

Classifies all classifiable elements (currently `Page` and `Region`) within the collection.

```python
# Assume 'pdf' is loaded and 'layout_regions' is an ElementCollection of Regions
layout_regions = pdf.find_all("region")
region_types = ["paragraph", "list", "table", "figure", "caption"]

# Classify all detected regions based on vision
layout_regions.classify_all(region_types, model="vision")

# Count table regions using filter()
table_regions = layout_regions.filter(lambda region: region.category == "table")
print(f"Found {len(table_regions)} regions classified as tables.")
```
