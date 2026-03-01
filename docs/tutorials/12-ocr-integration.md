# OCR Integration for Scanned Documents

Optical Character Recognition (OCR) allows you to extract text from scanned documents where the text isn't embedded in the PDF. This tutorial demonstrates how to work with scanned documents.

```python
#%pip install "natural-pdf[all]"
```

```python
from natural_pdf import PDF

# Load a PDF
pdf = PDF("https://github.com/jsoma/natural-pdf/raw/refs/heads/main/pdfs/needs-ocr.pdf")
page = pdf.pages[0]

# Try extracting text without OCR
text_without_ocr = page.extract_text()
f"Without OCR: {len(text_without_ocr)} characters extracted"
```

## Applying OCR and Finding Elements

The core method is `page.apply_ocr()`. This runs the OCR process and adds `TextElement` objects to the page. You can specify the engine and languages.

**Note:** Re-applying OCR to the same page or region will automatically remove any previously generated OCR elements for that area before adding the new ones.

**Tip:** After applying OCR, you can use [spatial navigation](08-spatial-navigation.md) to extract values relative to labels. For example, find "Total:" with OCR, then use `.right()` to get the value next to it.

```python
# Apply OCR using the default engine (EasyOCR) for English
page.apply_ocr(languages=['en'])

# Select all text pieces found by OCR
text_elements = page.find_all('text[source=ocr]')
print(f"Found {len(text_elements)} text elements using default OCR")

# Visualize the elements
text_elements.show()
```

## Visualizing OCR Confidence Scores

OCR engines provide confidence scores for each detected text element. You can visualize these scores using gradient colors to quickly identify areas that may need attention:

```python
# Visualize confidence scores with gradient colors (auto-detected as quantitative)
text_elements.show(group_by='confidence')

# Use different colormaps for better visualization
text_elements.show(group_by='confidence', color='viridis')  # Blue to yellow
text_elements.show(group_by='confidence', color='plasma')   # Purple to yellow
text_elements.show(group_by='confidence', color='RdYlBu')   # Red-yellow-blue

# Focus on a specific confidence range
text_elements.show(group_by='confidence', bins=[0.3, 0.8])  # Only show 0.3-0.8 range

# Create custom bins for confidence levels
text_elements.show(group_by='confidence', bins=[0, 0.5, 0.8, 1.0])  # Low/medium/high
```

This makes it easy to spot low-confidence OCR results that might need manual review or correction. You'll automatically get a color scale showing the confidence range instead of a discrete legend.

```python
# Apply OCR using PaddleOCR for English
page.apply_ocr(engine='paddle', languages=['en'])
print(f"Found {len(page.find_all('text[source=ocr]'))} elements after English OCR.")

# Apply OCR using PaddleOCR for Chinese
page.apply_ocr(engine='paddle', languages=['zh'])
print(f"Found {len(page.find_all('text[source=ocr]'))} elements after Chinese OCR.")

text_with_ocr = page.extract_text()
print(f"\nExtracted text after OCR:\n{text_with_ocr[:150]}...")
```

You can also use `.describe()` to see a summary of the OCR outcome...

```python
page.describe()
```

...or `.inspect()` on the text elements for individual details.

```python
page.find_all('text').inspect()
```

## Setting Default OCR Options

You can set global default OCR options using `natural_pdf.options`. These defaults will be used automatically when you call `apply_ocr()` without specifying parameters.

```python
import natural_pdf as npdf

# Set global OCR defaults
npdf.options.ocr.engine = 'surya'          # Default OCR engine
npdf.options.ocr.min_confidence = 0.7      # Default confidence threshold

# Now all OCR calls use these defaults
pdf = npdf.PDF("https://github.com/jsoma/natural-pdf/raw/refs/heads/main/pdfs/needs-ocr.pdf")
pdf.pages[0].apply_ocr()  # Uses: engine='surya', min_confidence=0.7

# You can still override defaults for specific calls
pdf.pages[0].apply_ocr(engine='easyocr', languages=['fr'])  # Override engine and languages
```

This is especially useful when processing many documents with the same OCR settings, as you don't need to specify the parameters repeatedly.

## Advanced OCR Configuration

For more control, import and use the specific `Options` class for your chosen engine within the `apply_ocr` call.

```python
from natural_pdf.ocr import PaddleOCROptions, EasyOCROptions, SuryaOCROptions, RapidOCROptions

# Re-apply OCR using EasyOCR with specific options
easy_opts = EasyOCROptions(
    paragraph=False,
)
page.apply_ocr(engine='easyocr', languages=['en'], min_confidence=0.1, options=easy_opts)

paddle_opts = PaddleOCROptions()
page.apply_ocr(engine='paddle', languages=['en'], options=paddle_opts)

surya_opts = SuryaOCROptions()
page.apply_ocr(engine='surya', languages=['en'], min_confidence=0.5, options=surya_opts)
```

RapidOCR is a lightweight alternative that uses ONNX-converted PaddleOCR models (~15MB vs ~500MB):

```python
# Lightweight OCR — no heavy framework needed
page.apply_ocr(engine='rapidocr', languages=['en'])
```

**Install:** `pip install rapidocr`.

## PaddleOCR-VL (VLM-Based OCR)

PaddleOCR-VL uses a Vision Language Model for document understanding. It can handle complex layouts, charts, and mixed content better than traditional OCR.

```python
from natural_pdf.ocr import PaddleOCRVLOptions

# Basic usage
page.apply_ocr(engine='paddlevl')

# With options
vl_opts = PaddleOCRVLOptions(
    use_layout_detection=True,
    use_chart_recognition=True,
)
page.apply_ocr(engine='paddlevl', options=vl_opts)
```

**Install:** `pip install paddlepaddle paddleocr "paddlex[ocr]"`.

## VLM-Based OCR

You can use any OpenAI-compatible vision model for OCR. Pass `engine="vlm"` with a `model` and `client`. The model returns grounded bounding boxes with text — best results come from Qwen-VL family models.

```python
from openai import OpenAI

client = OpenAI()  # or any OpenAI-compatible client

page.apply_ocr(
    engine="vlm",
    model="Qwen/Qwen3-VL-2B-Instruct",
    client=client,
)
```

Gemini models also work:

```python
from openai import OpenAI

client = OpenAI(
    api_key="your-api-key",
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

page.apply_ocr(engine="vlm", model="gemini-2.5-flash", client=client)
```

Use `instructions` to append hints to the auto-generated prompt (e.g., expected language or document type):

```python
page.apply_ocr(
    engine="vlm",
    model="gemini-2.5-flash",
    client=client,
    instructions="The text is from a Greek legal document."
)
```

### Detect Then Correct with VLM

A common pattern: use a fast engine to detect text locations, then use a VLM to correct the text per-element. This combines fast detection with high-quality recognition.

```python
from openai import OpenAI

client = OpenAI(
    api_key="your-api-key",
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# Step 1: detect text bounding boxes (no recognition)
page.apply_ocr(detect_only=True)

# Step 2: correct each detected element with a VLM
page.find_all('text').apply_ocr(
    engine="vlm",
    model="gemini-2.5-flash-lite",
    client=client,
    instructions="Return only the exact text visible. Fix OCR misspellings."
)
```

When called on an `ElementCollection` of OCR elements with a VLM, each element is rendered individually and sent to the model for correction. The original bounding boxes are preserved — only the text is updated.

The `detect_only=True` parameter runs detection without recognition. This is useful when you want a fast engine (EasyOCR, Surya) to find where text is, then a separate step to read it.

## Language Codes

All OCR engines accept standard ISO language codes like `'en'`, `'fr'`, `'de'`, `'ja'`, `'zh'`, `'ko'`. PaddleOCR automatically normalizes these to its internal format (e.g. `'ja'` becomes `'japan'`, `'zh'` becomes `'ch'`), so you don't need to look up engine-specific codes.

```python
# Standard codes work across all engines
page.apply_ocr(engine='easyocr', languages=['ja'])
page.apply_ocr(engine='paddle', languages=['ja'])   # auto-normalized to 'japan'
page.apply_ocr(engine='surya', languages=['ja'])
```

## Interactive OCR Correction / Debugging

If OCR results aren't perfect, you can use the bundled interactive web application (SPA) to review and correct them.

1.  **Package the data:**
    After running `apply_ocr` (or `apply_layout`), use `create_correction_task_package` to create a zip file containing the PDF images and detected elements.

    ```python
    from natural_pdf.utils.packaging import create_correction_task_package

    page.apply_ocr()

    create_correction_task_package(pdf, "correction_package.zip", overwrite=True)
    ```

2.  **Run the SPA:**
    The correction app is bundled with the library. Start a local server pointing to it:

    ```bash
    python -m http.server 8000 -d "$(python -c 'import natural_pdf; import os; print(os.path.join(os.path.dirname(natural_pdf.__file__), "spa"))')"
    ```

3.  **Use the SPA:**
    Open `http://localhost:8000` in your browser. Drag the `correction_package.zip` file onto the page to load the document. You can then click on text elements to correct the OCR results.


## Working with Multiple Pages

Apply OCR or layout analysis to all pages using the `PDF` object.

```python
# Process all pages in the document

# Apply OCR to all pages (example using EasyOCR)
pdf.apply_ocr(engine='easyocr', languages=['en'])
print(f"Applied OCR to {len(pdf.pages)} pages.")

# Or apply layout analysis to all pages (example using Paddle)
# pdf.apply_layout(engine='paddle')
# print(f"Applied Layout Analysis to {len(pdf.pages)} pages.")

# Extract text from all pages (uses OCR results if available)
all_text_content = pdf.extract_text(page_separator="\\n\\n---\\n\\n")

print(f"\nCombined text from all pages:\n{all_text_content[:500]}...")
```

## Saving PDFs with Searchable Text

After applying OCR to a PDF, you can save a new version with the recognized text embedded as an invisible layer. This makes the text searchable and copyable in standard PDF viewers.

```python
pdf.save_pdf("searchable_output.pdf", ocr=True)
```

**Install:** `pip install "natural-pdf[export]"` (requires pikepdf and img2pdf).

## Combining OCR with Spatial Navigation

After OCR, use spatial navigation to extract structured data from scanned documents. This is especially useful for forms and invoices.

```python
from natural_pdf import PDF

pdf = PDF("scanned_invoice.pdf")
page = pdf.pages[0]

# Apply OCR first
page.apply_ocr(engine='easyocr', languages=['en'])

# Now use spatial navigation to extract values
# Find a label and get the value to its right
total_label = page.find('text:contains("Total:")')
if total_label:
    total_value = total_label.right(width=150).extract_text().strip()
    print(f"Total: {total_value}")

# Extract multiple fields
fields = {}
for label_text in ["Invoice #:", "Date:", "Amount Due:"]:
    label = page.find(f'text:contains("{label_text}")')
    if label:
        value = label.right(width=200).extract_text().strip()
        fields[label_text.rstrip(":")] = value

print(fields)
```

See [Tutorial 08: Spatial Navigation](08-spatial-navigation.md) for more techniques.

## Detecting Checkboxes

Scanned forms often contain checkboxes. The `detect_checkboxes()` method finds them and determines whether each one is checked or unchecked.

```python
from natural_pdf import PDF

pdf = PDF("https://github.com/jsoma/natural-pdf/raw/refs/heads/main/pdfs/01-practice.pdf")
page = pdf.pages[0]

# Detect checkboxes
checkboxes = page.detect_checkboxes()
print(f"Found {len(checkboxes)} checkboxes")

for cb in checkboxes:
    print(f"  {'[x]' if cb.is_checked else '[ ]'} confidence={cb.confidence:.2f}")
```

The default engine uses a YOLO12n model that detects and classifies checkboxes in a single pass — no separate classification step needed. It works on both vector PDFs and scanned images.

```python
# Visualize detected checkboxes
checkboxes = page.find_all('region[type=checkbox]')
checkboxes.show(group_by='checkbox_state')
```

### Combining with Spatial Navigation

You can use spatial navigation to find the label next to each checkbox:

```python
for cb in page.detect_checkboxes():
    label = cb.right(width=200).extract_text().strip()
    status = "checked" if cb.is_checked else "unchecked"
    print(f"  {status}: {label}")
```

### Engine Options

Multiple detection engines are available:

* **default** — YOLO12n 2-class model (checked/unchecked). Best accuracy, used automatically.
* **vector** — Detects checkbox-shaped rectangles in native PDF markup. Instant, no rendering needed. Tried first in auto mode for vector PDFs.
* **onnx** — Generic ONNX engine for custom YOLO-format models from HuggingFace.

```python
# Use a specific engine
page.detect_checkboxes(engine="vector")

# Adjust confidence threshold
page.detect_checkboxes(confidence=0.5)

# Use a custom ONNX model from HuggingFace
page.detect_checkboxes(engine="onnx", model="some-user/some-model")
```

## Correcting OCR Results

If OCR results aren't perfect, you can programmatically correct them using `correct_ocr()`. It takes a callback function that receives each OCR element and returns corrected text.

```python
from openai import OpenAI
from natural_pdf import PDF

client = OpenAI()

pdf = PDF("scanned_document.pdf")
page = pdf.pages[0]
page.apply_ocr()

# Define a correction function using an LLM
def correct_text(region):
    text = region.extract_text()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Correct the spelling of this OCR'd text. Preserve original formatting."},
            {"role": "user", "content": text},
        ],
    )
    return response.choices[0].message.content

# Apply correction to all OCR'd text on the page
page.correct_ocr(correct_text)
```

For difficult documents, use a vision model to re-OCR specific regions instead of correcting text:

```python
from natural_pdf.ocr.utils import direct_ocr_llm

def correct_with_vision(region):
    return direct_ocr_llm(
        region,
        client,
        prompt="OCR this image patch. Return only the exact text content visible.",
        resolution=150,
        model="gpt-4o"
    )

page.correct_ocr(correct_with_vision)
```

## Deskewing Scanned Pages

Scanned documents are often slightly rotated. You can detect and correct skew before or after OCR.

```python
from natural_pdf import PDF

pdf = PDF("skewed_scan.pdf")
page = pdf.pages[0]

# Detect the skew angle (in degrees)
angle = page.detect_skew_angle()
print(f"Skew angle: {angle:.2f}°")

# Get a deskewed image of the page
deskewed_image = page.deskew()
```

To create a new deskewed PDF (image-based):

```python
# Deskew all pages and get a new PDF
deskewed_pdf = pdf.deskew()

# Or deskew specific pages
deskewed_pdf = pdf.deskew(pages=[0, 1, 2])

# Save the result
deskewed_pdf.save_pdf("deskewed_output.pdf")
```

Detection engines: `"projection"` (default), `"hough"`, or `"standard"`.

**Note:** `pdf.deskew()` returns an image-based PDF. Text, OCR results, and annotations from the original are not carried over — apply OCR again on the deskewed result if needed.

**Install:** `pip install "natural-pdf[export]"` (requires img2pdf for PDF output).
