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
page.apply_ocr(engine='paddle', languages=['ch'])
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
pdf.pages[0].apply_ocr()  # Uses: engine='surya', languages=['en', 'es'], min_confidence=0.7

# You can still override defaults for specific calls
pdf.pages[0].apply_ocr(engine='easyocr', languages=['fr'])  # Override engine and languages
```

This is especially useful when processing many documents with the same OCR settings, as you don't need to specify the parameters repeatedly.

## Advanced OCR Configuration

For more control, import and use the specific `Options` class for your chosen engine within the `apply_ocr` call.

```python
from natural_pdf.ocr import PaddleOCROptions, EasyOCROptions, SuryaOCROptions

# Re-apply OCR using EasyOCR with specific options
easy_opts = EasyOCROptions(
    paragraph=False,
)
page.apply_ocr(engine='easyocr', languages=['en'], min_confidence=0.1, options=easy_opts)

paddle_opts = PaddleOCROptions()
page.apply_ocr(engine='paddle', languages=['en'], options=paddle_opts)

surya_opts = SuryaOCROptions()
page.apply_ocr(engine='surya', languages=['en'], min_confidence=0.5, detect_only=True, options=surya_opts)
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
    Navigate to the SPA directory within the installed `natural_pdf` library in your terminal and start a simple web server.

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

After applying OCR to a PDF, you can save a new version of the PDF where the recognized text is embedded as an invisible layer. This makes the text searchable and copyable in standard PDF viewers.

Use the `save_searchable()` method on the `PDF`

## TODO

* Add guidance on installing only the OCR engines you need (e.g. `pip install "natural-pdf[ai] easyocr"`) instead of the heavy `[all]` extra.
* Show how to use `detect_only=True` to combine OCR detection with external recognition for higher accuracy (ties into fine-tuning tutorial).
* Include an example of saving a searchable PDF via `pdf.save_searchable("output.pdf")` after OCR.
* Mention `resolution` parameter trade-offs (speed vs accuracy) when calling `apply_ocr`.
* Provide a quick snippet demonstrating `.viewer()` for interactive visual QC of OCR results.
