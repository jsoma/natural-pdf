# OCR Integration

Natural PDF includes OCR (Optical Character Recognition) to extract text from scanned documents or images embedded in PDFs.

## OCR Engine Comparison

Natural PDF supports multiple OCR engines:

| Feature              | EasyOCR                            | PaddleOCR                                | Surya OCR                             |
|----------------------|------------------------------------|------------------------------------------|---------------------------------------|
| **Installation**     | `natural-pdf[easyocr]`             | `natural-pdf[paddle]`                    | `natural-pdf[surya]`                  |
| **Primary Strength** | Good general performance, simpler  | Excellent Asian language, speed        | High accuracy, multilingual lines     |
| **Speed**            | Moderate                           | Fast                                     | Moderate (GPU recommended)            |
| **Memory Usage**     | Higher                             | Efficient                                | Higher (GPU recommended)            |
| **Paragraph Detect** | Yes (via option)                   | No                                       | No (focuses on lines)                 |
| **Handwritten**      | Better support                     | Limited                                  | Limited                               |
| **Small Text**       | Moderate                           | Good                                     | Good                                  |
| **When to Use**      | General documents, handwritten text| Asian languages, speed-critical tasks    | Highest accuracy needed, line-level   |

## Basic OCR Usage

Apply OCR directly to a page or region:

```python
from natural_pdf import PDF

# Assume 'page' is a Page object from a PDF
page = pdf.pages[0]

# Apply OCR using the default engine (or specify one)
ocr_elements = page.apply_ocr(languages=['en'])

# Extract text (will use the results from apply_ocr if run previously)
text = page.extract_text()
print(text)
```

## Configuring OCR

Specify the engine and basic options directly:

## OCR Configuration

```python
# Use PaddleOCR for Chinese and English
ocr_elements = page.apply_ocr(engine='paddle', languages=['zh-cn', 'en'])

# Use EasyOCR with a lower confidence threshold
ocr_elements = page.apply_ocr(engine='easyocr', languages=['en'], min_confidence=0.3)
```

For advanced, engine-specific settings, use the Options classes:

```python
from natural_pdf.ocr import PaddleOCROptions, EasyOCROptions, SuryaOCROptions

# --- Configure PaddleOCR ---
paddle_opts = PaddleOCROptions(
    languages=['en', 'zh-cn'],
    use_gpu=True,         # Explicitly enable GPU if available
    use_angle_cls=False,  # Disable text direction classification (if text is upright)
    det_db_thresh=0.25,   # Lower detection threshold (more boxes, potentially noisy)
    rec_batch_num=16      # Increase recognition batch size for potential speedup on GPU
    # rec_char_dict_path='/path/to/custom_dict.txt' # Optional: Path to a custom character dictionary
    # See PaddleOCROptions documentation or source code for all parameters
 )
ocr_elements = page.apply_ocr(engine='paddle', options=paddle_opts)

# --- Configure EasyOCR ---
easy_opts = EasyOCROptions(
    languages=['en', 'fr'],
    gpu=True,            # Explicitly enable GPU if available
    paragraph=True,      # Group results into paragraphs (if structure is clear)
    detail=1,            # Ensure bounding boxes are returned (required)
    text_threshold=0.6,  # Confidence threshold for text detection (adjust based on tuning table)
    link_threshold=0.4,  # Standard EasyOCR param, uncomment if confirmed in wrapper
    low_text=0.4,        # Standard EasyOCR param, uncomment if confirmed in wrapper
    batch_size=8         # Processing batch size (adjust based on memory)
    # See EasyOCROptions documentation or source code for all parameters
 )
ocr_elements = page.apply_ocr(engine='easyocr', options=easy_opts)

# --- Configure Surya OCR ---
# Surya focuses on line detection and recognition
surya_opts = SuryaOCROptions(
    languages=['en', 'de'], # Specify languages for recognition
    # device='cuda',       # Use GPU ('cuda') or CPU ('cpu') <-- Set via env var TORCH_DEVICE
    min_confidence=0.4   # Example: Adjust minimum confidence for results
    # Core Surya options like device, batch size, and thresholds are typically
    # set via environment variables (see note below).
)
ocr_elements = page.apply_ocr(engine='surya', options=surya_opts)
```

## Applying OCR Directly

The `page.apply_ocr(...)` and `region.apply_ocr(...)` methods are the primary way to run OCR:

```python
# Apply OCR to a page and get the OCR elements
ocr_elements = page.apply_ocr(engine='easyocr')
print(f"Found {len(ocr_elements)} text elements via OCR")

# Apply OCR to a specific region
title = page.find('text:contains("Title")')
content_region = title.below(height=300)
region_ocr_elements = content_region.apply_ocr(engine='paddle', languages=['en'])
```

## OCR Engines

Choose the engine best suited for your document and language requirements using the `engine` parameter in `apply_ocr`.

## Finding and Working with OCR Text

After applying OCR, work with the text just like regular text:

```python
# Find all OCR text elements
ocr_text = page.find_all('text[source=ocr]')

# Find high-confidence OCR text
high_conf = page.find_all('text[source=ocr][confidence>=0.8]')

# Extract text only from OCR elements
ocr_text_content = page.find_all('text[source=ocr]').extract_text()

# Filter OCR text by content
names = page.find_all('text[source=ocr]:contains("Smith")', case=False)
```

## Visualizing OCR Results

See OCR results to help debug issues:

```python
# Apply OCR 
ocr_elements = page.apply_ocr()

# Highlight all OCR elements
for element in ocr_elements:
    # Color based on confidence
    if element.confidence >= 0.8:
        color = "green"  # High confidence
    elif element.confidence >= 0.5:
        color = "yellow"  # Medium confidence
    else:
        color = "red"  # Low confidence
        
    element.highlight(color=color, label=f"OCR ({element.confidence:.2f})")

# Get the visualization as an image
image = page.to_image(labels=True)
# Just return the image in a Jupyter cell
image

# Highlight only high-confidence elements
high_conf = page.find_all('text[source=ocr][confidence>=0.8]')
high_conf.highlight(color="green", label="High Confidence OCR")
```

## Detect + LLM OCR

Sometimes you have a difficult piece of content where you need to use a local model to identify the content, then send it off in pieces to be identified by the LLM. You can do this with Natural PDF!

```python
from natural_pdf import PDF
from natural_pdf.ocr.utils import direct_ocr_llm
import openai

pdf = PDF("needs-ocr.pdf")
page = pdf.pages[0]

# Detect
page.apply_ocr('paddle', resolution=120, detect_only=True)

# Build the framework
client = openai.OpenAI(base_url="https://api.anthropic.com/v1/",  api_key='sk-XXXXX')
prompt = """OCR this image. Return only the exact text from the image. Include misspellings,
punctuation, etc. Do not surround it with quotation marks. Do not include translations or comments.
The text is from a Greek spreadsheet, so most likely content is Modern Greek or numeric."""

# This returns the cleaned-up text
def correct(region):
    return direct_ocr_llm(region, client, prompt=prompt, resolution=300, model="claude-3-5-haiku-20241022")

# Run 'correct' on each text element
page.correct_ocr(correct)

# You're done!
```

## Debugging OCR

```python
from natural_pdf.utils.packaging import create_correction_task_package

create_correction_task_package(pdf, "original.zip", overwrite=True)
```

This will at *some point* be official-ized, but for now you can look at `templates/spa` and see the correction package.

## Next Steps

With OCR capabilities, you can explore:

- [Layout Analysis](../layout-analysis/index.ipynb) for automatically detecting document structure
- [Document QA](../document-qa/index.ipynb) for asking questions about your documents
- [Visual Debugging](../visual-debugging/index.ipynb) for visualizing OCR results