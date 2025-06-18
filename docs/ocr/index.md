# Getting Text from Scanned Documents

Got a PDF that's actually just a bunch of scanned images? Or maybe a PDF where the text got mangled somehow? OCR (Optical Character Recognition) is your friend. Natural PDF can extract text from image-based PDFs using several different OCR engines.

## Which OCR Engine Should You Use?

Natural PDF supports multiple OCR engines, each with different strengths:

| Engine              | Best For | Speed | Memory | Notes |
|---------------------|----------|-------|--------|-------|
| **EasyOCR**         | General documents, handwritten text | Moderate | Higher | Good all-around choice |
| **PaddleOCR**       | Asian languages, when speed matters | Fast | Efficient | Great for Chinese, Japanese, Korean |
| **Surya OCR**       | Highest accuracy needed | Moderate | Higher (GPU helps) | Best quality results |
| **Gemini**          | Complex layouts (via API) | Depends on API | N/A | Requires API key |

If you try to use an engine that isn't installed, Natural PDF will tell you exactly what to install.

## Basic OCR: Just Make It Work

The simplest approach - just apply OCR to a page and get the text:

```python
from natural_pdf import PDF

# Load a PDF that needs OCR
pdf = PDF("https://github.com/jsoma/natural-pdf/raw/refs/heads/main/pdfs/needs-ocr.pdf")
page = pdf.pages[0]

# Apply OCR using the default engine
ocr_elements = page.apply_ocr(languages=['en'])

# Extract the text (uses OCR results automatically)
text = page.extract_text()
print(text)
```

## Choosing Your OCR Engine

Pick the engine that fits your needs:

```python
# Use PaddleOCR for Chinese and English documents
ocr_elements = page.apply_ocr(engine='paddle', languages=['zh-cn', 'en'])

# Use EasyOCR with looser confidence requirements
ocr_elements = page.apply_ocr(engine='easyocr', languages=['en'], min_confidence=0.3)
```

## Fine-Tuning OCR Settings

For more control, use the engine-specific options classes:

```python
from natural_pdf.ocr import PaddleOCROptions, EasyOCROptions, SuryaOCROptions

# Configure PaddleOCR with custom settings
paddle_opts = PaddleOCROptions(
    # Check the documentation for all available options
)
ocr_elements = page.apply_ocr(engine='paddle', options=paddle_opts)

# Configure EasyOCR for better paragraph detection
easy_opts = EasyOCROptions(
    languages=['en', 'fr'],
    gpu=True,            # Use GPU if available
    paragraph=True,      # Group text into paragraphs
    text_threshold=0.6,  # How confident to be about text detection
    batch_size=8         # Process multiple regions at once
)
ocr_elements = page.apply_ocr(engine='easyocr', options=easy_opts)

# Configure Surya for high-accuracy line detection
surya_opts = SuryaOCROptions(
    languages=['en', 'de'],
    min_confidence=0.4   # Minimum confidence for results
)
ocr_elements = page.apply_ocr(engine='surya', options=surya_opts)
```

## How OCR Actually Works

When you run `page.apply_ocr()`, here's what happens:

```python
# Apply OCR to a page - this adds text elements to the page
ocr_elements = page.apply_ocr(engine='easyocr')
print(f"Found {len(ocr_elements)} text elements via OCR")

# You can also OCR just a specific region
title = page.find('text:contains("Title")')
if title:
    content_region = title.below(height=300)
    region_ocr_elements = content_region.apply_ocr(engine='paddle', languages=['en'])
```

*Note: Running OCR again on the same area will replace the previous OCR results.*

## Working with OCR Results

Once you've run OCR, the text elements work just like regular PDF text:

```python
# Find all OCR-generated text
ocr_text = page.find_all('text[source=ocr]')

# Find only high-confidence OCR text
high_conf = page.find_all('text[source=ocr][confidence>=0.8]')

# Extract just the OCR text
ocr_content = page.find_all('text[source=ocr]').extract_text()

# Search within OCR results
names = page.find_all('text[source=ocr]:contains("Smith")', case=False)
```

## Debugging OCR Results

See what the OCR engine actually found:

```python
# Run OCR first
ocr_elements = page.apply_ocr()

# Color-code by confidence level
for element in ocr_elements:
    if element.confidence >= 0.8:
        color = "green"     # High confidence
    elif element.confidence >= 0.5:
        color = "yellow"    # Medium confidence  
    else:
        color = "red"       # Low confidence
        
    element.highlight(color=color, label=f"OCR ({element.confidence:.2f})")

# Show the results
page.to_image()
```

```python
# Or just highlight the good stuff
high_conf = page.find_all('text[source=ocr][confidence>=0.8]')
high_conf.highlight(color="green", label="High Confidence OCR")
page.to_image()
```

## Advanced: Local Detection + LLM Cleanup

For really tricky documents, you can use a local model to find text regions, then send those specific regions to a language model for cleanup:

```python
from natural_pdf.ocr.utils import direct_ocr_llm
import openai

pdf = PDF("https://github.com/jsoma/natural-pdf/raw/refs/heads/main/pdfs/needs-ocr.pdf")
page = pdf.pages[0]

# Step 1: Find text regions locally (fast, no API calls)
page.apply_ocr('paddle', resolution=120, detect_only=True)

# Step 2: Set up LLM for cleanup
client = openai.OpenAI(
    base_url="https://api.anthropic.com/v1/",  
    api_key='your-api-key-here'
)

prompt = """OCR this image. Return only the exact text from the image. 
Include misspellings, punctuation, etc. Do not add quotes or comments.
The text is from a Greek spreadsheet, so expect Modern Greek or numbers."""

# Step 3: Define cleanup function
def correct(region):
    return direct_ocr_llm(
        region, 
        client, 
        prompt=prompt, 
        resolution=300, 
        model="claude-3-5-haiku-20241022"
    )

# Step 4: Apply cleanup to each detected region
page.correct_ocr(correct)

# You're done! The page now has cleaned-up text
```

## Interactive OCR Correction

Natural PDF includes a web app for reviewing and correcting OCR results:

1. **Package your PDF data:**
   ```python
   from natural_pdf.utils.packaging import create_correction_task_package
   
   # After running OCR on your PDF
   create_correction_task_package(pdf, "correction_package.zip", overwrite=True)
   ```

2. **Start the web app:**
   ```bash
   # Find where Natural PDF is installed
   NATURAL_PDF_PATH=$(python -c "import site; print(site.getsitepackages()[0])")/natural_pdf
   
   # Start the web server
   cd $NATURAL_PDF_PATH/templates/spa
   python -m http.server 8000
   ```

3. **Use the app:**
   Open `http://localhost:8000` in your browser and drag in your `correction_package.zip` file to review and edit the OCR results.

## Next Steps

Once you've got OCR working:

- [Layout Analysis](../layout-analysis/index.ipynb): Automatically detect document structure
- [Document QA](../document-qa/index.ipynb): Ask questions about your newly-readable documents