# OCR Integration

Natural PDF includes OCR (Optical Character Recognition) to extract text from scanned documents or images embedded in PDFs.

## OCR Engine Comparison

Natural PDF supports multiple OCR engines:

| Feature | EasyOCR | PaddleOCR |
| ------- | ------- | --------- |
| **Default in Library** | No | Yes |
| **Performance** | Works better for most Western documents | Excellent for Asian languages |
| **Speed** | Moderate | Fast |
| **Memory Usage** | Higher | Efficient |
| **Paragraph Detection** | Yes | No |
| **Handwritten Text** | Better support | Limited |
| **Small Text** | Moderate | Good |
| **Custom Models** | Limited | Fully supported |
| **When to Use** | Most general documents, handwritten text | Asian languages, when speed is critical |

## Basic OCR Usage

To enable OCR:

```python
from natural_pdf import PDF

# Enable OCR when opening the PDF
pdf = PDF('scanned_document.pdf', ocr=True)

# Extract text (OCR will be applied automatically)
text = pdf.pages[0].extract_text()
print(text)
```

## Auto OCR Mode

In "auto" mode, OCR is only applied when necessary:

```python
# Enable auto OCR mode
pdf = PDF('mixed_document.pdf', ocr='auto')

# OCR will only be applied to pages that need it
for page in pdf.pages:
    text = page.extract_text()  # OCR applied only if page has little/no text
    print(f"Page {page.index + 1}: {len(text)} characters extracted")
```

## OCR Configuration

You can customize OCR settings:

```python
# Set OCR parameters
pdf = PDF('document.pdf', ocr={
    'enabled': True,               # Enable OCR
    'languages': ['en'],           # Language(s) to recognize
    'min_confidence': 0.5,         # Minimum confidence threshold
    'paragraph': True,             # Try to group words into paragraphs
})

# Extract text with OCR
text = pdf.pages[0].extract_text()
```

## Multiple Languages

OCR supports multiple languages:

```python
# Recognize English and Spanish text
pdf = PDF('multilingual.pdf', ocr={
    'enabled': True,
    'languages': ['en', 'es']
})

# Multiple languages with PaddleOCR
pdf = PDF('multilingual_document.pdf', 
          ocr_engine='paddleocr',
          ocr={
              'enabled': True,
              'languages': ['zh', 'ja', 'ko', 'en']  # Chinese, Japanese, Korean, English
          })
```

## Applying OCR Directly

You can apply OCR to a page or region on demand:

```python
# Apply OCR to a page and get the OCR elements
ocr_elements = page.apply_ocr()
print(f"Found {len(ocr_elements)} text elements via OCR")

# Apply OCR to a specific region
title = page.find('text:contains("Title")')
content_region = title.below(height=300)
region_ocr_elements = content_region.apply_ocr()
```

## OCR Engines

You can choose between different OCR engines:

```python
# Use EasyOCR (default)
pdf = PDF('document.pdf', ocr_engine='easyocr')

# Use PaddleOCR (often more accurate)
pdf = PDF('document.pdf', ocr_engine='paddleocr')

# Configure PaddleOCR-specific parameters
pdf = PDF('document.pdf', 
          ocr_engine='paddleocr',
          ocr={
              'enabled': True,
              'use_angle_cls': False,  # Disable text direction detection
              'det_db_thresh': 0.3,    # Text detection threshold
              'rec_batch_num': 6       # Recognition batch size
          })
```

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

## OCR Debugging

For troubleshooting OCR problems:

```python
# Create an interactive HTML debug report
pdf.debug_ocr("ocr_debug.html")

# Specify which pages to include
pdf.debug_ocr("ocr_debug.html", pages=[0, 1, 2])
```

The debug report shows:
- The original image
- Text found with confidence scores
- Boxes around each detected word
- Options to sort and filter results

## Troubleshooting OCR

Having problems with OCR? Our [OCR Challenges and Solutions](../explanations/ocr-challenges.md) guide provides detailed information about:

- Comparing EasyOCR and PaddleOCR engines
- Fixing issues with low-quality scans
- Handling mixed languages and complex layouts
- Optimizing OCR parameters for better results

## OCR Parameter Tuning

### Parameter Recommendation Table

| Issue | Engine | Parameter | Recommended Value | Effect |
|-------|--------|-----------|-------------------|--------|
| Missing text | EasyOCR | `text_threshold` | 0.1 - 0.3 (default: 0.7) | Lower values detect more text but may increase false positives |
| Missing text | PaddleOCR | `det_db_thresh` | 0.1 - 0.3 (default: 0.3) | Lower values detect more text areas |
| Low quality scan | EasyOCR | `contrast_ths` | 0.05 - 0.1 (default: 0.1) | Lower values help with low contrast documents |
| Low quality scan | PaddleOCR | `det_limit_side_len` | 1280 - 2560 (default: 960) | Higher values improve detail detection |
| Accuracy vs. speed | EasyOCR | `decoder` | "wordbeamsearch" (accuracy)<br>"greedy" (speed) | Word beam search is more accurate but slower |
| Accuracy vs. speed | PaddleOCR | `rec_batch_num` | 1 (accuracy)<br>8+ (speed) | Larger batches process faster but use more memory |
| Small text | Both | `min_confidence` | 0.3 - 0.4 (default: 0.5) | Lower confidence threshold to capture small/blurry text |
| Text orientation | PaddleOCR | `use_angle_cls` | `True` | Enable angle classification for rotated text |
| Asian languages | PaddleOCR | `lang` | "ch", "japan", "korea" | Use PaddleOCR for Asian languages |

## Next Steps

With OCR capabilities, you can explore:

- [Layout Analysis](../layout-analysis/index.md) for automatically detecting document structure
- [Document QA](../document-qa/index.md) for asking questions about your documents
- [Understanding PDF Fonts](../explanations/pdf-fonts.md) for font-related text extraction issues
- [Visual Debugging](../visual-debugging/index.md) for visualizing OCR results