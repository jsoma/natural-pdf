# OCR Challenges and Solutions

OCR (Optical Character Recognition) seems simple in concept—turn images of text into actual text—but it's full of interesting challenges. This guide covers common OCR problems and how to solve them.

## When Do You Need OCR?

You'll need OCR in several common scenarios:

1. **Scanned documents**: The obvious case—documents that were printed and scanned.
2. **Image-only PDFs**: PDFs created from images or scans without text layers.
3. **Protected PDFs**: Some PDFs have security settings that prevent text extraction.
4. **Problematic fonts**: When fonts are embedded incorrectly or have unusual encoding.

## How OCR Works (The Short Version)

OCR generally follows these steps:

1. **Preprocessing**: Adjust the image (binarization, deskewing, noise removal)
2. **Text detection**: Find where the text is located in the image
3. **Character recognition**: Identify individual characters in those regions
4. **Post-processing**: Correct errors using dictionaries or language models

Different OCR engines handle these steps in different ways, which is why they perform differently on various documents.

## OCR Engines Compared

Natural PDF supports different OCR engines, each with strengths and weaknesses:

### EasyOCR

**Strengths**:
- Simple to use and configure
- Good support for European languages
- Reasonable performance on clean documents

**Weaknesses**:
- Slower than PaddleOCR
- Struggles with complex layouts
- Less accurate on small or low-contrast text

### PaddleOCR

**Strengths**:
- Fast processing
- Excellent performance on many languages
- Better with complex layouts and small text
- More accurate in many real-world scenarios

**Weaknesses**:
- More complex parameter tuning
- Larger model files

## Common OCR Problems and Solutions

### 1. Low Image Quality

**Problem**: Blurry, low-resolution, or noisy images lead to poor OCR results.

**Solution**:
```python
# Increase resolution when generating the image for OCR
page.apply_ocr(resolution=300)  # Default is 200 DPI

# For noisy images, adjust preprocessing parameters
page.apply_ocr(
    text_threshold=0.6,  # Text detection confidence (default: 0.7)
    low_text=0.3,        # Text low-bound score (default: 0.4)
    link_threshold=0.3   # Link confidence threshold (default: 0.4)
)
```

### 2. Rotated or Skewed Text

**Problem**: Text that isn't perfectly horizontal can cause OCR to fail.

**Solution**:
```python
# Enable text rotation detection in PaddleOCR
pdf = PDF('skewed_document.pdf', 
          ocr_engine='paddleocr',
          ocr={
              'enabled': True,
              'use_angle_cls': True  # Detect text direction
          })
```

### 3. Mixed Languages

**Problem**: Documents with multiple languages confuse single-language OCR models.

**Solution**:
```python
# Specify multiple languages
pdf = PDF('multilingual.pdf', ocr={
    'enabled': True,
    'languages': ['en', 'fr', 'de']  # English, French, German
})

# For Asian languages mixed with others, PaddleOCR often works better
pdf = PDF('mixed_languages.pdf', 
          ocr_engine='paddleocr',
          ocr={
              'enabled': True,
              'languages': ['en', 'zh', 'ja']  # English, Chinese, Japanese
          })
```

### 4. Small Text

**Problem**: Tiny text often gets missed or misread by OCR.

**Solution**:
```python
# Increase image resolution for OCR
page.apply_ocr(resolution=400)  # Higher resolution for small text

# Adjust magnification ratio
page.apply_ocr(mag_ratio=2.0)  # Default is 1.5
```

### 5. Complex Layouts

**Problem**: Multi-column text, tables, and other complex layouts can confuse OCR.

**Solution**:
```python
# Use layout analysis first to detect regions
page.analyze_layout()

# Then apply OCR to specific regions
text_regions = page.find_all('region[type=plain-text]')
for region in text_regions:
    region_text = region.extract_text(ocr=True)
    print(region_text)
```

### 6. Low OCR Confidence

**Problem**: OCR returns text but with low confidence scores.

**Solution**:
```python
# Visualize confidence scores to identify problem areas
ocr_elements = page.apply_ocr()
for element in ocr_elements:
    if element.confidence < 0.5:
        element.highlight(color="red", label=f"Low conf: {element.confidence:.2f}")
    else:
        element.highlight(color="green", label=f"High conf: {element.confidence:.2f}")
image = page.to_image(labels=True)
image

# Filter by confidence
high_confidence = page.find_all('text[source=ocr][confidence>=0.7]')
high_confidence_text = high_confidence.extract_text()
```

## When OCR Isn't Working

If OCR is giving poor results even after tuning, try these approaches:

1. **Try a different engine**: If EasyOCR isn't working well, try PaddleOCR and vice versa.

   ```python
   # Switch engines to compare results
   pdf_easy = PDF('document.pdf', ocr_engine='easyocr')
   pdf_paddle = PDF('document.pdf', ocr_engine='paddleocr')
   
   # Extract with both and compare
   easy_text = pdf_easy.pages[0].extract_text(ocr=True)
   paddle_text = pdf_paddle.pages[0].extract_text(ocr=True)
   ```

2. **Pre-process the PDF**: Sometimes converting the PDF to images externally and cleaning them up with tools like ImageMagick before OCRing can help.

3. **Focus on regions**: Apply OCR to smaller, targeted regions rather than the whole page.

   ```python
   # Get just the important part of the page
   important_area = page.create_region(100, 200, 500, 600)
   important_text = important_area.extract_text(ocr=True)
   ```

4. **Combine results**: Use both native text extraction and OCR, then choose the better one.

   ```python
   # Get both and compare
   native_text = page.extract_text(ocr=False)
   ocr_text = page.extract_text(ocr=True)
   
   # Choose the longer one (often the better extraction)
   final_text = native_text if len(native_text) > len(ocr_text) else ocr_text
   ```

## The Document QA Alternative

Sometimes extracting perfect text isn't necessary. If you're looking to answer questions about a document, Document QA might work better:

```python
# Ask questions directly rather than extracting text first
result = pdf.ask("What was the total revenue in 2023?")
if result.get("found", False):
    print(f"Answer: {result['answer']}")
```

Document QA uses the image along with text positions to understand content, so it can often handle cases where OCR alone struggles.

## Why Document QA Can Be Better Than Just Using an LLM

If you've thought "why not just feed the text to ChatGPT?", there are good reasons to use Document QA instead:

1. **No hallucinations**: Document QA only returns information actually present in the document
2. **Visual context**: It understands layout, so it can interpret tables and know when content is in headers vs body text
3. **Shows you the source**: You can see exactly where the answer came from
4. **Better with complex layouts**: It understands when text is arranged in columns, tables, or other structures

## Further Reading

- [OCR Integration in Natural PDF](../ocr/index.md)
- [Document QA](../document-qa/index.md)
- [Layout Analysis](../layout-analysis/index.md)