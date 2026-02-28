# OCR Then Navigate

Process scanned documents by applying OCR first, then using spatial navigation to extract data.

**When to use this pattern:**

- Scanned paper forms
- Faxed documents
- Image-based PDFs (no selectable text)
- Old documents digitized as images

## The Problem

You open a PDF and `extract_text()` returns nothing - it's a scanned image. You need OCR to make the content searchable, then you can use the same spatial navigation patterns as with text-based PDFs.

## Sample PDF

This tutorial uses `pdfs/cookbook/scanned_form.pdf` - a simulated scanned application form.

```python
import natural_pdf as npdf

pdf = npdf.PDF("pdfs/cookbook/scanned_form.pdf")
page = pdf.pages[0]

# Check if there's text
text = page.extract_text()
print(f"Text length: {len(text)}")  # May be 0 or minimal
```

## Step 1: Apply OCR

```python
# Apply OCR to make content searchable
page.apply_ocr()

# Now text extraction works
text = page.extract_text()
print(text[:200])
```

**Default engine:** EasyOCR (works well for most documents)

## Step 2: Navigate Like Normal

After OCR, use the same patterns as text-based PDFs:

```python
# Find a label and get its value
name_label = page.find('text:contains("Name:")')
if name_label:
    name = name_label.right().extract_text().strip()
    print(f"Name: {name}")

# Extract multiple fields
fields = ['Name:', 'Date of Birth:', 'Phone:', 'Email:']
data = {}
for field in fields:
    label = page.find(f'text:contains("{field}")')
    if label:
        data[field] = label.right().extract_text().strip()

print(data)
```

## Choosing an OCR Engine

Natural PDF supports multiple OCR engines:

| Engine | Notes | Installation |
|--------|-------|--------------|
| `easyocr` | Good starting point. Supports 80+ languages. | `pip install easyocr` |
| `surya` | Handles multi-language and dense layouts. | `pip install "surya-ocr<0.15"` |
| `paddle` | Best CJK (Chinese/Japanese/Korean) support. | `pip install paddlepaddle paddleocr` |
| `paddlevl` | VLM-based — understands charts and complex layouts. | `pip install paddlepaddle paddleocr` |
| `doctr` | Smaller model footprint. | `pip install python-doctr` |

```python
# Use a specific engine
page.apply_ocr(engine="surya")

# Or for the whole PDF
pdf.apply_ocr(engine="easyocr")
```

## Adjusting OCR Quality

### Resolution

Higher resolution improves accuracy but takes longer:

```python
# Default is 150 DPI
page.apply_ocr(resolution=150)

# Higher resolution for small text
page.apply_ocr(resolution=300)
```

### Confidence Filtering

OCR results include confidence scores. Filter out low-confidence text:

```python
# Apply OCR
page.apply_ocr()

# Find only high-confidence text
reliable_text = page.find_all('text[confidence>=0.8]')
print(f"Found {len(reliable_text)} reliable text elements")

# Use in selectors
label = page.find('text[confidence>=0.7]:contains("Name:")')
```

## Handling Poor Quality Scans

### Check OCR Quality First

```python
page.apply_ocr()

# Review what was detected
all_text = page.find_all('text')
for elem in all_text[:10]:
    conf = elem.confidence if hasattr(elem, 'confidence') else 'N/A'
    print(f"[{conf:.2f}] {elem.extract_text()[:50]}")
```

### Visualize OCR Results

```python
# Show the page with OCR bounding boxes
page.find_all('text').show()
```

### Multiple OCR Passes

For difficult documents, try different engines:

```python
# First pass with EasyOCR
page.apply_ocr(engine="easyocr")
easyocr_text = page.extract_text()

# If results are poor, try Surya
if len(easyocr_text) < 100:  # Suspiciously short
    page.clear_ocr()  # Remove previous results
    page.apply_ocr(engine="surya")
```

## Extracting Form Fields

Scanned forms often have checkboxes. Use `detect_checkboxes()` instead of text matching — it uses a YOLO model that works on both scanned and vector PDFs:

```python
# Detect checkboxes on the page
checkboxes = page.detect_checkboxes()

for cb in checkboxes:
    label = cb.right(width=200).extract_text().strip()
    status = "checked" if cb.is_checked else "unchecked"
    print(f"  {status}: {label}")
```

See the [OCR Integration tutorial](../tutorials/12-ocr-integration.md#detecting-checkboxes) for engine options and details.

## Complete Example

```python
import natural_pdf as npdf

def extract_scanned_form(pdf_path, ocr_engine="easyocr"):
    """Extract data from a scanned form."""
    pdf = npdf.PDF(pdf_path)
    page = pdf.pages[0]

    # Apply OCR
    page.apply_ocr(engine=ocr_engine, resolution=200)

    data = {}

    # Define fields to extract (label text -> output key)
    fields = {
        'Name:': 'name',
        'Date of Birth:': 'dob',
        'Address:': 'address',
        'Phone:': 'phone',
        'Email:': 'email',
        'Application Date:': 'application_date',
        'Reference Number:': 'reference_number',
    }

    for label_text, key in fields.items():
        # Use case-insensitive matching (OCR might get case wrong)
        label = page.find(f'text:contains("{label_text}")', case=False)
        if label:
            # Get value with confidence filtering
            value_region = label.right()
            # Extract only confident text
            confident_text = value_region.find_all('text[confidence>=0.6]')
            if confident_text:
                data[key] = ' '.join(e.extract_text() for e in confident_text).strip()
            else:
                data[key] = value_region.extract_text().strip()
        else:
            data[key] = None

    # Extract amount (often has OCR issues with $ symbol)
    amount_label = page.find('text:contains("Amount")', case=False)
    if amount_label:
        raw_amount = amount_label.right().extract_text()
        # Clean up common OCR errors
        import re
        amount = re.sub(r'[^\d.]', '', raw_amount)  # Keep only digits and decimal
        try:
            data['amount'] = float(amount) if amount else None
        except ValueError:
            data['amount'] = raw_amount  # Keep raw if parsing fails

    pdf.close()
    return data

# Usage
form_data = extract_scanned_form("pdfs/cookbook/scanned_form.pdf")
for key, value in form_data.items():
    print(f"{key}: {value}")
```

## Batch Processing Scanned PDFs

```python
import natural_pdf as npdf
from pathlib import Path
import pandas as pd

def process_scanned_batch(pdf_dir, ocr_engine="easyocr"):
    """Process a directory of scanned PDFs."""
    results = []

    for pdf_path in Path(pdf_dir).glob("*.pdf"):
        try:
            data = extract_scanned_form(str(pdf_path), ocr_engine)
            data['source_file'] = pdf_path.name
            results.append(data)
        except Exception as e:
            print(f"Error processing {pdf_path.name}: {e}")
            results.append({'source_file': pdf_path.name, 'error': str(e)})

    return pd.DataFrame(results)

# Process all scanned forms
# df = process_scanned_batch("scanned_forms/")
# df.to_csv("extracted_data.csv", index=False)
```

## Troubleshooting

### "OCR returns garbage text"

- Increase resolution: `apply_ocr(resolution=300)`
- Try a different engine: `apply_ocr(engine="surya")`
- Check the scan quality - very poor scans may need preprocessing

### "Can't find labels after OCR"

OCR might split or merge text differently than expected:

```python
# See all detected text
for elem in page.find_all('text'):
    print(f"'{elem.extract_text()}'")

# Use partial matching
label = page.find('text:contains("Nam")', case=False)  # Partial match
```

### "Confidence scores are all low"

The document may need preprocessing:

```python
# For now, lower your threshold
label = page.find('text[confidence>=0.3]:contains("Name")')

# Or ignore confidence
label = page.find('text:contains("Name")')
```

## Next Steps

- [Label-Value Extraction](label-value-extraction.md) - More on finding labels and values
- [One Page = One Row](one-page-one-row.md) - Process multiple scanned forms
- [Batch Processing](batch-processing.md) - Handle hundreds of scanned PDFs
