# Checkbox Detection in Natural PDF

Natural PDF now includes built-in checkbox detection using computer vision models. This feature can automatically detect checkboxes in PDF documents and determine whether they are checked or unchecked.

## Quick Start

```python
import natural_pdf as npdf

# Load PDF and detect checkboxes
pdf = npdf.PDF("form.pdf")
checkboxes = pdf[0].detect_checkboxes()

# Check results
for cb in checkboxes:
    print(f"Checkbox at {cb.bbox}: {'✓' if cb.is_checked else '✗'}")
```

## Features

### 1. Basic Detection
```python
# Detect all checkboxes on a page
checkboxes = page.detect_checkboxes()

# Access checkbox properties
checkbox = checkboxes[0]
print(checkbox.is_checked)      # True/False
print(checkbox.checkbox_state)  # "checked"/"unchecked"
print(checkbox.confidence)      # Detection confidence (0-1)
```

### 2. Using Selectors
```python
# Find checked/unchecked boxes
checked = page.find_all('checkbox:checked')
unchecked = page.find_all('checkbox:unchecked')

# All checkboxes
all_checkboxes = page.find_all('checkbox')

# By attributes (note: use is_checked, not checked)
checked = page.find_all('checkbox[is_checked=true]')
```

### 3. Limited Detection
When you know the expected number of checkboxes:
```python
# Get top 10 checkboxes by confidence
checkboxes = page.detect_checkboxes(limit=10)
```

### 4. Multi-level Detection
```python
# Entire PDF
all_checkboxes = pdf.detect_checkboxes()

# Page collection
pages = pdf[0:5]
checkboxes = pages.detect_checkboxes()

# Within a region
region = page.find('text:contains("Options")').below()
checkboxes = region.detect_checkboxes()
```

### 5. Visualization
```python
# Show detected checkboxes
checkboxes.show()

# Checkboxes display their state in repr
print(checkboxes[0])
# <Region type='checkbox' [checked] bbox=(100, 200, 120, 220)>
```

## Advanced Configuration

### Custom Detection Options
```python
from natural_pdf.analyzers.checkbox import CheckboxOptions

# Higher confidence threshold (default is 0.05)
options = CheckboxOptions(confidence=0.5)
checkboxes = page.detect_checkboxes(options=options)

# Different resolution (default is 150 DPI)
checkboxes = page.detect_checkboxes(resolution=300)

# GPU acceleration
checkboxes = page.detect_checkboxes(device='cuda')
```

### Custom Models
```python
# Use a different checkbox detection model
options = CheckboxOptions(
    model_repo="your-org/your-checkbox-model",
    label_mapping={
        "empty_box": "unchecked",
        "ticked_box": "checked",
    }
)
checkboxes = page.detect_checkboxes(options=options)
```

### Disable Text Filtering
```python
# If your checkboxes contain text for some reason
checkboxes = page.detect_checkboxes(reject_with_text=False)

# Or with options
options = CheckboxOptions(reject_with_text=False)
checkboxes = page.detect_checkboxes(options=options)
```

## Implementation Details

- **Default Model**: Uses `wendys-llc/rtdetr-v2-r50-chkbx` RT-DETR model
- **Low Confidence**: Default confidence is 0.02 (very low to catch all checkboxes)
- **Resolution**: Renders at 150 DPI by default for efficiency
- **No Overlaps**: Aggressive NMS rejects ANY overlapping detections
- **Text Filtering**: Automatically rejects detections containing text (real checkboxes should be empty)
- **Architecture**: Follows the same pattern as layout detection for consistency

## Common Use Cases

### Form Processing
```python
# Extract form checkbox states
form_data = {}
for cb in page.detect_checkboxes():
    # Find nearby text label
    label = cb.left('text').extract_text() or cb.above('text').extract_text()
    form_data[label] = cb.is_checked
```

### Validation
```python
# Ensure all required checkboxes are checked
required = ["Terms", "Privacy", "Age"]
checkboxes = page.detect_checkboxes()

for req in required:
    cb = page.find(f'text:contains("{req}")').right('checkbox:first')
    if not cb or not cb.is_checked:
        print(f"Warning: {req} not checked!")
```

### Batch Processing
```python
# Process multiple forms
for pdf_path in pdf_files:
    pdf = npdf.PDF(pdf_path)
    results = []

    for page in pdf.pages:
        checkboxes = page.detect_checkboxes(limit=20)
        checked_count = len([cb for cb in checkboxes if cb.is_checked])
        results.append({
            'page': page.number,
            'total': len(checkboxes),
            'checked': checked_count
        })
```

## Troubleshooting

1. **No checkboxes detected**: Try lowering confidence threshold
2. **Too many false positives**: Increase confidence threshold
3. **Missing transformers**: Install with `pip install transformers torch`
4. **Selector syntax**: Use `:checked`/`:unchecked` or `[is_checked=true]`
