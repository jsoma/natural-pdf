# Understanding Fonts in PDFs

Fonts in PDFs can be tricky. If you've extracted text and gotten strange characters or missing content, font issues are often the culprit. This guide explains how PDF fonts work and how to handle common problems you might encounter.

## How Fonts Work in PDFs

PDFs can include fonts in several ways:

1. **Standard Fonts**: A set of 14 standard fonts that all PDF viewers support (like Times, Helvetica, Courier)
2. **Embedded Fonts**: Fonts included directly within the PDF file
3. **Referenced Fonts**: Fonts referenced by name but expected to be available on the user's system
4. **Subset Fonts**: Embedded fonts containing only the characters actually used in the document

### Font Naming in PDFs

One of the most confusing aspects of PDF fonts is their naming. You'll often see font names like `ABCDEF+Arial-Bold`. This strange format has two components:

- **Prefix (ABCDEF+)**: A unique identifier for a font subset
- **Base Name (Arial-Bold)**: The actual font family and style

The prefix is generated during PDF creation and helps distinguish between different subsets of the same font. For example, a document might have:

- `ABCDEF+Arial-Bold` containing only the characters "Hello"
- `XYZPQR+Arial-Bold` containing only the characters "World"

Both are subsets of Arial Bold, but they contain different character sets to minimize file size.

## Diagnosing Font Issues with Natural PDF

### Listing All Fonts in a Document

Natural PDF can help you understand what fonts are used in your document:

```python
from natural_pdf import PDF
from collections import defaultdict

pdf = PDF("document.pdf")

# Create a dictionary to store fonts
fonts = defaultdict(int)

# Collect all fonts from all pages
for page in pdf.pages:
    text_elements = page.find_all('text')
    for elem in text_elements:
        fonts[elem.fontname] = fonts.get(elem.fontname, 0) + 1

# Print the fonts and their usage count
print("Fonts used in document:")
for font, count in sorted(fonts.items(), key=lambda x: x[1], reverse=True):
    print(f"- {font}: {count} occurrences")

# Analyze font variants (subsets)
variants = defaultdict(set)
for font in fonts.keys():
    if '+' in font:
        prefix, base = font.split('+', 1)
        variants[base].add(prefix)

print("\nFont variants:")
for base, prefixes in variants.items():
    print(f"- {base}: {len(prefixes)} variants ({', '.join(prefixes)})")
```

### Finding Elements with Specific Font Variants

You can use Natural PDF's selectors to find text with specific font variants:

```python
# Find all elements using a specific font variant
arial_bold_variant = page.find_all('text[fontname*="ABCDEF+Arial-Bold"]')

# Or use the font-variant attribute (which extracts the prefix)
variant_elements = page.find_all('text[font-variant="ABCDEF"]')

# Find all elements using the base font, regardless of variant
all_arial_bold = page.find_all('text[fontname*="Arial-Bold"]')
```

## Common Font Challenges in PDFs

### 1. Font Subsets and Text Extraction

When PDFs use font subsets, some characters might not be properly mapped, leading to incorrect text extraction. You might encounter:

- Gibberish text (`"Ð¡ŽŠ"` instead of "Hello")
- Missing characters (appearing as □ or ?)
- Completely incorrect character mappings

Natural PDF helps handle this:

```python
# For PDFs with font encoding issues, OCR can be more reliable
pdf = PDF("problematic_document.pdf", ocr="auto")

# Extract text - will use OCR if native text extraction fails
text = pdf.pages[0].extract_text()
```

### 2. Identifying Text by Visual Appearance vs Font Name

Sometimes you want to find text that looks visually similar, regardless of how the font is technically named:

```python
# Group text by visual properties (size, weight, etc.)
text_styles = page.analyze_text_styles()

# Check the properties of each style
for style_name, elements in text_styles.items():
    if elements:
        example = elements[0]
        print(f"{style_name}:")
        print(f"  Font: {example.fontname}")
        print(f"  Size: {example.size}")
        print(f"  Bold: {example.bold}")
        print(f"  Italic: {example.italic}")
        print(f"  Example: {example.text[:20]}...")
```

### 3. Font Color Handling

PDFs represent colors in different ways. Natural PDF normalizes these for easy use:

```python
# Find text with specific colors
red_text = page.find_all('text[color~=red]')
blue_text = page.find_all('text[color~=blue]')

# Check the color of a specific element
element = page.find('text:contains("Important")')
if element:
    print(f"Color: {element.color}")  # Returns as RGB tuple
    
    # Visualize the color
    element.highlight(label=f"Color: {element.color}")
    image = page.to_image()
    image
```

## Font Embedding and Accessibility

Understanding font embedding is important for PDF accessibility and preservation:

- **Fully embedded fonts**: The PDF contains all font data needed to render the text
- **Subset embedded fonts**: Only the characters used in the document are embedded
- **Referenced fonts**: The PDF relies on the viewing system having the font installed

Natural PDF can help you determine if fonts might cause problems:

```python
# Advanced: Check for non-embedded fonts that might cause issues
from collections import Counter

non_embedded_fonts = Counter()

for page in pdf.pages:
    text_elements = page.find_all('text')
    for elem in text_elements:
        # Use pdfplumber's native font info
        font_dict = elem._obj.get('fontname', '')
        
        # If a font doesn't contain a '+', it might not be embedded
        if '+' not in font_dict and font_dict not in ['Times-Roman', 'Helvetica', 'Courier']:
            non_embedded_fonts[font_dict] += 1

if non_embedded_fonts:
    print("Warning: Document contains possibly non-embedded fonts:")
    for font, count in non_embedded_fonts.most_common():
        print(f"- {font}: {count} occurrences")
```

## When to Use OCR Instead of Native Text Extraction

Sometimes, even with embedded fonts, text extraction may fail due to:

1. Custom encoding tables
2. Security features in the PDF
3. Non-standard fonts with incorrect character mappings
4. Text rendered as vector graphics or images

In these cases, OCR might yield better results:

```python
# Compare native extraction vs OCR
native_text = page.extract_text(ocr=False)
ocr_text = page.extract_text(ocr=True)

print(f"Native extraction: {len(native_text)} characters")
print(f"OCR extraction: {len(ocr_text)} characters")

# If lengths differ dramatically, you may have font encoding issues
if abs(len(native_text) - len(ocr_text)) > len(native_text) * 0.5:
    print("Significant difference detected - possible font encoding issues")
    
    # View sample of both for comparison
    print("\nNative sample:", native_text[:100])
    print("\nOCR sample:", ocr_text[:100])
```

## Best Practices for Font Handling

1. **Check font coverage first**: Use the font analysis example above to understand what fonts your document uses
2. **Use font-variant in selectors**: When you need to target specific variants of a font
3. **Try OCR when text extraction fails**: If you see gibberish or missing text, OCR may give better results
4. **Group by visual appearance**: Use `analyze_text_styles()` to find text that looks similar, regardless of technical font names

Understanding font handling in PDFs will help you extract text more effectively and handle edge cases that often arise in real-world documents.

## Further Reading

- [Understanding Font Naming in PDFs](https://helpx.adobe.com/acrobat/using/pdf-fonts.html)
- [PDF Specification - Font Resources](https://opensource.adobe.com/dc-acrobat-sdk-docs/pdfstandards/PDF32000_2008.pdf)
- [OCR Integration in Natural PDF](../ocr/index.md)