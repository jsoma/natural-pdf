# Selectors 101

Selectors are the heart of Natural PDF. They let you find elements in your PDF using a simple, CSS-like syntax.

## Quick Reference

```
'text'                          # All text elements
'text:bold'                     # Bold text (pseudo-class)
'text:contains("Invoice")'      # Text containing "Invoice"
'text[size>12]'                 # Text with font size > 12 (attribute)
'text:bold[size>=14]'           # Bold text AND size >= 14 (combined)
```

## The Basics

### Element Types

Every selector starts with an element type:

| Type | What It Finds | Example |
|------|---------------|---------|
| `text` | Text content | `page.find('text')` |
| `line` | Lines and rules | `page.find('line')` |
| `rect` | Rectangles and boxes | `page.find('rect')` |
| `image` | Embedded images | `page.find('image')` |
| `region` | Layout-detected areas | `page.find('region')` |

```python
# Find the first text element
first_text = page.find('text')

# Find all lines
all_lines = page.find_all('line')
```

### Pseudo-Classes (`:name`)

Pseudo-classes filter by state or content. They use a **colon** (`:`) prefix.

| Pseudo-Class | Description | Example |
|--------------|-------------|---------|
| `:bold` | Bold text | `'text:bold'` |
| `:italic` | Italic text | `'text:italic'` |
| `:contains("X")` | Contains the text "X" | `'text:contains("Invoice")'` |
| `:startswith("X")` | Starts with "X" | `'text:startswith("Total")'` |
| `:endswith("X")` | Ends with "X" | `'text:endswith(":")'` |
| `:horizontal` | Horizontal lines | `'line:horizontal'` |
| `:vertical` | Vertical lines | `'line:vertical'` |

```python
# Find bold text
bold = page.find('text:bold')

# Find text containing "Total"
total = page.find('text:contains("Total")')

# Case-insensitive search
total = page.find('text:contains("total")', case=False)
```

!!! warning "Common Mistake"
    Use **colon** (`:`) not **dot** (`.`) for pseudo-classes:

    - ✅ `'text:bold'` (correct)
    - ❌ `'text.bold'` (wrong - this is CSS class syntax)

### Attribute Filters (`[attr=value]`)

Attribute filters match specific properties. They use **brackets** (`[]`).

| Operator | Meaning | Example |
|----------|---------|---------|
| `=` | Equals | `'text[size=12]'` |
| `!=` | Not equals | `'text[size!=12]'` |
| `>` | Greater than | `'text[size>12]'` |
| `>=` | Greater or equal | `'text[size>=12]'` |
| `<` | Less than | `'text[size<12]'` |
| `<=` | Less or equal | `'text[size<=12]'` |
| `*=` | Contains | `'text[fontname*=Arial]'` |
| `^=` | Starts with | `'text[fontname^=Times]'` |
| `$=` | Ends with | `'text[fontname$=Bold]'` |

```python
# Find large text (size > 14)
large = page.find_all('text[size>14]')

# Find text in Arial font
arial = page.find_all('text[fontname*=Arial]')

# Find high-confidence OCR results
confident = page.find_all('text[confidence>=0.9]')
```

#### Common Attributes

| Attribute | Element Types | Description |
|-----------|---------------|-------------|
| `size` | text | Font size in points |
| `fontname` | text | Font family name |
| `confidence` | text (OCR) | OCR confidence (0-1) |
| `source` | text | Origin: `pdf` or `ocr` |
| `type` | region | Region type from layout analysis |
| `width` | line, rect | Element width |
| `height` | line, rect, image | Element height |
| `fill` | rect | Has fill color |
| `stroke` | rect, line | Has stroke/border |

### Combining Selectors

You can combine pseudo-classes and attributes for precise matching:

```python
# Bold text larger than 14pt
headers = page.find_all('text:bold[size>14]')

# Text containing "Total" in Arial font
totals = page.find_all('text:contains("Total")[fontname*=Arial]')

# Bold, large text containing "Summary"
summary = page.find('text:bold[size>=16]:contains("Summary")')
```

!!! tip "Selector Order"
    The order should be: `type` → `:pseudo-classes` → `[attributes]` → `:contains()`

    ```python
    'text:bold[size>14]:contains("Summary")'
    #    ^      ^            ^
    #    |      |            +-- :contains last
    #    |      +-- attributes after pseudo
    #    +-- pseudo after type
    ```

## Finding Elements

### `find()` - First Match

Returns the first matching element, or `None` if not found.

```python
# Find the first bold text
title = page.find('text:bold')

# Always check for None!
if title:
    print(title.extract_text())
else:
    print("Not found")
```

### `find_all()` - All Matches

Returns an `ElementCollection` with all matching elements.

```python
# Find all bold text
all_bold = page.find_all('text:bold')

print(f"Found {len(all_bold)} bold elements")

# ElementCollection has useful methods
first = all_bold.first  # First element or None
texts = all_bold.extract_text()  # Extract text from all
```

## Advanced Patterns

### Regex Matching

Use `regex=True` for pattern matching:

```python
# Find invoice numbers like "INV-12345"
invoice = page.find('text:contains("INV-\\d+")', regex=True)

# Find dates in MM/DD/YYYY format
dates = page.find_all('text:contains("\\d{2}/\\d{2}/\\d{4}")', regex=True)

# Match multiple words
page.find('text:contains("Total|Sum|Amount")', regex=True)
```

### Layout Regions

After running layout analysis, you can find detected regions:

```python
# First, detect layout
page.analyze_layout(engine='yolo')

# Find all detected tables
tables = page.find_all('region[type=table]')

# Find specific region types
titles = page.find_all('region[type=title]')
figures = page.find_all('region[type=figure]')
```

### OCR Elements

After applying OCR, filter by source and confidence:

```python
# Apply OCR
page.apply_ocr()

# Find OCR text (not native PDF text)
ocr_text = page.find_all('text[source=ocr]')

# Find high-confidence OCR only
confident = page.find_all('text[source=ocr][confidence>=0.8]')

# Find native PDF text only
native = page.find_all('text[source=pdf]')
```

## Common Patterns Cheat Sheet

| Goal | Selector |
|------|----------|
| All text | `'text'` |
| Bold text | `'text:bold'` |
| Large text (headings) | `'text[size>=14]'` |
| Text containing "X" | `'text:contains("X")'` |
| Case-insensitive search | `page.find('text:contains("x")', case=False)` |
| Horizontal lines | `'line:horizontal'` |
| Thick lines | `'line[width>=2]'` |
| Filled rectangles | `'rect[fill]'` |
| Detected tables | `'region[type=table]'` |
| High-confidence OCR | `'text[source=ocr][confidence>=0.8]'` |

## Troubleshooting

### "My selector finds nothing"

1. **Start broad, then narrow down:**
   ```python
   # See what's on the page
   all_text = page.find_all('text')
   print(f"Total: {len(all_text)} text elements")
   all_text.show()  # Visualize them
   ```

2. **Check your spelling and case:**
   ```python
   # This finds "Invoice" but not "INVOICE" or "invoice"
   page.find('text:contains("Invoice")')

   # Use case=False for case-insensitive
   page.find('text:contains("invoice")', case=False)
   ```

3. **Check the syntax:**
   ```python
   # Wrong
   page.find('text.bold')              # Use colon, not dot
   page.find('text[contains="X"]')     # :contains is a pseudo-class
   page.find('text:contains(Invoice)') # Need quotes around text

   # Correct
   page.find('text:bold')
   page.find('text:contains("Invoice")')
   ```

### "I get AttributeError: 'NoneType'"

`find()` returns `None` when nothing matches. Always check:

```python
# Wrong - crashes if not found
text = page.find('text:contains("Missing")').extract_text()

# Correct - handle None
element = page.find('text:contains("Missing")')
if element:
    text = element.extract_text()
else:
    text = ""
```

## Next Steps

- **[Quickstart](quickstart.md)** - See selectors in action
- **[Finding Elements Tutorial](../tutorials/02-finding-elements.ipynb)** - Deep dive into finding elements
- **[Spatial Navigation](../tutorials/08-spatial-navigation.ipynb)** - Use `.below()`, `.above()` after finding
