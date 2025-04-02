# Text Analysis

Analyzing the properties of text elements, such as their font, size, style, and color, can be crucial for understanding document structure and extracting specific information. Natural PDF provides tools to access and analyze these properties.

## Introduction

Beyond just the sequence of characters, the *style* of text carries significant meaning. Headings are often larger and bolder, important terms might be italicized, and different sections might use distinct fonts. This page covers how to access and utilize this stylistic information.

## Accessing Font Information

Every `TextElement` (representing characters or words) holds information about its font properties.

```python
from natural_pdf import PDF

pdf = PDF('document.pdf')
page = pdf.pages[0]

# Find the first word element
word = page.find('word') 

if word:
    print(f"Text: {word.text}")
    print(f"Font Name: {word.fontname}") # Font reference (e.g., F1) or name
    print(f"Real Font Name: {getattr(word, 'real_fontname', 'N/A')}") # Actual font name if resolved
    print(f"Size: {word.size:.2f} pt")
    print(f"Color: {getattr(word, 'color', 'N/A')}") # Non-stroking color
    print(f"Is Bold: {word.bold}")
    print(f"Is Italic: {word.italic}") 
```

- `fontname`: Often an internal reference (like 'F1', 'F2') or a basic name.
- `real_fontname`: The library attempts to resolve `fontname` to the actual font name embedded in the PDF (e.g., 'Arial-BoldMT'). Access using `getattr` as it might not always be present.
- `size`: Font size in points.
- `color`: The non-stroking color, typically a tuple representing RGB or Grayscale values (e.g., `(0.0, 0.0, 0.0)` for black). Access using `getattr`.
- `bold`, `italic`: Boolean flags indicating if the font style is bold or italic (heuristically determined based on font name conventions).

## Working with Text Styles

You can directly select text based on its style using pseudo-classes in selectors:

```python
# Find all bold text elements
bold_text = page.find_all('text:bold')

# Find all italic text elements
italic_text = page.find_all('text:italic')

# Find text that is both bold and larger than 12pt
bold_headings = page.find_all('text:bold[size>=12]')

print(f"Found {len(bold_text)} bold elements.")
print(f"Found {len(italic_text)} italic elements.")
```

## Analyzing Font Usage Across a Page

You can gather statistics about font usage on a page:

```python
from collections import Counter

all_words = page.find_all('word')

# Count occurrences of each font name and size combination
font_usage = Counter()
for word in all_words:
    font_key = (getattr(word, 'real_fontname', word.fontname), word.size)
    font_usage[font_key] += 1

print("Font Usage Summary:")
for (font, size), count in font_usage.most_common():
    print(f"- Font: {font}, Size: {size:.2f}pt, Count: {count}")

```

## Visualizing Text Properties

Use highlighting to visually inspect text properties. Grouping by attributes like `fontname` or `size` can be very insightful.

```python
# Highlight words, grouping by font name
# Each font will get a different color in the output image/preview
page.find_all('word').highlight(group_by='fontname', label_format="{fontname}")

# Show a temporary preview grouping by font size category
def size_category(element):
    if element.size >= 14: return "Large"
    if element.size >= 10: return "Medium"
    return "Small"

# Apply the function to create a temporary 'size_cat' attribute for grouping
words = page.find_all('word')
for w in words: w.size_cat = size_category(w) 

# Show preview grouped by the custom category
words.show(group_by='size_cat') 

# Don't forget to clean up the temporary attribute if needed
# for w in words: delattr(w, 'size_cat') 
```

This allows you to quickly see patterns in font usage across the page layout. 