# Extract Clean Text Without Headers and Footers

You've got a PDF where you need the main content, but every page has headers, footers, watermarks, or other junk that's messing up your text extraction. Here's how to get just the content you want.

## The Problem

PDFs often have repeated content on every page that you don't want:

- Company headers with logos and contact info
- Page numbers and footers
- "CONFIDENTIAL" watermarks
- Navigation elements from web-to-PDF conversions

When you extract text normally, all this noise gets mixed in with your actual content.

## Quick Solution: Exclude by Pattern

If the unwanted content is consistent across pages, you can exclude it once:

```python
from natural_pdf import PDF

pdf = PDF("https://github.com/jsoma/natural-pdf/raw/refs/heads/main/pdfs/01-practice.pdf")
page = pdf.pages[0]

# Find and exclude the header (top 10% of page)
header_region = page.create_region(0, 0, page.width, page.height * 0.1)
page.add_exclusion(header_region)

# Find and exclude footer (bottom 10% of page)
footer_region = page.create_region(0, page.height * 0.9, page.width, page.height)
page.add_exclusion(footer_region)

# Now extract clean text
clean_text = page.extract_text()
```

## Exclude Specific Elements

For more precision, exclude specific text or elements:

```python
# Exclude anything containing "CONFIDENTIAL"
confidential = page.find('text:contains("CONFIDENTIAL")')
if confidential:
    page.add_exclusion(confidential.above())  # Everything above it

# Exclude page numbers (usually small text with just numbers)
page_nums = page.find_all('text:contains("^\\d+$")', regex=True)
for num in page_nums:
    page.add_exclusion(num)

# Exclude elements by position (like top-right logos)
top_right = page.create_region(page.width * 0.7, 0, page.width, page.height * 0.15)
page.add_exclusion(top_right)
```

## Apply Exclusions to All Pages

Set up exclusions that work across your entire document:

```python
# Define exclusions that adapt to each page
def exclude_header(page):
    # Top 50 points of every page
    return page.create_region(0, 0, page.width, 50)

def exclude_footer(page):
    # Bottom 30 points of every page
    return page.create_region(0, page.height - 30, page.width, page.height)

def exclude_watermark(page):
    # Find "DRAFT" watermark if it exists
    draft = page.find('text:contains("DRAFT")')
    return draft.create_region() if draft else None

# Apply to entire PDF
pdf.add_exclusion(exclude_header, label="Headers")
pdf.add_exclusion(exclude_footer, label="Footers")
pdf.add_exclusion(exclude_watermark, label="Watermarks")

# Extract clean text from any page
clean_text = pdf.pages[0].extract_text()  # Headers/footers automatically excluded
```

## Remove Noise from Scanned Documents

For scanned PDFs, apply OCR first, then filter by confidence:

```python
# Apply OCR
page.apply_ocr(engine='easyocr', languages=['en'])

# Only use high-confidence OCR text
reliable_text = page.find_all('text[source=ocr][confidence>=0.8]')
clean_text = reliable_text.extract_text()

# Or exclude low-confidence noise
noisy_text = page.find_all('text[source=ocr][confidence<0.5]')
for noise in noisy_text:
    page.add_exclusion(noise)
```

## Handle Multi-Column Layouts

Extract text from specific columns or sections:

```python
# Extract just the main content column (avoiding sidebars)
main_column = page.create_region(
    x0=page.width * 0.1,      # Start 10% from left
    top=page.height * 0.15,   # Skip header area
    x1=page.width * 0.7,      # End before sidebar
    bottom=page.height * 0.9   # Stop before footer
)

column_text = main_column.extract_text()
```

## Visual Debugging

See what you're excluding before committing:

```python
# Highlight what you're about to exclude
header = page.create_region(0, 0, page.width, 50)
footer = page.create_region(0, page.height - 30, page.width, page.height)

# Show the page to verify
header.show()
footer.show()

# If it looks right, apply the exclusions
page.add_exclusion(header)
page.add_exclusion(footer)

# Review what it looks like
page.show(exclusions='red')
```

## Compare Before and After

```python
# Extract with and without exclusions to see the difference
full_text = page.extract_text(use_exclusions=False)
clean_text = page.extract_text(use_exclusions=True)

print(f"Original: {len(full_text)} characters")
print(f"Clean: {len(clean_text)} characters")
print(f"Removed: {len(full_text) - len(clean_text)} characters")
```

## Common Patterns

### Corporate Reports
```python
# Remove headers with logos and contact info
page.add_exclusion(page.create_region(0, 0, page.width, 80))

# Remove footers with page numbers and dates
page.add_exclusion(page.create_region(0, page.height - 40, page.width, page.height))
```

### Academic Papers
```python
# Remove running headers with paper title
header = page.find('text[size<=10]').above() if page.find('text[size<=10]') else None
if header:
    page.add_exclusion(header)

# Remove footnotes (small text at bottom)
footnotes = page.find_all('text[size<=8]')
for note in footnotes:
    if note.top > page.height * 0.8:  # Bottom 20% of page
        page.add_exclusion(note)
```

### Government Documents
```python
# Remove classification markings
classifications = page.find_all('text:contains("CONFIDENTIAL|SECRET|UNCLASSIFIED")', regex=True)
for mark in classifications:
    page.add_exclusion(mark)

# Remove agency headers
agency_header = page.find('text:contains("Department of|Agency|Office of")', regex=True)
if agency_header:
    page.add_exclusion(agency_header.above())
```

## When Things Go Wrong

- **Problem**: Headers vary between pages
- **Solution**: Use adaptive exclusions

```py
def smart_header_exclusion(page):
    # Look for common header patterns
    logo = page.find('image')
    company_name = page.find('text:contains("ACME Corp")')

    if logo:
        return logo.above()
    elif company_name and company_name.top < page.height * 0.2:
        return company_name.above()
    else:
        return page.create_region(0, 0, page.width, 60)  # Fallback

pdf.add_exclusion(smart_header_exclusion)
```

- **Problem**: Need to preserve some header information
- **Solution**: Extract before excluding

```py
# Get the document title from the header first
title = page.find('text[size>=14]:bold')
document_title = title.text if title else "Unknown"

# Then exclude the header for clean body text
page.add_exclusion(page.create_region(0, 0, page.width, 100))
body_text = page.extract_text()
```

## Handling Right-to-left (Arabic, Hebrew) Text
Natural-PDF now automatically detects bidirectional (RTL) lines and applies the Unicode **BiDi** algorithm when you call `page.extract_text()`.
This means the returned string is in *logical* reading order with brackets/parentheses correctly mirrored and Western digits left untouched.

```python
# Arabic example – no special flags required
page = pdf.pages[0]
body = page.extract_text()  # parentheses and numbers appear correctly

# String queries work naturally
row = page.find("text:contains('الجريدة الرسمية')")

# Disable the BiDi pass if you need raw PDF order
raw = page.extract_text(bidi=False)
```

> Tip: This RTL handling is line-aware, so mixed LTR/RTL documents (e.g. Arabic with English dates) still extract correctly without affecting Latin text on other pages.
