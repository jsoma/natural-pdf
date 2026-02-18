# Finding Sections

Extract content between section headers - "get everything from 'Summary' to 'Conclusion'".

**When to use this pattern:**

- Extract specific chapters or sections from reports
- Pull content between known headers
- Get narrative text bounded by structural elements

## The Problem

You have a document with multiple sections. You need to extract just the "Financial Highlights" section, which starts after that header and ends before "Risks and Challenges".

## Sample PDF

This tutorial uses `pdfs/cookbook/annual_report.pdf` - a report with multiple titled sections.

```python
import natural_pdf as npdf

pdf = npdf.PDF("pdfs/cookbook/annual_report.pdf")
pdf.pages[0].show()
```

## Basic Pattern: Using `until`

The `until` parameter in directional methods stops the region at a matching element:

```python
import natural_pdf as npdf

pdf = npdf.PDF("pdfs/cookbook/annual_report.pdf")
page = pdf.pages[0]

# Find the section header (use partial text - PDF text can be split)
header = page.find('text:contains("HIGHLIGHTS")')

if header:
    # Get content until the next section header
    section = header.below(until='text:contains("OPERATIONAL")')
    content = section.extract_text()
    print(content)

pdf.close()
```

**Note:** PDF text elements can be split unexpectedly. Use partial matches or multiple search terms.

## Finding Headers by Style

Section headers often have consistent styling. Use attribute selectors:

```python
# Find headers by font size
headers = page.find_all('text[size>=13]:bold')
for h in headers:
    print(h.extract_text())

# Find headers by color (if colored)
# headers = page.find_all('text[color=#1F4E79]')
```

## Extracting Multiple Sections

Process all sections in a document:

```python
import natural_pdf as npdf

pdf = npdf.PDF("pdfs/cookbook/annual_report.pdf")
page = pdf.pages[0]

# Find all section headers
headers = page.find_all('text:bold[size>=13]')

sections = {}
header_list = list(headers)

for i, header in enumerate(header_list):
    title = header.extract_text().strip()

    # Skip the main title
    if 'ANNUAL REPORT' in title:
        continue

    # Get content until next header (or end of page)
    if i < len(header_list) - 1:
        next_header = header_list[i + 1]
        # Use the next header's selector as the boundary
        content_region = header.below(until=f'text:contains("{next_header.extract_text()[:20]}")')
    else:
        # Last section - get to end of page
        content_region = header.below()

    sections[title] = content_region.extract_text().strip()

# Print sections
for title, content in sections.items():
    print(f"\n=== {title} ===")
    print(content[:200] + "..." if len(content) > 200 else content)

pdf.close()
```

## Using Generic Header Patterns

When you don't know exact header text, use patterns:

```python
# Stop at ANY bold text (next header)
section = header.below(until='text:bold')

# Stop at any large text
section = header.below(until='text[size>=14]')

# Stop at horizontal line
section = header.below(until='line:horizontal')

# Stop at page number pattern
section = header.below(until='text:contains("Page ")')
```

## Sections Spanning Multiple Pages

Use `multipage=True` for sections that cross page boundaries:

```python
# Find section start
header = page.find('text:contains("EXECUTIVE SUMMARY")')

# Get content across pages until next major section
section = header.below(
    until='text:contains("FINANCIAL HIGHLIGHTS")',
    multipage=True
)

text = section.extract_text()
```

## Extracting Subsections

Some documents have nested sections. Handle them hierarchically:

```python
# Find main section
main_section = page.find('text:contains("OPERATIONAL ACHIEVEMENTS")')
main_region = main_section.below(until='text:contains("STRATEGIC INITIATIVES")')

# Find subsections within the main section
bullets = main_region.find_all('text:contains("•")')
for bullet in bullets:
    # Get text of each bullet point
    bullet_text = bullet.extract_text().strip()
    print(f"  - {bullet_text}")
```

## Extracting Tables Within Sections

```python
# Find the financial section
fin_header = page.find('text:contains("FINANCIAL HIGHLIGHTS")')
fin_section = fin_header.below(until='text:contains("OPERATIONAL")')

# Extract table from within that section
table = fin_section.extract_table()
if table:
    df = table.to_df()
    print(df)
```

## Complete Example: Report Section Extractor

```python
import natural_pdf as npdf

def extract_report_sections(pdf_path):
    """Extract all sections from a report PDF."""
    pdf = npdf.PDF(pdf_path)

    all_sections = {}

    for page_num, page in enumerate(pdf.pages):
        # Find section headers (bold, large text)
        headers = page.find_all('text:bold[size>=13]')

        for i, header in enumerate(headers):
            title = header.extract_text().strip()

            # Skip document title and empty headers
            if not title or 'ANNUAL REPORT' in title or 'continued' in title.lower():
                continue

            # Determine boundary
            remaining_headers = list(headers)[i+1:]
            if remaining_headers:
                # Stop at next header on same page
                next_title = remaining_headers[0].extract_text()[:20]
                content_region = header.below(until=f'text:contains("{next_title}")')
            else:
                # Last header on page - check for continuation
                content_region = header.below(multipage=True)

            content = content_region.extract_text().strip()

            # Remove the header text from content if included
            if content.startswith(title):
                content = content[len(title):].strip()

            all_sections[title] = {
                'page': page_num + 1,
                'content': content,
                'word_count': len(content.split())
            }

    pdf.close()
    return all_sections

# Usage
sections = extract_report_sections("pdfs/cookbook/annual_report.pdf")

for title, info in sections.items():
    print(f"\n{title} (Page {info['page']}, {info['word_count']} words)")
    print("-" * 50)
    preview = info['content'][:300]
    print(preview + "..." if len(info['content']) > 300 else preview)
```

## Building a Table of Contents

```python
def build_toc(pdf_path):
    """Build a table of contents from section headers."""
    pdf = npdf.PDF(pdf_path)

    toc = []
    for page_num, page in enumerate(pdf.pages):
        headers = page.find_all('text:bold[size>=13]')

        for header in headers:
            title = header.extract_text().strip()
            if title and 'ANNUAL REPORT' not in title:
                toc.append({
                    'title': title,
                    'page': page_num + 1,
                    'y_position': header.y0  # For ordering
                })

    pdf.close()

    # Sort by page then position
    toc.sort(key=lambda x: (x['page'], x['y_position']))
    return toc

toc = build_toc("pdfs/cookbook/annual_report.pdf")
for entry in toc:
    print(f"Page {entry['page']}: {entry['title']}")
```

## Troubleshooting

### "Section boundary not found"

The `until` selector might not match. Debug with:

```python
# See what elements exist
all_bold = page.find_all('text:bold')
for elem in all_bold:
    print(f"Found: '{elem.extract_text()}'")
```

### "Getting content from wrong section"

Make sure the header selector is specific enough:

```python
# Too generic - might match wrong element
header = page.find('text:contains("Summary")')

# More specific
header = page.find('text:bold:contains("EXECUTIVE SUMMARY")')
```

### "Content includes the next header"

The `until` element is excluded by default. If you're seeing it, check your selector:

```python
# Make sure until selector matches the header element
section = header.below(until='text:bold:contains("NEXT SECTION")')
```

## Next Steps

- [Label-Value Extraction](label-value-extraction.md) - Extract data within sections
- [Multipage Content](multipage-content.md) - Handle sections across pages
- [One Page = One Row](one-page-one-row.md) - Process repeating document structures
