# Section Extraction

Documents are often organized into logical sections like chapters, articles, or content blocks. This tutorial shows how to extract these sections using natural-pdf, using a library weeding log as an example.

```python
#%pip install "natural-pdf[all]"
```

```python
from natural_pdf import PDF

# Load the PDF using the relative path
pdf = PDF("https://github.com/jsoma/natural-pdf/raw/refs/heads/main/pdfs/Atlanta_Public_Schools_GA_sample.pdf")
page = pdf.pages[0]

# Find horizontal lines that separate book entries
horizontal_lines = page.find_all('line:horizontal')

# Visualize the potential section boundaries
horizontal_lines.highlight(color="red", label="Section Boundaries")
page.to_image()
```

```python
# Count what we found
len(horizontal_lines)
```

## Basic Section Extraction

```python
# Extract sections based on horizontal lines
# Each section starts at a horizontal line and ends at the next one
book_sections = page.get_sections(
    start_elements=horizontal_lines,
    boundary_inclusion='start'  # Include the boundary in the section
)

# Visualize each section
page.clear_highlights()
for section in book_sections:
    section.highlight()
page.to_image()
```

```python
# Display section count and preview the first section
{
    "total_sections": len(book_sections),
    "first_section_text": book_sections[0].extract_text()[:100] + "..." if book_sections else "No sections found"
}
```

## Working with Section Content

```python
# Extract and display content from the first few book entries
book_entries = []

for i, section in enumerate(book_sections[:5]):
    # Extract the section text
    text = section.extract_text().strip()
    
    # Try to parse book information
    title = ""
    author = ""
    isbn = ""
    
    # Extract title (typically the first line)
    title_match = section.find('text:contains("Title:")')
    if title_match:
        title_value = title_match.right(width=400).extract_text()
        title = title_value.strip()
    
    # Extract author
    author_match = section.find('text:contains("Author:")')
    if author_match:
        author_value = author_match.right(width=400).extract_text()
        author = author_value.strip()
    
    # Extract ISBN
    isbn_match = section.find('text:contains("ISBN:")')
    if isbn_match:
        isbn_value = isbn_match.right(width=400).extract_text()
        isbn = isbn_value.strip()
    
    # Add to our collection
    book_entries.append({
        "number": i + 1,
        "title": title,
        "author": author,
        "isbn": isbn,
        "preview": text[:50] + "..." if len(text) > 50 else text
    })

# Display the structured book entries
import pandas as pd
pd.DataFrame(book_entries)
```

## Using Different Section Boundaries

```python
page.viewer()
```

```python
# Find title elements with specific selectors
title_elements = page.find('line[width=2]').below().find_all('text[fontname="AAAAAB+font000000002a8d158a"][size=10]')
title_elements.show()
```

```python
# Extract sections starting from each title
# This now directly returns an ElementCollection
title_sections = page.get_sections(
    start_elements=title_elements,
    boundary_inclusion='start'
)

# Show the title-based sections
page.clear_highlights()
title_sections.highlight()
page.to_image()
```

```python
# Count the sections found
len(title_sections)
```

## Section Boundary Inclusion Options

```python
# Use horizontal line elements as section dividers
dividers = page.find_all('line[horizontal]')

# Compare the different boundary inclusion options
inclusion_options = {
    'none': page.get_sections(start_elements=dividers, boundary_inclusion='none'),
    'start': page.get_sections(start_elements=dividers, boundary_inclusion='start'),
    'end': page.get_sections(start_elements=dividers, boundary_inclusion='end'),
    'both': page.get_sections(start_elements=dividers, boundary_inclusion='both')
}

# Count sections with each option
section_counts = {option: len(sections) for option, sections in inclusion_options.items()}
section_counts
```

## Custom Section Boundaries

```python
# Define specific start and end points - let's extract just one book entry
# We'll look for the first and second horizontal lines
page.clear_highlights()

start_point = title_elements[0]
end_point = title_elements[1]

# Extract the section between these points
single_book_entry = page.get_sections(
    start_elements=[start_point],
    end_elements=[end_point],
    boundary_inclusion='start'  # Include the start but not the end
)
    
# Visualize the custom section
single_book_entry.highlight(color="green", label="Single Book Entry")
    
print(single_book_entry[0].extract_text())

page.to_image()
```

## Multi-page Sections

```python
# Get sections across the first two pages
multi_page_sections = [] # Initialize as a list

for page_num in range(min(2, len(pdf.pages))):
    page = pdf.pages[page_num]
    
    # Find horizontal lines on this page
    title_elements = page.find('line[width=2]').below().find_all('text[fontname="AAAAAB+font000000002a8d158a"][size=10]')
    
    # Get sections for this page (returns ElementCollection)
    page_sections = page.get_sections(
        start_elements=title_elements,
        boundary_inclusion='start'
    )
    
    # Add elements from the collection to our list
    multi_page_sections.extend(page_sections) # list.extend works with iterables like ElementCollection

# Display info about each section (showing first 3)
[{
    "page": section.page.number + 1,  # 1-indexed page number for display
    "text": section.extract_text()[:50] + "..." if len(section.extract_text()) > 50 else section.extract_text()
} for section in multi_page_sections]
```

## Building a Book Database

```python
# Extract all book entries across multiple pages
book_database = []

# Process first 3 pages (or fewer if the document is shorter)
for page_num in range(min(3, len(pdf.pages))):
    page = pdf.pages[page_num]
    
    # Find horizontal lines on this page
    title_elements = page.find('line[width=2]').below().find_all('text[fontname="AAAAAB+font000000002a8d158a"][size=10]')
    
    # Get sections for this page
    book_sections = page.get_sections(
        start_elements=title_elements,
        boundary_inclusion='start'
    )
    
    # Process each book section
    for section in book_sections:
        # Skip sections that are too short (might be headers/footers)
        if len(section.extract_text()) < 50:
            continue
            
        # Extract book information
        book_info = {"page": page_num + 1}
        
        for field in ["Title:", "Author:", "ISBN:", "Publisher:", "Copyright:"]:
            field_element = section.find(f'text:contains("{field}")')
            if field_element:
                field_name = field.strip(':').lower()
                field_value = field_element.extract_text().replace(field, '').strip()
                book_info[field_name] = field_value

        # Below the field name
        for field in ["Price", "Acquired", "Barcode", "Removed By"]:
            field_element = section.find(f'text:contains("{field}")')
            if field_element:
                field_name = field.lower()
                field_value = field_element.below(height=10, width='element').expand(right=50).extract_text().strip()
                book_info[field_name] = field_value

        book_database.append(book_info)

# Display sample entries (first 3)
import pandas as pd

df = pd.json_normalize(book_database)
df.head()
```

Section extraction lets you break down documents into logical parts, making it easier to generate summaries, extract specific content, and create structured data from semi-structured documents. In this example, we've shown how to convert a PDF library catalog into a structured book database. 