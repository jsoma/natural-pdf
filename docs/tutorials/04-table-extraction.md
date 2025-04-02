# Basic Table Extraction

PDFs often contain tables, and `natural-pdf` provides methods to extract their data, building on `pdfplumber`'s capabilities.

Let's extract the "Violations" table from our practice PDF.

```python
#%pip install "natural-pdf[all]"
```


```python
from natural_pdf import PDF

# Load a PDF
pdf = PDF("https://github.com/jsoma/natural-pdf/raw/refs/heads/main/pdfs/01-practice.pdf")
page = pdf.pages[0]

# Use extract_tables() to find all tables on the page.
# It returns a list of tables, where each table is a list of lists.
tables_data = page.extract_tables()

# Display the first table found
tables_data[0] if tables_data else "No tables found"

# You can also visualize the general area of the first table 
# by finding elements in that region
if tables_data:
    # Find a header element in the table
    statute_header = page.find('text:contains("Statute")')
    if statute_header:
        # Show the area
        statute_header.below(height=100).highlight(color="green", label="Table Area")
        page.to_image()
```

This code uses `page.extract_tables()` which attempts to automatically detect tables based on visual cues like lines and whitespace. The result is a list of lists, representing the rows and cells of the table.

<div class="admonition note">
<p class="admonition-title">Table Settings and Limitations</p>

    The default `extract_tables()` works well for simple, clearly defined tables. However, it might struggle with:
    *   Tables without clear borders or lines.
    *   Complex merged cells.
    *   Tables spanning multiple pages.

    `pdfplumber` (and thus `natural-pdf`) allows passing `table_settings` dictionaries to `extract_tables()` for more control over the detection strategy (e.g., `"vertical_strategy": "text"`, `"horizontal_strategy": "text"`).

    For even more robust table detection, especially for tables without explicit lines, using Layout Analysis (like `page.analyze_layout(engine='tatr')`) first, finding the table `region`, and then calling `region.extract_table()` can yield better results. We'll explore layout analysis in a later tutorial.
</div> 