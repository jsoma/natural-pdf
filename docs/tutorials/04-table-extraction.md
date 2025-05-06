# Basic Table Extraction

PDFs often contain tables, and `natural-pdf` provides methods to extract their data, building on `pdfplumber`'s capabilities.

Let's extract the "Violations" table from our practice PDF.

```python
#%pip install "natural-pdf[all]"
```

## pdfplumber-based extraction

```python
from natural_pdf import PDF

pdf = PDF("https://github.com/jsoma/natural-pdf/raw/refs/heads/main/pdfs/01-practice.pdf")
page = pdf.pages[0]

tables = page.extract_tables()
tables[0]
```

## TATR-based extraction

When you do a TATR layout analysis, it uses a little magic to find borders and boundaries. A region analyzed by TATR will automatically use the `tatr` extraction method.

```python
from natural_pdf import PDF

pdf = PDF("https://github.com/jsoma/natural-pdf/raw/refs/heads/main/pdfs/01-practice.pdf")
page = pdf.pages[0]

page.analyze_layout('tatr')
page.find('table').show()
```

```python
page.find('table').extract_table()
```