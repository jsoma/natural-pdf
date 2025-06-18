# Pulling Structured Data from PDFs

Ever had a pile of invoices, reports, or forms where you need to extract the same pieces of information from each one? That's where structured data extraction shines. Instead of manually copying invoice numbers and dates, you can tell Natural PDF exactly what information you want and let it find those details automatically.

You'll need more than the basic install for this:
```bash
# Install the OpenAI (or compatible) client library
pip install openai

# Or pull in the full AI stack (classification, QA, search, etc.)
pip install "natural_pdf[ai]"
```

## The Simple Approach: Just Tell It What You Want

Don't want to mess around with schemas? Just make a list of what you're looking for:

```python
from natural_pdf import PDF

pdf = PDF("https://github.com/jsoma/natural-pdf/raw/refs/heads/main/pdfs/01-practice.pdf")
page = pdf.pages[0]

# Extract data using just a list - no schema required!
data = page.extract(schema=["site", "date", "violation count", "inspector"]).extracted()

print(data.site)  # "ACME Manufacturing Plant"
print(data.date)  # "2024-03-15"
print(data.violation_count)  # "3"
```

Natural PDF automatically builds a schema behind the scenes and extracts the data. Each field becomes a string, and you get confidence scores for free:

```python
# Check how confident the extraction was
print(data.site_confidence)  # 0.89
print(data.date_confidence)  # 0.95
```

This works completely offline - no API keys or internet connection needed. It uses a local document question-answering model that understands both text and layout.

## Working Offline (No Internet Required)

Maybe you're dealing with sensitive documents or just don't want to send everything to the cloud:

```python
# This works completely offline
page.extract(schema=["company", "total", "due_date"])
```

The offline engine is pretty smart - it looks at both the text content and how things are visually laid out on the page. For sketchy results, you can set a confidence threshold:

```python
# Only accept answers the model is confident about
page.extract(schema=["amount", "date"], min_confidence=0.8)
```

If an answer falls below your threshold, it gets set to `None` instead of giving you questionable data.

Want to use a local LLM instead? Tools like [LM Studio](https://lmstudio.ai/) or [Msty](https://msty.app/) can run models locally with an OpenAI-compatible API:

```python
from openai import OpenAI

# Point to your local LLM server
client = OpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")

page.extract(schema=InvoiceSchema, client=client)
```

Just heads up - local LLMs are much slower than the document QA approach for simple extractions!

## Building Custom Schemas

For more complex extractions, you can define exactly what you want using Pydantic:

```python
from natural_pdf import PDF
from pydantic import BaseModel, Field
from openai import OpenAI

# Set up your LLM client (using Gemini here)
client = OpenAI(
    api_key="YOUR_API_KEY",
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/" 
)

# Load the PDF
pdf = PDF("https://github.com/jsoma/natural-pdf/raw/refs/heads/main/pdfs/01-practice.pdf")
page = pdf.pages[0]

# Define exactly what you want to extract
class ReportInfo(BaseModel):
    inspection_number: str = Field(description="The main report identifier")
    inspection_date: str = Field(description="When the inspection happened")
    inspection_service: str = Field(description="Name of inspection service")
    site_name: str = Field(description="Location that was inspected")
    summary: str = Field(description="Visit summary")
    violation_count: int = Field(description="Number of violations found")

# Extract the data
page.extract(schema=ReportInfo, client=client, model="gemini-2.5-flash") 

# Get your structured data
report_data = page.extracted() 
print(report_data.inspection_number)
print(report_data.violation_count)
```

## Managing Multiple Extractions

- Results get stored under the key `"default-structured"` by default
- Use `analysis_key` to store multiple different extractions from the same document
- Trying to extract with an existing key will fail unless you use `overwrite=True`

```python
# Extract using a specific key
page.extract(InvoiceInfo, client=client, analysis_key="invoice_header")

# Access that specific extraction
header_data = page.extracted(analysis_key="invoice_header") 
company = page.extracted('company_name', analysis_key="invoice_header")
```

## Text vs Vision Extraction

You can choose how to send the document to the LLM:

- `using='text'` (default): Sends the text content with layout preserved
- `using='vision'`: Sends an image of the page

```python
# Send text content (faster, cheaper)
page.extract(schema=MySchema, client=client, using='text')

# Send page image (better for visual layouts)
page.extract(schema=MySchema, client=client, using='vision')
```

## Processing Multiple Documents

The extraction methods work on any part of a PDF - regions, pages, collections - making it easy to process lots of documents:

```python
# Extract from a specific region
header_region.extract(InvoiceInfo, client=client)
company = header_region.extracted('company_name')

# Process multiple pages at once
results = pdf.pages[:5].apply(
    lambda page: page.extract(
        schema=InvoiceInfo, 
        client=client, 
        analysis_key="page_invoice_info"
    )
)

# Access results for any page
pdf.pages[0].extracted('company_name', analysis_key="page_invoice_info")
```

This approach lets you turn unstructured PDF content into clean, structured data you can actually work with.
