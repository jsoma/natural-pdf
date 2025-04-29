# Structured Data Extraction

Extracting specific, structured information (like invoice numbers, dates, or addresses) from documents often requires more than simple text extraction. Natural PDF integrates with LLMs to pull out [structured data](https://platform.openai.com/docs/guides/structured-outputs).

You need to install more than just the tiny baby default `natural_pdf` for this:
```
# Install just the LLM portions
pip install "natural_pdf[llm]"

# Install eeeeeverything
pip install "natural_pdf[all]"
```

## Introduction

This feature allows you to define the exact data structure you want using a Pydantic model and then instruct an LLM to populate that structure based on the content of a PDF element (like a `Page` or `Region`).

> Not sure how to write a Pydantic schema? Just ask an LLM! "Write me a Pydantic schema to pull out an invoice number (an integer), a company name (string) and a date (string)." It'll go fine.

## Basic Extraction

1.  **Define a Schema:** Create a Pydantic model for your desired data.
2.  **Extract:** Use `.extract()` on a `PDF`, `Page`, or `Region` object.
3.  **Access:** Use `.extracted()` to retrieve the results.

```python
from natural_pdf import PDF
from pydantic import BaseModel, Field
from openai import OpenAI

# Initialize your LLM client
# Anything OpenAI-compatible works!
client = OpenAI(
    api_key="ANTHROPIC_API_KEY",  # Your Anthropic API key
    base_url="https://api.anthropic.com/v1/"  # Anthropic's API endpoint
)

# Load the PDF
pdf = PDF("path/to/your/document.pdf")
page = pdf.pages[0]

# Define your schema
class InvoiceInfo(BaseModel):
    invoice_number: str = Field(description="The main invoice identifier")
    total_amount: float = Field(description="The final amount due")
    company_name: Optional[str] = Field(None, description="The name of the issuing company")

# Extract data
page.extract(schema=InvoiceInfo, client=client) 

# Access the full result object
full_data = page.extracted() 
print(full_data)

# Access a single field
inv_num = page.extracted('invoice_number')
print(f"Invoice Number: {inv_num}") 
```

## Keys and Overwriting

- By default, results are stored under the key `"default-structured"` in the element's `.analyses` dictionary.
- Use the `analysis_key` parameter in `.extract()` to store results under a different name (e.g., `analysis_key="customer_details"`).
- Attempting to extract using an existing `analysis_key` will raise an error unless `overwrite=True` is specified.

```python
# Extract using a specific key
page.extract(InvoiceInfo, client=client, analysis_key="invoice_header")

# Access using the specific key
header_data = page.extracted(analysis_key="invoice_header") 
company = page.extracted('company_name', analysis_key="invoice_header")
```

## Text vs vision

When sending a page (or a region or etc) to an LLM, you can choose either `using='text'` (default) or `using='vision'`.

- `text` sends the text, somewhat respecting layout using `.extract_text(layout=True)`
- `vision` sends an image of the page with `.to_image(resolution=72)` (no highlights or labels)

## Batch and bulk extraction

If you have a lot of pages or a lot of PDFs or a lot of anything, the `.extract()` and `.extracted()` methods work identically on most parts of a PDF - regions, pages, collections of pdfs, etc, allowing a lot of flexibility in what you analyze.

```python
# Assuming 'header_region' is a Region object you defined
header_region.extract(InvoiceInfo, client)
company = header_region.extracted('company_name')
```

Furthermore, you can apply extraction to collections of elements (like `pdf.pages`, or the result of `pdf.find_all(...)`) using the `.apply()` method. This iterates through the collection and calls `.extract()` on each item.

```python
# Example: Extract InvoiceInfo from the first 5 pages
results = pdf.pages[:5].apply(
    lambda page: page.extract(
        client=client,
        schema=InvoiceInfo, 
        client=client, 
        analysis_key="page_invoice_info",
    )
)

# Access results for the first page in the collection
pdf.pages[0].extracted('company_name', analysis_key="page_invoice_info")
```

This provides a powerful way to turn unstructured PDF content into structured, usable data.
