# Structured Data Extraction

Extracting specific, structured information (like invoice numbers, dates, or addresses) from documents often requires more than simple text extraction. Natural PDF integrates with Large Language Models (LLMs) via Pydantic schemas to achieve this.

## Introduction

This feature allows you to define the exact data structure you want using a Pydantic model and then instruct an LLM to populate that structure based on the content of a PDF element (like a `Page` or `Region`).

## Basic Extraction

1.  **Define a Schema:** Create a Pydantic model for your desired data.
2.  **Extract:** Use the `.extract()` method on a `PDF`, `Page`, or `Region` object.
3.  **Access:** Use the `.extracted()` method to retrieve the results.

```python
from natural_pdf import PDF
from pydantic import BaseModel, Field
from openai import OpenAI # Example client

# Example: Initialize your LLM client
client = OpenAI() 

# Load the PDF
pdf = PDF("path/to/your/document.pdf")
page = pdf.pages[0]

# 1. Define your schema
class InvoiceInfo(BaseModel):
    invoice_number: str = Field(description="The main invoice identifier")
    total_amount: float = Field(description="The final amount due")
    company_name: Optional[str] = Field(None, description="The name of the issuing company")

# 2. Extract data (using default analysis_key="default-structured")
page.extract(schema=InvoiceInfo, client=client) 

# 3. Access the results
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
page.extract(InvoiceInfo, client, analysis_key="invoice_header")

# Access using the specific key
header_data = page.extracted(analysis_key="invoice_header") 
company = page.extracted('company_name', analysis_key="invoice_header")
```

## Applying to Regions and Collections

The `.extract()` and `.extracted()` methods work identically on `Region` objects, allowing you to target specific areas of a page for structured data extraction.

```python
# Assuming 'header_region' is a Region object you defined
header_region.extract(InvoiceInfo, client)
company = header_region.extracted('company_name')
```

Furthermore, you can apply extraction to collections of elements (like `pdf.pages`, or the result of `pdf.find_all(...)`) using the `.apply()` method. This iterates through the collection and calls `.extract()` on each item.

```python
# Example: Extract InvoiceInfo from the first 5 pages
results = pdf.pages[:5].apply(
    'extract', 
    schema=InvoiceInfo, 
    client=client, 
    analysis_key="page_invoice_info", # Use a specific key for batch results
    overwrite=True # Allow overwriting if run multiple times
)

# Access results for the first page in the collection
first_page_company = results[0].extracted('company_name', analysis_key="page_invoice_info")
```

This provides a powerful way to turn unstructured PDF content into structured, usable data.
