# Extract Data from Forms and Invoices

You have a stack of invoices, forms, or structured documents where you need to pull out specific pieces of information - invoice numbers, totals, dates, names, etc. Here's how to automate that extraction.

## The Problem

Manual data entry from PDFs is slow and error-prone. You need to:
- Extract the same fields from hundreds of similar documents
- Handle slight variations in layout between documents  
- Get structured data you can actually work with
- Maintain accuracy while processing quickly

## Quick Solution: List the Fields You Want

Don't overthink it - just tell Natural PDF what information you're looking for:

```python
from natural_pdf import PDF

pdf = PDF("https://github.com/jsoma/natural-pdf/raw/refs/heads/main/pdfs/01-practice.pdf")
page = pdf.pages[0]

# Extract data using a simple list that matches the inspection report columns
data = page.extract(schema=["site", "violation count", "date", "inspection number", "summary"]).extracted()

# Access the extracted information
print(f"Site: {data.site}")
print(f"Violations: {data.violation_count}")
print(f"Date: {data.date}")
print(f"Inspection #: {data.inspection_number}")

# Check confidence levels
print(f"Confidence – Site: {data.site_confidence:.2f}")
print(f"Confidence – Violations: {data.violation_count_confidence:.2f}")
```

This works completely offline using document question-answering models.

## For Complex Data: Use Pydantic Schemas

When you need more control over data types and validation:

```python
from pydantic import BaseModel, Field
from openai import OpenAI

# Define exactly what you want to extract for the inspection report
class InspectionReport(BaseModel):
    site_name: str = Field(description="Name of the inspection site")
    violation_count: int = Field(description="Number of violations found")
    inspection_date: str = Field(description="Inspection date in any format")
    inspection_number: str = Field(description="Inspection reference ID")
    summary: str = Field(description="Inspection summary paragraph")

# Set up LLM client (using Anthropic here)
client = OpenAI(
    api_key="your-api-key",
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# Extract structured data
page.extract(schema=InspectionReport, client=client, model="gemini-2.0-flash")

try:
    report_data = page.extracted()
    print(f"Site: {report_data.site_name}")
    print(f"Violations: {report_data.violation_count}")
    print(f"Inspection #: {report_data.inspection_number}")
except Exception as e:
    print("Extraction failed with error", e)
```

## Handle Different Document Layouts

For documents that vary in structure, use spatial hints:

```python
# Sometimes data is in specific areas of the page
header_region = page.create_region(0, 0, page.width, page.height * 0.3)
footer_region = page.create_region(0, page.height * 0.7, page.width, page.height)

# Extract company info from header
company_data = header_region.extract(
    schema=["company name", "address", "phone"]
).extracted()

# Extract totals from footer  
totals_data = footer_region.extract(
    schema=["subtotal", "tax", "total"]
).extracted()
```

## Process Multiple Documents

Batch process a folder of similar documents:

```python
import os
from pathlib import Path

# Define your extraction schema
class FormData(BaseModel):
    applicant_name: str
    application_date: str  
    reference_number: str
    status: str = Field(default="unknown")

# Process all PDFs in a folder
form_results = []
pdf_folder = Path("forms/")

for pdf_file in pdf_folder.glob("*.pdf"):
    print(f"Processing {pdf_file.name}...")
    
    pdf = PDF(str(pdf_file))
    page = pdf.pages[0]  # Assuming single-page forms
    
    # Extract data
    page.extract(schema=FormData, client=client)
    data = page.extracted()
    
    # Add filename for tracking
    result = {
        "filename": pdf_file.name,
        "applicant_name": data.applicant_name,
        "application_date": data.application_date,
        "reference_number": data.reference_number,
        "status": data.status
    }
    form_results.append(result)
    
    pdf.close()  # Clean up

# Save results to CSV
import pandas as pd
df = pd.DataFrame(form_results)
df.to_csv("extracted_form_data.csv", index=False)
print(f"Processed {len(form_results)} forms")
```

## Handle Scanned Documents

For image-based PDFs, apply OCR first:

```python
# Apply OCR before extraction
page.apply_ocr(engine='easyocr', languages=['en'])

# Filter out low-confidence OCR text to avoid noise
reliable_text = page.find_all('text[source=ocr][confidence>=0.8]')
print(f"Using {len(reliable_text)} high-confidence OCR elements")

# Now extract data (works on OCR'd text)
data = page.extract(schema=["invoice number", "total", "date"]).extracted()
```

## Common Form Patterns

## Validation and Error Handling

Check your extracted data for common issues:

```py
def validate_invoice_data(data):
    issues = []
    
    # Check for missing required fields
    if not data.invoice_number or data.invoice_number.strip() == "":
        issues.append("Missing invoice number")
    
    # Validate amounts
    if data.total_amount <= 0:
        issues.append("Invalid total amount")
    
    # Check date format
    try:
        from datetime import datetime
        datetime.strptime(data.invoice_date, "%Y-%m-%d")
    except ValueError:
        # Try common date formats
        common_formats = ["%m/%d/%Y", "%d/%m/%Y", "%B %d, %Y"]
        date_valid = False
        for fmt in common_formats:
            try:
                datetime.strptime(data.invoice_date, fmt)
                date_valid = True
                break
            except ValueError:
                continue
        if not date_valid:
            issues.append(f"Invalid date format: {data.invoice_date}")
    
    return issues

# Validate extracted data
validation_issues = validate_invoice_data(invoice_data)
if validation_issues:
    print("Data quality issues found:")
    for issue in validation_issues:
        print(f"- {issue}")
else:
    print("Data validation passed!")
```

## Improve Accuracy with Context

Give the AI more context for better extraction:

```py
# Add context about the document type
extraction_prompt = """
This is a medical insurance claim form. 
Extract the following information, paying attention to:
- Policy numbers are usually 10-12 digits
- Claim amounts should be in dollars
- Dates should be in MM/DD/YYYY format
- Provider names are usually at the top of the form
"""

class InsuranceClaim(BaseModel):
    policy_number: str = Field(description="Insurance policy number (10-12 digits)")
    claim_amount: float = Field(description="Total claim amount in USD")
    service_date: str = Field(description="Date of service in MM/DD/YYYY format")
    provider_name: str = Field(description="Healthcare provider name")
    patient_name: str = Field(description="Patient full name")

# Use custom prompt for better results
page.extract(
    schema=InsuranceClaim, 
    client=client,
    prompt=extraction_prompt
)
```

## Debug Extraction Issues

When extraction isn't working well:

```py
# 1. Check what text the AI can actually see
extracted_text = page.extract_text()
print("Available text:")
print(extracted_text[:500])  # First 500 characters

# 2. Try extracting with lower confidence threshold
data = page.extract(
    schema=["invoice number", "total"], 
    min_confidence=0.5  # Lower threshold
).extracted()

# 3. Check confidence scores for each field
for field_name in data.__fields__:
    confidence_field = f"{field_name}_confidence"
    if hasattr(data, confidence_field):
        confidence = getattr(data, confidence_field)
        value = getattr(data, field_name)
        print(f"{field_name}: '{value}' (confidence: {confidence:.2f})")

# 4. Try vision mode if text mode fails
if any(getattr(data, f"{field}_confidence", 0) < 0.7 for field in ["invoice_number", "total"]):
    print("Low confidence detected, trying vision mode...")
    page.extract(schema=["invoice number", "total"], client=client, using='vision')
    data = page.extracted()
```

