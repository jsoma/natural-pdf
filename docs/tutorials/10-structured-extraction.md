# Structured Extraction with LLMs

Use `.extract()` to pull structured data from PDFs using an LLM. Define a Pydantic schema, pass an OpenAI-compatible client, and get back typed results with optional citations and confidence scores.

## Basic Usage

```python
import os
from pydantic import BaseModel
from openai import OpenAI
from natural_pdf import PDF

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

class InvoiceData(BaseModel):
    invoice_number: str
    date: str
    total: float
    vendor: str

pdf = PDF("invoice.pdf")
page = pdf.pages[0]

result = page.extract(InvoiceData, client=client)

# Attribute access
print(result.invoice_number)   # "INV-2024-00789"
print(result.total)            # 1250.00

# Item access gives FieldResult objects
print(result["vendor"].value)  # "Acme Corp"

# Convert to dict
print(result.to_dict())        # {"invoice_number": "INV-2024-00789", ...}

# Iterate
for name, field in result:
    print(f"{name}: {field.value}")

pdf.close()
```

## Text vs Vision Mode

By default `.extract()` sends the page's text layer to the LLM. For scanned documents or when layout matters visually, use `using='vision'`:

```python
# Text mode (default) — sends extracted text
result = page.extract(MySchema, client=client, using="text")

# Vision mode — sends a rendered image of the page
result = page.extract(MySchema, client=client, using="vision")
```

## Custom Prompts and Instructions

Override the default prompt entirely or append domain-specific guidance:

```python
# Custom prompt
result = page.extract(
    MySchema,
    client=client,
    prompt="Extract inspection details. Dates should be in ISO format.",
)

# Instructions — appended to the default (or custom) prompt
result = page.extract(
    MySchema,
    client=client,
    instructions="Monetary values should be in USD. If a field is ambiguous, prefer null.",
)
```

## Citations

Add `citations=True` to trace each extracted value back to the specific PDF elements it came from:

```python
result = page.extract(MySchema, client=client, citations=True)

# Each field has a .citations ElementCollection
result["vendor"].citations       # ElementCollection of source TextElements
result["vendor"].citations.show()  # Highlight sources on the page

# Show all citations at once
result.show()

# Access all citations as a dict
result.all_citations  # {"vendor": ElementCollection, "date": ElementCollection, ...}
```

Citations work by sending line-numbered text to the LLM and asking it to return verbatim quotes. These quotes are then aligned back to PDF elements using pdfplumber's TextMap provenance data.

**Note:** Citations require `using='text'` (the default). They are not supported with `using='vision'`.

## Confidence Scoring

Add `confidence=True` to get a 0.0–1.0 confidence score for each extracted field:

```python
result = page.extract(MySchema, client=client, confidence=True)

result["vendor"].confidence  # 0.95
result.confidences           # {"vendor": 0.95, "date": 0.85, ...}
```

The prompt asks the LLM to self-report confidence using these anchors. These are the LLM's own assessments — they are not calibrated or independently verified:

| Score | Prompt Anchor |
|-------|---------------|
| 0.0 | Not present or completely uncertain |
| 0.2 | Weakly implied but not stated |
| 0.5 | Partially supported or ambiguous |
| 0.8 | Supported with minor inference |
| 1.0 | Explicitly stated in the text |

### Categorical Confidence

Instead of numeric scores, use a list of levels. You define what they mean, or the LLM interprets them from the label:

```python
result = page.extract(
    MySchema,
    client=client,
    confidence=["low", "medium", "high"],
)
result["vendor"].confidence  # "high"
```

Or provide explicit descriptions for each level:

```python
result = page.extract(
    MySchema,
    client=client,
    confidence={
        "low": "implied or inferred",
        "medium": "strongly implied",
        "high": "clearly and explicitly stated",
    },
)
```

## Annotated PDF Export

Save extraction results as a native PDF with highlight annotations and a sidebar legend:

```python
result = page.extract(MySchema, client=client, citations=True, confidence=True)
result.save_pdf("annotated.pdf")
```

**Install:** `pip install "natural-pdf[export]"` (requires pikepdf).

Each field's citation elements become `/Highlight` annotations on the corresponding pages. The sidebar shows field names, extracted values, and colors matching the highlights.

You can also visualize inline:

```python
result.show()  # Displays enriched legend labels (field name + value)
```

### Controlling which pages appear

Both `.show()` and `.save_pdf()` accept a `pages` parameter:

```python
# .show() defaults to pages="cited" — only pages with citation elements
result.show()                     # cited pages only
result.show(pages="all")          # every page in the source PDF

# .save_pdf() defaults to pages="all" — the full source PDF with annotations
result.save_pdf("annotated.pdf")                   # all pages
result.save_pdf("annotated.pdf", pages="cited")    # only annotated pages
```

## Extracting from Regions, Pages, and PDFs

`.extract()` works on pages, regions, and entire PDFs:

```python
# From a specific region
header = page.find('text:contains("Invoice")').below(until='text:contains("Items")')
result = header.extract(MySchema, client=client)

# From an entire PDF (multi-page)
result = pdf.extract(MySchema, client=client, citations=True)
```

## Choosing a Model

Pass `model=` to select which LLM to use:

```python
result = page.extract(MySchema, client=client, model="gpt-4o-mini")
```
