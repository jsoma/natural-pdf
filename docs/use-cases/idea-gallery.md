# Idea Gallery

Real-world document types mapped to extraction patterns. Find your use case and jump to the right tutorial.

## How to Use This Page

1. Find a document type similar to yours
2. Note which patterns apply
3. Follow the linked tutorials

## Government & Public Records

| Document Type | Primary Pattern | Secondary Pattern | Key Features |
|---------------|-----------------|-------------------|--------------|
| **Inspection Reports** | [One Page = One Row](../cookbook/one-page-one-row.md) | [Messy Tables](../cookbook/messy-tables.md) | Repeating forms, violation tables |
| **FOIA Responses** | [Finding Sections](../cookbook/finding-sections.md) | [OCR Then Navigate](../cookbook/ocr-then-navigate.md) | Often scanned, redacted sections |
| **Campaign Finance** | [One Page = One Row](../cookbook/one-page-one-row.md) | [Messy Tables](../cookbook/messy-tables.md) | Donor/expenditure tables |
| **Police Incident Logs** | [Messy Tables](../cookbook/messy-tables.md) | [Multipage Content](../cookbook/multipage-content.md) | Multi-line entries, continuation rows |
| **Budget Documents** | [Multipage Content](../cookbook/multipage-content.md) | [Label-Value Extraction](../cookbook/label-value-extraction.md) | Tables spanning pages |
| **Permit Applications** | [Label-Value Extraction](../cookbook/label-value-extraction.md) | [OCR Then Navigate](../cookbook/ocr-then-navigate.md) | Form fields, sometimes handwritten |
| **Court Filings** | [Finding Sections](../cookbook/finding-sections.md) | [Label-Value Extraction](../cookbook/label-value-extraction.md) | Case metadata, structured sections |

## Financial Documents

| Document Type | Primary Pattern | Secondary Pattern | Key Features |
|---------------|-----------------|-------------------|--------------|
| **Invoices** | [Label-Value Extraction](../cookbook/label-value-extraction.md) | [Messy Tables](../cookbook/messy-tables.md) | Header fields + line items |
| **Expense Reports** | [One Page = One Row](../cookbook/one-page-one-row.md) | [Label-Value Extraction](../cookbook/label-value-extraction.md) | Receipts, grouped expenses |
| **Bank Statements** | [Messy Tables](../cookbook/messy-tables.md) | [Multipage Content](../cookbook/multipage-content.md) | Transaction tables |
| **Tax Forms** | [Label-Value Extraction](../cookbook/label-value-extraction.md) | [OCR Then Navigate](../cookbook/ocr-then-navigate.md) | Fixed layouts, OCR for handwritten |
| **Annual Reports** | [Finding Sections](../cookbook/finding-sections.md) | [Label-Value Extraction](../cookbook/label-value-extraction.md) | Narrative + financial tables |
| **SEC Filings (10-K)** | [Finding Sections](../cookbook/finding-sections.md) | [Multipage Content](../cookbook/multipage-content.md) | Long documents, nested tables |

## Legal Documents

| Document Type | Primary Pattern | Secondary Pattern | Key Features |
|---------------|-----------------|-------------------|--------------|
| **Contracts** | [Finding Sections](../cookbook/finding-sections.md) | [Label-Value Extraction](../cookbook/label-value-extraction.md) | Clause extraction, party names |
| **Real Estate Deeds** | [Label-Value Extraction](../cookbook/label-value-extraction.md) | [OCR Then Navigate](../cookbook/ocr-then-navigate.md) | Often scanned, property details |
| **NDAs** | [Finding Sections](../cookbook/finding-sections.md) | - | Standard clauses |
| **Court Orders** | [Finding Sections](../cookbook/finding-sections.md) | [Label-Value Extraction](../cookbook/label-value-extraction.md) | Case number, parties, rulings |
| **Patent Documents** | [Finding Sections](../cookbook/finding-sections.md) | [Multipage Content](../cookbook/multipage-content.md) | Claims, descriptions |

## Healthcare & Insurance

| Document Type | Primary Pattern | Secondary Pattern | Key Features |
|---------------|-----------------|-------------------|--------------|
| **Insurance Policies** | [Finding Sections](../cookbook/finding-sections.md) | [Label-Value Extraction](../cookbook/label-value-extraction.md) | Coverage details, exclusions |
| **Explanation of Benefits** | [Label-Value Extraction](../cookbook/label-value-extraction.md) | [Messy Tables](../cookbook/messy-tables.md) | Procedure codes, amounts |
| **Medical Intake Forms** | [One Page = One Row](../cookbook/one-page-one-row.md) | [OCR Then Navigate](../cookbook/ocr-then-navigate.md) | Checkboxes, handwritten |
| **Lab Reports** | [Label-Value Extraction](../cookbook/label-value-extraction.md) | [Messy Tables](../cookbook/messy-tables.md) | Test results, reference ranges |

## Academic & Research

| Document Type | Primary Pattern | Secondary Pattern | Key Features |
|---------------|-----------------|-------------------|--------------|
| **Research Papers** | [Finding Sections](../cookbook/finding-sections.md) | [Messy Tables](../cookbook/messy-tables.md) | Abstract, methods, results |
| **Syllabi** | [Finding Sections](../cookbook/finding-sections.md) | [Label-Value Extraction](../cookbook/label-value-extraction.md) | Schedule tables, grading |
| **Transcripts** | [Messy Tables](../cookbook/messy-tables.md) | [One Page = One Row](../cookbook/one-page-one-row.md) | Course listings |
| **Grant Applications** | [Finding Sections](../cookbook/finding-sections.md) | [Label-Value Extraction](../cookbook/label-value-extraction.md) | Budget tables, milestones |

## HR & Administrative

| Document Type | Primary Pattern | Secondary Pattern | Key Features |
|---------------|-----------------|-------------------|--------------|
| **Resumes/CVs** | [Finding Sections](../cookbook/finding-sections.md) | - | Work history, education |
| **Job Applications** | [Label-Value Extraction](../cookbook/label-value-extraction.md) | [One Page = One Row](../cookbook/one-page-one-row.md) | Form fields |
| **W-2 / Tax Documents** | [Label-Value Extraction](../cookbook/label-value-extraction.md) | - | Fixed box positions |
| **Employee Reviews** | [Finding Sections](../cookbook/finding-sections.md) | [Label-Value Extraction](../cookbook/label-value-extraction.md) | Ratings, comments |

## Pattern Selection Guide

### Start Here Based on Your Goal

**"I need one row per page/document"**
→ [One Page = One Row](../cookbook/one-page-one-row.md)

**"I need to find a label and get the value next to it"**
→ [Label-Value Extraction](../cookbook/label-value-extraction.md)

**"My table has problems (multi-line, merged cells, etc.)"**
→ [Messy Tables](../cookbook/messy-tables.md)

**"The content continues across multiple pages"**
→ [Multipage Content](../cookbook/multipage-content.md)

**"I need to extract a specific section"**
→ [Finding Sections](../cookbook/finding-sections.md)

**"The PDF is scanned/image-based"**
→ [OCR Then Navigate](../cookbook/ocr-then-navigate.md)

### Combining Patterns

Most real extractions combine patterns:

1. **Invoice Processing**
   - [Label-Value Extraction](../cookbook/label-value-extraction.md) for header fields
   - [Messy Tables](../cookbook/messy-tables.md) for line items

2. **Batch Form Processing**
   - [One Page = One Row](../cookbook/one-page-one-row.md) for the loop
   - [Label-Value Extraction](../cookbook/label-value-extraction.md) for each field
   - [OCR Then Navigate](../cookbook/ocr-then-navigate.md) if scanned

3. **Report Analysis**
   - [Finding Sections](../cookbook/finding-sections.md) to locate content
   - [Multipage Content](../cookbook/multipage-content.md) for long sections
   - [Messy Tables](../cookbook/messy-tables.md) for data tables within

## Not Finding Your Use Case?

These patterns cover most extraction needs. If your document doesn't fit:

1. **Check if it's a layout issue** - Try [layout analysis](../tutorials/07-layout-analysis.ipynb)
2. **Check if it needs AI** - Consider [document Q&A](../tutorials/06-document-qa.ipynb)
3. **Check the tutorials** - Browse the [getting started guide](../getting-started/quickstart.md)

## Contributing

Have a document type that worked well with Natural PDF? Consider contributing an example to help others with similar documents.
