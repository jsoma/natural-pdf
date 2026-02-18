# Batch Processing PDFs

When working with FOIA responses, court records, or large document dumps, you often need to process hundreds or thousands of PDFs efficiently. This guide covers practical patterns for batch processing that journalists and data analysts use every day.

## Processing Multiple PDFs in a Directory

The simplest approach is to iterate over a directory of PDFs:

```python
from pathlib import Path
from natural_pdf import PDF

results = []

for pdf_path in Path("documents/").glob("*.pdf"):
    pdf = PDF(str(pdf_path))

    # Extract what you need
    text = pdf.pages[0].extract_text()
    results.append({
        "file": pdf_path.name,
        "text": text
    })

    # Always close when done to free memory
    pdf.close()

# Now results contains data from all PDFs
```

For recursive searches (subdirectories too):

```python
# Use ** to match any subdirectory depth
for pdf_path in Path("documents/").glob("**/*.pdf"):
    pdf = PDF(str(pdf_path))
    # ... process ...
    pdf.close()
```

## Using PDFCollection

For more sophisticated batch work, `PDFCollection` handles loading, iteration, and common operations:

```python
from natural_pdf import PDFCollection

# Load from a directory
collection = PDFCollection("documents/foia_response/")

# Or from a list of specific files
collection = PDFCollection([
    "case_001.pdf",
    "case_002.pdf",
    "case_003.pdf"
])

# Or use a glob pattern
collection = PDFCollection("records/*.pdf")

# Or mix URLs and local files
collection = PDFCollection([
    "local_file.pdf",
    "https://example.com/remote.pdf"
])
```

### Iterating Over PDFs

```python
from natural_pdf import PDFCollection

collection = PDFCollection("court_records/")

for pdf in collection:
    print(f"Processing: {pdf.path}")

    for page in pdf.pages:
        # Extract tables from each page
        tables = page.find_all('region[type=table]')
        for table in tables:
            data = table.extract_table()
            # ... save or process table data ...

    pdf.close()
```

### Using find_all Across Collections

Search across all PDFs at once:

```python
collection = PDFCollection("invoices/")

# Find all mentions of a specific company across all PDFs
mentions = collection.find_all('text:contains("ACME Corp")')
print(f"Found {len(mentions)} mentions across {len(collection)} PDFs")

# Extract text from each match
for element in mentions:
    print(f"Page {element.page.number}: {element.extract_text()}")
```

## Handling Missing Elements (Find or Skip)

When extracting specific data, not every PDF will have the element you're looking for. Use the "find or skip" pattern to handle this gracefully:

```python
from pathlib import Path
from natural_pdf import PDF

results = []

for pdf_path in Path("invoices/").glob("*.pdf"):
    pdf = PDF(str(pdf_path))
    page = pdf.pages[0]

    # Try to find the element
    total_element = page.find('text:contains("Total:")')

    if total_element:
        # Element found - extract the value
        value = total_element.right(width=100).extract_text().strip()
        results.append({
            "file": pdf_path.name,
            "total": value
        })
    else:
        # Element not found - skip or log
        print(f"Skipping {pdf_path.name}: No 'Total:' found")

    pdf.close()
```

For extracting multiple fields, check each one:

```python
for pdf_path in Path("forms/").glob("*.pdf"):
    pdf = PDF(str(pdf_path))
    page = pdf.pages[0]

    record = {"file": pdf_path.name}

    # Extract each field if present
    for field_name in ["Name:", "Date:", "Amount:"]:
        label = page.find(f'text:contains("{field_name}")')
        if label:
            record[field_name.rstrip(":")] = label.right(width=200).extract_text().strip()
        else:
            record[field_name.rstrip(":")] = None  # or "" or "N/A"

    results.append(record)
    pdf.close()
```

This pattern prevents crashes when documents have inconsistent layouts.

## Error Handling and Resumption

Real-world document dumps often contain corrupted files or unexpected formats. Build resilience into your processing:

```python
from pathlib import Path
from natural_pdf import PDF
import json

# Track progress for resumption
progress_file = Path("processing_progress.json")

def load_progress():
    if progress_file.exists():
        return json.loads(progress_file.read_text())
    return {"completed": [], "failed": [], "results": []}

def save_progress(progress):
    progress_file.write_text(json.dumps(progress, indent=2))

# Main processing loop
progress = load_progress()
completed_set = set(progress["completed"])

pdf_files = list(Path("documents/").glob("*.pdf"))

for pdf_path in pdf_files:
    filename = str(pdf_path)

    # Skip already processed files
    if filename in completed_set:
        print(f"Skipping (already done): {pdf_path.name}")
        continue

    try:
        pdf = PDF(filename)

        # Your extraction logic here
        result = {
            "file": pdf_path.name,
            "pages": len(pdf.pages),
            "text": pdf.pages[0].extract_text()[:500]  # First 500 chars
        }

        progress["results"].append(result)
        progress["completed"].append(filename)

        pdf.close()

    except Exception as e:
        print(f"Error processing {pdf_path.name}: {e}")
        progress["failed"].append({
            "file": filename,
            "error": str(e)
        })

    # Save progress after each file (allows resumption)
    save_progress(progress)

print(f"Completed: {len(progress['completed'])}")
print(f"Failed: {len(progress['failed'])}")
```

### Writing Results Incrementally

For large jobs, write results as you go rather than storing everything in memory:

```python
from pathlib import Path
from natural_pdf import PDF
import csv

output_file = Path("extracted_data.csv")

# Open in append mode so we can resume
with open(output_file, "a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["file", "page", "text"])

    # Write header only if file is empty
    if output_file.stat().st_size == 0:
        writer.writeheader()

    for pdf_path in Path("documents/").glob("*.pdf"):
        try:
            pdf = PDF(str(pdf_path))

            for page in pdf.pages:
                writer.writerow({
                    "file": pdf_path.name,
                    "page": page.number,
                    "text": page.extract_text()[:1000]
                })

            # Flush to disk after each PDF
            f.flush()
            pdf.close()

        except Exception as e:
            print(f"Error: {pdf_path.name} - {e}")
```

## Progress Tracking with tqdm

For long-running jobs, progress bars help you estimate completion time:

```python
from pathlib import Path
from natural_pdf import PDF
from tqdm import tqdm

pdf_files = list(Path("documents/").glob("*.pdf"))

# Basic progress bar
for pdf_path in tqdm(pdf_files, desc="Processing PDFs"):
    pdf = PDF(str(pdf_path))
    # ... process ...
    pdf.close()

# Nested progress bars for pages within PDFs
for pdf_path in tqdm(pdf_files, desc="PDFs", position=0):
    pdf = PDF(str(pdf_path))

    for page in tqdm(pdf.pages, desc="Pages", position=1, leave=False):
        # ... process each page ...
        pass

    pdf.close()
```

### Progress with PDFCollection

PDFCollection has built-in progress bar support for OCR:

```python
from natural_pdf import PDFCollection

collection = PDFCollection("scanned_docs/")

# OCR all PDFs with progress bar
collection.apply_ocr(engine='easyocr', languages=['en'])
```

## Memory Management

PDFs can consume significant memory, especially when processing images or running OCR. Here are strategies to keep memory usage under control:

### Close PDFs After Processing

```python
from natural_pdf import PDF

# Always close when done
pdf = PDF("large_document.pdf")
# ... process ...
pdf.close()

# Or use a context manager pattern
from contextlib import contextmanager

@contextmanager
def open_pdf(path):
    pdf = PDF(path)
    try:
        yield pdf
    finally:
        pdf.close()

# Usage
with open_pdf("document.pdf") as pdf:
    text = pdf.pages[0].extract_text()
# PDF is automatically closed here
```

### Process in Batches

For very large document sets, process in batches to limit concurrent memory usage:

```python
from pathlib import Path
from natural_pdf import PDF

def batch_process(pdf_paths, batch_size=10):
    """Process PDFs in batches to manage memory."""
    results = []

    for i in range(0, len(pdf_paths), batch_size):
        batch = pdf_paths[i:i + batch_size]
        print(f"Processing batch {i//batch_size + 1} ({len(batch)} files)")

        for pdf_path in batch:
            pdf = PDF(str(pdf_path))

            # Extract data
            results.append({
                "file": pdf_path.name,
                "text": pdf.extract_text()
            })

            # Important: close immediately after processing
            pdf.close()

        # Optional: force garbage collection between batches
        import gc
        gc.collect()

    return results

# Usage
pdf_files = list(Path("massive_dump/").glob("*.pdf"))
all_results = batch_process(pdf_files, batch_size=20)
```

### Limit Pages Processed

When you only need specific pages:

```python
from natural_pdf import PDF

pdf = PDF("huge_report.pdf")

# Only process first 5 pages
for page in pdf.pages[:5]:
    text = page.extract_text()
    # ...

pdf.close()
```

## Practical Example: FOIA Document Processing

Here is a complete example for processing a FOIA response where you need to:
1. Extract text from all documents
2. Find specific keywords
3. Export results to CSV

```python
from pathlib import Path
from natural_pdf import PDF
from tqdm import tqdm
import csv
import re

# Configuration
input_dir = Path("foia_response/")
output_file = Path("foia_analysis.csv")
keywords = ["violation", "penalty", "fine", "warning"]

# Compile regex for efficiency
keyword_pattern = re.compile("|".join(keywords), re.IGNORECASE)

# Get all PDFs
pdf_files = sorted(input_dir.glob("**/*.pdf"))
print(f"Found {len(pdf_files)} PDFs to process")

# Process and write results
with open(output_file, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "file", "page", "keyword_count", "keywords_found", "excerpt"
    ])
    writer.writeheader()

    for pdf_path in tqdm(pdf_files, desc="Analyzing documents"):
        try:
            pdf = PDF(str(pdf_path))

            for page in pdf.pages:
                text = page.extract_text()

                # Find all keyword matches
                matches = keyword_pattern.findall(text)

                if matches:
                    # Get context around first match
                    first_match = keyword_pattern.search(text)
                    start = max(0, first_match.start() - 100)
                    end = min(len(text), first_match.end() + 100)
                    excerpt = text[start:end].replace("\n", " ")

                    writer.writerow({
                        "file": pdf_path.name,
                        "page": page.number,
                        "keyword_count": len(matches),
                        "keywords_found": ", ".join(set(m.lower() for m in matches)),
                        "excerpt": excerpt
                    })

            pdf.close()

        except Exception as e:
            print(f"\nError processing {pdf_path.name}: {e}")

print(f"\nResults saved to {output_file}")
```

## Parallel Processing with OCR

For OCR-heavy workloads, PDFCollection supports parallel processing:

```python
from natural_pdf import PDFCollection

collection = PDFCollection("scanned_documents/")

# Process 4 PDFs concurrently
collection.apply_ocr(
    engine='easyocr',
    languages=['en'],
    max_workers=4
)

# Now extract text from OCR results
for pdf in collection:
    for page in pdf.pages:
        text = page.extract_text()
        # ... process text ...
    pdf.close()
```

## Classifying Documents

Use `.classify()` to automatically categorize PDFs based on content:

```python
from natural_pdf import PDF

pdf = PDF("unknown_document.pdf")

# Classify the entire PDF using text content
pdf.classify(
    ['invoice', 'contract', 'receipt', 'report'],
    using='text'
)

print(f"Category: {pdf.category}")
print(f"Confidence: {pdf.category_confidence:.2f}")
```

## Classifying Individual Pages

For multi-page documents with mixed content, classify each page:

```python
from natural_pdf import PDF

pdf = PDF("mixed_document.pdf")

# Classify pages using vision (looks at the image)
pdf.classify_pages(
    ['diagram', 'text', 'table', 'blank'],
    using='vision'
)

# See what each page was classified as
for page in pdf.pages:
    print(f"Page {page.number}: {page.category} ({page.category_confidence:.3f})")
```

## Filtering Pages by Category

After classification, filter to get specific page types:

```python
# Get only diagram pages
diagram_pages = pdf.pages.filter(lambda page: page.category == 'diagram')
diagram_pages.show(show_category=True)

# Get text-heavy pages
text_pages = pdf.pages.filter(lambda page: page.category == 'text')
print(f"Found {len(text_pages)} text pages")
```

## Grouping Pages

Use `groupby()` to organize pages by category or other criteria:

```python
from natural_pdf import PDF

pdf = PDF("large_report.pdf")
pdf.classify_pages(['diagram', 'text', 'table', 'blank'], using='vision')

# Group by category
groups = pdf.pages.groupby(lambda page: page.category)

# See what groups exist
groups.info()

# Access a specific group
diagrams = groups.get('diagram')
if diagrams:
    diagrams.show()
```

## Saving Filtered Pages as New PDF

Extract specific pages to a new PDF file:

```python
# Save only diagram pages to a new PDF
pdf.pages.filter(
    lambda page: page.category == 'diagram'
).save_pdf("diagrams_only.pdf", original=True)

# Save text pages
pdf.pages.filter(
    lambda page: page.category == 'text'
).save_pdf("text_pages.pdf", original=True)
```

## Tips for Journalists and Data Analysts

1. **Start small**: Test your pipeline on 5-10 documents before running on thousands.

2. **Log everything**: Keep a log of what was processed, what failed, and why.

3. **Use checksums**: If documents might be duplicated, hash them to avoid reprocessing.

4. **Validate results**: Spot-check extraction quality on a random sample.

5. **Plan for resumption**: Long jobs will inevitably be interrupted. Save progress frequently.

6. **Mind the rate limits**: If fetching PDFs from URLs, add delays to avoid overwhelming servers.

## Related Patterns

Combine batch processing with these extraction patterns:

- [One Page = One Row](one-page-one-row.md) - Extract one record per page from batches of forms
- [Label-Value Extraction](label-value-extraction.md) - Extract specific fields from each document
- [OCR Then Navigate](ocr-then-navigate.md) - Batch process scanned documents

```python
import time
import hashlib

def get_file_hash(path):
    """Get MD5 hash of file for deduplication."""
    return hashlib.md5(Path(path).read_bytes()).hexdigest()

# Skip duplicates
seen_hashes = set()
for pdf_path in Path("documents/").glob("*.pdf"):
    file_hash = get_file_hash(pdf_path)
    if file_hash in seen_hashes:
        print(f"Skipping duplicate: {pdf_path.name}")
        continue
    seen_hashes.add(file_hash)

    # Process unique file...
```
