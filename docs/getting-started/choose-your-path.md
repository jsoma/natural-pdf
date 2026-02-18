# Choose Your Path

Different backgrounds and goals need different starting points. Find your path below.

---

## By Your Goal

### "I need to extract data from PDFs quickly"

**Best path:** Start with the Quickstart, then go straight to a Cookbook recipe.

1. [Quickstart](quickstart.md) (5 min)
2. [Idea Gallery](../use-cases/idea-gallery.md) - Find your document type
3. Jump to the matching Cookbook recipe

**You'll learn:** Basic extraction, how to find the right pattern for your document.

---

### "I need to extract tables"

**Best path:** Focus on table extraction techniques.

1. [Quickstart](quickstart.md) (5 min)
2. [Tables Tutorial](../tutorials/04-table-extraction.ipynb)
3. [Messy Tables](../cookbook/messy-tables.md) - When default extraction fails

**You'll learn:** Table detection, extraction, handling complex tables.

---

### "I need to process scanned/image PDFs"

**Best path:** Learn OCR integration.

1. [Quickstart](quickstart.md) (5 min)
2. [OCR Tutorial](../tutorials/12-ocr-integration.ipynb)
3. [OCR Then Navigate](../cookbook/ocr-then-navigate.md)

**You'll learn:** Applying OCR, choosing engines, extracting from scanned documents.

---

### "I need to build a repeatable pipeline"

**Best path:** Learn the patterns, then batch processing.

1. [Quickstart](quickstart.md) (5 min)
2. [Core Concepts](concepts.md)
3. [Batch Processing](../cookbook/batch-processing.md)
4. [Idea Gallery](../use-cases/idea-gallery.md) - Find your document pattern

**You'll learn:** Robust patterns, error handling, processing multiple files.

---

### "I need to extract specific sections"

**Best path:** Master spatial navigation.

1. [Quickstart](quickstart.md) (5 min)
2. [Spatial Navigation Tutorial](../tutorials/08-spatial-navigation.ipynb)
3. [Finding Sections](../cookbook/finding-sections.md)

**You'll learn:** `.below()`, `.above()`, extracting content between markers.

---

### "I need to use AI for extraction"

**Best path:** Learn the AI features.

1. [Quickstart](quickstart.md) (5 min)
2. [Document QA Tutorial](../tutorials/06-document-qa.ipynb)
3. [Layout Analysis](../tutorials/07-layout-analysis.ipynb)

**You'll learn:** Document Q&A, structured extraction, layout detection.

---

## By Your Background

### Python Beginner

You've done a Python tutorial but aren't comfortable with the language yet.

**Recommended path:**

1. **[Quickstart](quickstart.md)** - Copy-paste working code
2. **[Selectors 101](selectors.md)** - Understand the selector syntax
3. **[Idea Gallery](../use-cases/idea-gallery.md)** - Find a recipe for your document type
4. Pick **one Cookbook recipe** and follow it step by step

**Tips:**

- Don't try to understand everything at once
- Copy the examples, then modify small pieces
- Use `element.show()` to see what you're finding
- Always check if `find()` returned `None` before using the result

---

### Data Analyst (Pandas Expert)

You're comfortable with Python and pandas, but new to PDFs.

**Recommended path:**

1. **[Quickstart](quickstart.md)** - See the API style
2. **[Core Concepts](concepts.md)** - Understand the object model
3. **[Tables Tutorial](../tutorials/04-table-extraction.ipynb)** - Get data into DataFrames
4. **[Batch Processing](../cookbook/batch-processing.md)** - Build pipelines

**Tips:**

- Tables export directly to DataFrames with `.to_df()`
- ElementCollections work like pandas with `.filter()` and `.apply()`
- Use `layout=True` in `extract_text()` for readable output

---

### Software Developer

You build production systems and need to evaluate or integrate the library.

**Recommended path:**

1. **[Installation](../installation/index.md)** - See dependency options
2. **[Core Concepts](concepts.md)** - Understand the architecture
3. **[Patterns & Pitfalls](../for-llms/common-patterns.md)** - API reference
4. **[Batch Processing](../cookbook/batch-processing.md)** - Error handling patterns
5. **[Troubleshooting](../cookbook/troubleshooting.md)** - Common issues

**Tips:**

- Install only what you need: `pip install natural-pdf` for core, add `[ai]` etc. as needed
- Use context managers or `pdf.close()` in loops
- OCR and layout analysis are the slow operations - profile before optimizing

---

### Researcher / Data Scientist

You work in Jupyter and need to extract structured data from documents.

**Recommended path:**

1. **[Quickstart](quickstart.md)** - See the basics
2. **[Finding Elements Tutorial](../tutorials/02-finding-elements.ipynb)** - Master selectors
3. **[Layout Analysis](../tutorials/07-layout-analysis.ipynb)** - Auto-detect structure
4. **[Document QA](../tutorials/06-document-qa.ipynb)** - AI extraction
5. **[Regions & Flows](../tutorials/15-working-with-regions.ipynb)** - Complex extractions

**Tips:**

- Use `.show()` constantly to visualize what you're finding
- `analyze_layout()` can detect tables, figures, and sections automatically
- For academic papers, start with [Finding Sections](../cookbook/finding-sections.md)

---

## Quick Start by Document Type

| Document Type | Start Here | Then Read |
|---------------|------------|-----------|
| **Invoices** | [Label-Value Extraction](../cookbook/label-value-extraction.md) | [Messy Tables](../cookbook/messy-tables.md) |
| **Forms** | [Label-Value Extraction](../cookbook/label-value-extraction.md) | [OCR Then Navigate](../cookbook/ocr-then-navigate.md) |
| **Reports** | [Finding Sections](../cookbook/finding-sections.md) | [Multipage Content](../cookbook/multipage-content.md) |
| **Scanned docs** | [OCR Tutorial](../tutorials/12-ocr-integration.ipynb) | [OCR Then Navigate](../cookbook/ocr-then-navigate.md) |
| **Tables** | [Tables Tutorial](../tutorials/04-table-extraction.ipynb) | [Messy Tables](../cookbook/messy-tables.md) |
| **FOIA responses** | [Finding Sections](../cookbook/finding-sections.md) | [Batch Processing](../cookbook/batch-processing.md) |
| **Contracts** | [Finding Sections](../cookbook/finding-sections.md) | [Multipage Content](../cookbook/multipage-content.md) |

See the [Idea Gallery](../use-cases/idea-gallery.md) for 30+ document types mapped to patterns.

---

## Still Not Sure?

If you're not sure where to start:

1. **Run the [Quickstart](quickstart.md)** - it's 5 minutes
2. **Load your own PDF** and try `page.extract_text()`
3. **Use `.show()`** to visualize what's in your document
4. **Browse the [Idea Gallery](../use-cases/idea-gallery.md)** for similar documents

The best way to learn is to experiment with your actual documents.
