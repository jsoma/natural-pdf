# Natural PDF

A Python library for PDF processing with a jQuery-like selector API, spatial navigation, OCR, and AI-powered extraction.

## Project Structure

```
natural_pdf/          # Library source
  core/               # PDF, Page, PageCollection — main entry points
  elements/           # Element, Region, ElementCollection
  services/           # Service delegation layer (navigation, OCR, extraction, etc.)
  selectors/          # CSS-like selector engine
  flows/              # Multi-page/multi-column content reflow (FlowRegion)
  ocr/                # OCR engine adapters (easyocr, surya, paddleocr, doctr)
  extraction/         # Structured data extraction (LLM, VLM, doc_qa)
  exporters/          # Searchable PDF, hOCR output
  tables/             # Table detection and extraction engines
docs/                 # mkdocs site source (tutorials, cookbooks, API reference)
tests/                # pytest suite
pdfs/                 # Test PDFs — use these, don't create new ones
scripts/              # Build/CI scripts (notebook executor, publish)
temp/                 # Temp file output (gitignored)
```

## Architecture

- **Service delegation**: Page/Region/Flow objects delegate to `self.services.<name>.method(self, ...)`. Services are registered via `@register_delegate` in `natural_pdf/services/`.
- **`.ask()` is sugar over `.extract()`**: Creates a one-field schema and calls `host.extract()`. Returns `StructuredDataResult`.
- **Extraction engines**: `doc_qa` (LayoutLM), `llm` (OpenAI-compatible), `vlm` (local HF VLM). Resolved in `ExtractionService._resolve_engine()`.

## How the Library Works

### Selectors
jQuery-like syntax for finding PDF elements:
```
'text'                          # All text elements
'text:bold'                     # Pseudo-class (colon)
'text:contains("Invoice")'     # Text content match
'text:ocr("Invoice")'          # OCR-tolerant match (handles garbled chars)
'text[size>12]'                 # Attribute filter (brackets)
'text:bold[size>=14]'           # Combined
'line:horizontal'               # Line orientation
'region[type=table]'            # Layout-detected regions
```

### Spatial Navigation
Elements support directional methods that return Regions:
- `.left()` / `.right()` default to `height='element'` (same row)
- `.above()` / `.below()` default to `width='full'` (full page width)
- All accept `multipage=True` to span pages (returns `FlowRegion` when crossing pages)
- Exclusions can be added via `pdf.add_exclusion(lambda page: ...)` — supports Region, ElementCollection, or lists

### OCR
Multiple engines: `rapidocr` (default), `easyocr`, `surya`, `paddle`, `paddlevl`, `doctr`. Applied via `page.apply_ocr(engine="rapidocr")`. GLM-OCR (0.9B VLM) works via `engine="vlm"` with `model="zai-org/GLM-OCR"` — runs layout detection + per-region OCR in-process.

### Extraction
- **Structured data**: `page.extract(MyPydanticModel)` or `page.extract(MyPydanticModel, engine="llm", client=...)`
- **Q&A shorthand**: `page.ask("What is the total?")` — wraps extract with a single-field schema
- **Classification**: `page.classify(["invoice", "receipt"], using="text")`

### Display
- `pdf.show()` limits to 30 pages by default; use `limit=N` or `limit=None`
- `page.show(exclusions='red')` visualizes exclusion zones

### Collections
- `pages.groupby('text[size=16]')` — group pages by selector text, pandas-style iteration
- `elements.apply(fn)` / `elements.filter(fn)` — transform and filter collections

## Method Return Types

| Method | Returns | Notes |
|--------|---------|-------|
| `PDF("path")` | `PDF` | Load from file, URL, or bytes |
| `pdf.pages` | `PageCollection` | Iterable, supports slicing |
| `page.find(selector)` | `Element \| None` | First match or None |
| `page.find_all(selector)` | `ElementCollection` | All matches (may be empty) |
| `element.below()` | `Region` | Spatial navigation |
| `region.extract_text()` | `str` | Text content |
| `page.extract_table()` | `TableResult` | Has `.to_df()` |
| `page.extract_tables()` | `List[TableResult]` | All tables on page |
| `page.apply_ocr()` | `ElementCollection` | OCR text elements |
| `page.analyze_layout()` | `ElementCollection` | Detected regions |
| `page.ask(question)` | `StructuredDataResult` | Has `.data`, `.success` |
| `page.extract(Schema)` | `StructuredDataResult` | Has `.data`, `.success` |
| `page.to_markdown()` | `str` | VLM-powered markdown conversion |
| `page.compare_ocr(engines=[...])` | `OcrComparison` | Has `.show()`, `.heatmap()`, `.coverage()`, `.diff()`, `.loupe()`, `.summary()`, `.apply()` |
| `pdf.search(query)` | `PageCollection` | Semantic search over pages |

## Common Mistakes

```python
# WRONG method names
page.get_text()           # Use: page.extract_text()
page.search("term")       # Use: page.find('text:contains("term")')
PDF.open("file.pdf")      # Use: PDF("file.pdf")

# WRONG selector syntax
page.find('text.bold')              # Use colon: 'text:bold'
page.find('text[contains="X"]')     # contains is a pseudo-class: 'text:contains("X")'
page.find('text(size>12)')          # Use brackets: 'text[size>12]'

# WRONG parameter names
page.find('text:contains("X")', case_sensitive=False)  # Use: case=False
page.apply_ocr(engine="easy_ocr")                      # Use: engine="easyocr"

# Must handle None from find()
element = page.find('text:contains("Missing")')
if element:
    text = element.extract_text()

# Must close PDFs in loops
for path in pdf_paths:
    pdf = PDF(path)
    try:
        ...
    finally:
        pdf.close()
```

## Development

### Environment
- Virtual environment: `.venv`
- Package manager: `uv`
- Run tests: `.venv/bin/pytest tests/ -x`
- CI sessions: `nox -s lint`, `nox -s test_minimal`, `nox -s test_full`

### Conventions
- Temp files go in `temp/`
- Test files go in `tests/`
- Use `pdfs/01-practice.pdf` for tests — don't create new PDFs
- Most changes need a test; prefer test-driven development
- Tutorials are markdown files in `docs/` — edit `.md`, not `.ipynb`

### Documentation Style
- **Say what to do, what happens, and what to try if it doesn't work.** No padding.
- **No empty adjectives.** Don't call things "robust", "seamless", "comprehensive", "intelligent", "friendly", or "lightweight" unless you're immediately explaining what that means concretely.
- **No unsubstantiated ratings.** Don't put "High/Medium/Low" in comparison tables unless there are benchmarks behind them. Use concrete notes instead (what it does, what it's for, install command).
- **No shrugging.** "Keep trying until one works!" is not guidance. Compare the options and say when to use each one.
- **No TODOs or Wish Lists in published docs.** Track those in issues, not in user-facing pages.
- **Don't claim things you haven't verified.** If confidence scores are LLM self-reports, say so. If you don't know whether engine A is faster than engine B, don't rank them.
- **Show the install command.** When a feature requires an optional dependency, state the `pip install` right there.
- **Prerequisites before code.** If `find_all('region[type=table]')` requires `analyze_layout()` first, show that call. Don't let copy-paste return empty results.
- **`.show()` returns a PIL Image.** When relevant, show `.save("output.png")` — users need to know they can save it.
