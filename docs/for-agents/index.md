---
fixture: pdfs/01-practice.pdf
tier: fast
---

# Natural PDF for coding agents

The contract surface for writing extraction code against natural-pdf: exact return types, what raises vs. what returns empty, the selector grammar, and the anti-patterns that produce plausible-but-wrong output. Every claim on this page is verified against the current codebase.

```python {.skip-execution}
from natural_pdf import PDF

pdf = PDF("document.pdf")        # path, URL, or bytes — there is no PDF.open()
page = pdf.pages[0]
```

## Return types

| Call | Returns | Notes |
|---|---|---|
| `PDF("path")` | `PDF` | Load from file path, URL, or bytes. Context manager: `with PDF(p) as pdf:` |
| `pdf.pages` | `PageCollection` | Iterable; slicing returns `PageCollection` |
| `pdf.pages[0]` | `Page` | |
| `page.find(selector)` | `Element \| None` | `None` on no match — never raises for a miss |
| `page.find_all(selector)` | `ElementCollection` | May be empty; `collection.first` is `Element \| None` |
| `element.above()` / `.below()` / `.left()` / `.right()` | `Region` | `multipage=True` may return `FlowRegion` |
| `page.region(left=, top=, right=, bottom=)` | `Region` | Also accepts `width=` / `height=` |
| `page.extract_text()` / `region.extract_text()` | `str` | Empty string when there is nothing to read |
| `collection.extract_text()` | `str` | The only correct way to join multiple elements' text |
| `page.extract_table()` | `TableResult` | `.to_df(header="first")` → `pandas.DataFrame` |
| `page.extract_tables()` | `list[TableResult]` | All tables on the page |
| `page.apply_ocr(...)` | `Page` (self) | Mutates the page, returns it for chaining. Select the new elements with `page.find_all('text[source=ocr]')` |
| `page.analyze_layout(...)` | `ElementCollection` | Detected regions; then `page.find_all('region[type=table]')` |
| `page.ask(question)` | `StructuredDataResult` | Fields: `.data`, `.success`, `.error_message`, `.model_used`, `.raw_output` |
| `page.extract(Schema)` | `StructuredDataResult` | Same fields; `Schema` is a Pydantic model class or a sequence of field names |
| `page.classify(labels, using="text")` | `ClassificationResult` | Mapping with `.category`, `.score`, `.scores`; also stored in `page.analyses` |
| `page.to_markdown()` | `str` | VLM-backed; falls back to `extract_text()` when no model is configured |
| `page.to_llm(detail=..., include_hints=..., max_chars=...)` | `str` | Text-only page representation for prompts; see below |
| `page.describe()` | `ElementSummary` | Renders as a markdown census of the page |
| `page.compare_ocr(engines=[...])` | `OcrComparison` | Methods: `.show()`, `.summary()`, `.heatmap()`, `.coverage()`, `.diff()`, `.loupe()`, `.apply(engine)` |
| `pdf.search(query, top_k=5)` | `PageCollection` | Semantic page ranking; requires the `ai` extra |
| `pages.groupby(by)` | `PageGroupBy` | `by` is a selector string or callable |
| `page.add_exclusion(...)` / `pdf.add_exclusion(...)` | the host (`Page` / `PDF`) | Chainable |
| `page.show()` / `element.show()` / `collection.show()` | `PIL.Image.Image` | Save with `.save("out.png")` |
| `pdf.close()` | `None` | Partial: already-materialized state may remain readable; page loading / OCR / rendering become unavailable |

Executed spot check:

```python
from natural_pdf import PDF

pdf = PDF("pdfs/01-practice.pdf")
page = pdf.pages[0]

checks = [
    ("pdf.pages", pdf.pages),
    ("page.find('text')", page.find('text')),
    ("page.find('text:contains(\"zzz\")')", page.find('text:contains("zzz")')),
    ("page.find_all('text')", page.find_all('text')),
    ("page.find('text').below()", page.find('text').below()),
    ("page.extract_text()", page.extract_text()),
    ("page.extract_table()", page.extract_table()),
    ("page.extract_table().to_df()", page.extract_table().to_df()),
    ("page.describe()", page.describe()),
    ("page.show()", page.show()),
]
for expr, value in checks:
    print(f"{expr:42} -> {type(value).__name__}")
```

## Error contract: None vs. empty vs. raised

**Returns `None`, never raises:** `page.find(...)` on no match; `collection.first` on an empty collection.

**Returns empty, never raises:** `page.find_all(...)` → empty `ElementCollection`; `extract_text()` → `""` on a page with no (non-excluded) text; `extract_table()` on a page with no detectable table.

**Reported on the result, not raised:** `extract()` / `ask()` engine-run failures come back as `StructuredDataResult` with `success=False` and the reason in `error_message` (`natural_pdf/services/extraction_service.py`). Always check `result.success` before using `result.data`. Argument errors (below) still raise before any engine runs.

**Raises** — all library exceptions derive from `natural_pdf.exceptions.NaturalPDFError` except the built-in `TypeError` / `ValueError` argument checks (full hierarchy: [Exception Reference](../reference/exceptions.md)):

| Exception | Raised when | Source |
|---|---|---|
| `TypeError` | A directional method gets a number in the cross-direction slot: `below(width=200)`, `right(height=50)`. Cross-direction takes `'full'` or `'element'`; the travel direction takes the number (`below(height=200)`). | `natural_pdf/elements/_directional.py` |
| `ValueError` | Unknown `deskew_kwargs` keys in `deskew()` / `detect_skew_angle()` | `natural_pdf/deskew/deskew_provider.py` |
| `ValueError` | `to_llm()` with `detail` not in `('brief', 'standard', 'full')` or `include_hints` not in `('none', 'descriptive', 'api')` | `natural_pdf/describe/to_llm.py` |
| `ValueError` | `region.find_all(overlap=...)` not in `('full', 'partial', 'center')` | `natural_pdf/services/selector_service.py` |
| `ValueError` / `TypeError` | `extract()` argument validation: unknown `engine`, unknown `using`, `engine="llm"` without `client=`, schema that is neither a Pydantic model class nor a sequence of field names | `natural_pdf/services/extraction_service.py` |
| `TextExtractionError` | `extract_text(layout=...)` with an invalid layout bbox (non-positive width/height), malformed character data, or a failed layout reconstruction | `natural_pdf/text/pipeline.py` |
| `OCRError` | An OCR provider returns a malformed payload (bad `image_size`, non-mapping results, wrong field types) | `natural_pdf/core/ocr_converter.py` |
| `ClassificationError` | Classification engine/payload failures, including batch classification | `natural_pdf/classification/pipelines.py`, `natural_pdf/core/pdf.py` |
| `OCREngineNotAvailableError` / `LayoutEngineNotAvailableError` | Requested engine's dependency is not installed; message includes the `pip install` hint | `natural_pdf/exceptions.py` |
| `ImportError` ("installed but failed to import") | The optional dependency exists but its import fails (environment conflict). Distinct message from "not installed" — do not respond by reinstalling the same package | `natural_pdf/utils/optional_imports.py` |

## Anti-patterns

### Never hand-join element `.text`

Element `.text` carries no spacing information about neighbors; joining fuses words. Any time two or more elements become one string, go through `.extract_text()`:

```python
row = page.find_all('text')[2:5]
print("".join(el.text for el in row))   # WRONG: words fuse
print(row.extract_text())               # RIGHT: assembled with spacing and reading order
```

### Never invent methods

These do not exist. Verified corrections:

| Does not exist | Use instead |
|---|---|
| `PDF.open("f.pdf")` | `PDF("f.pdf")` |
| `page.get_text()` | `page.extract_text()` |
| `page.search("term")` | `page.find('text:contains("term")')` — `search()` exists only on `PDF`, where it is semantic page ranking |
| `page.get_tables()` | `page.extract_tables()` |
| `page.apply_layout()` | `page.analyze_layout()` |
| `page.query(...)` / `page.select(...)` | `page.find(...)` / `page.find_all(...)` |

Parameter names that do not exist: `case_sensitive=False` → `case=False`; `engine="paddle_ocr"` → `engine="paddle"`.

### Always handle `find()` returning `None`

```python {.skip-execution}
element = page.find('text:contains("Total")')
if element is None:
    ...                      # handle the miss — chaining onto None raises AttributeError
value = element.right().extract_text()
```

### Close PDFs in batch loops

```python {.skip-execution}
for path in paths:
    with PDF(path) as pdf:          # PDF is a context manager
        rows.append(pdf.pages[0].extract_text())   # keep plain data, not Page/Element objects
```

### Region `find_all()` drops edge-crossing elements — pass `overlap="partial"`

The default `overlap="full"` keeps only elements *entirely inside* the region. An element that pokes past the region edge silently disappears:

```python
anchor = page.find('text:contains("Durham")')
clip = page.region(left=0, top=anchor.top - 2,
                   right=anchor.x0 + anchor.width / 2, bottom=anchor.bottom + 2)

print([e.text for e in clip.find_all('text')])                     # edge-crosser dropped
print([e.text for e in clip.find_all('text', overlap='partial')])  # included
```

`overlap="center"` (element's center inside the region) is the middle ground and the default for table cell assignment.

### Do not replace anchors with generic y-position grouping

When rows have a stable anchor (line numbers, case IDs, first-column record IDs), extract rows from the anchors (`extract_table_guided`, `extract_anchored_rows`) — the anchor is the evidence that decides which rows belong. Grouping all text by y-coordinate reintroduces every ambiguity the anchor solved.

### Prerequisites before selectors

`page.find_all('region[type=table]')` returns nothing until `page.analyze_layout()` has run. `text[source=ocr]` matches nothing until `page.apply_ocr()` has run.

## Selector grammar

Generated source of truth — every registered pseudo-class, attribute operator, and alias: [Selector Reference](../reference/selectors.md). The shape rules:

```
selector      = [type] clause*  ( ("|" | ",") selector )*     # OR of AND-chains
clause        = "[" name op value "]"                          # attribute comparison
              | ":" name [ "(" argument ")" ]                  # pseudo-class
type          = text | line | rect | image | curve | region | checkbox | * | <any region_type>
op            = (bare) | = | != | < | <= | > | >= | ~= | *= | ^= | $=
```

- **Colon vs. bracket is not interchangeable.** Content/style/position tests are pseudo-classes (`:contains("X")`, `:bold`, `:first`); property comparisons are bracket attributes (`[size>12]`, `[fontname*="Bold"]`). `text.bold`, `text[contains="X"]`, and `text(size>12)` are all invalid.
- **Quoting:** pseudo-class string arguments and string attribute values take single or double quotes: `text:contains("Invoice")`.
- **AND is implicit** (concatenate clauses, no spaces: `text:bold[size>=14]`); **OR** is `|` or `,` between whole selectors. OR selectors cannot contain relational pseudo-classes (`:above(...)`, `:below(...)`, `:near(...)`).
- **No combinators.** No descendant/child syntax (`div p`, `>`) — PDFs have no tree. Use relational pseudo-classes or navigation methods.
- `max()` / `min()` work as attribute values: `text[size=max()]`.
- Attribute names accept hyphens or underscores; type names are case-insensitive with spaces/underscores/hyphens normalized.

## Privacy contract for VLM/LLM calls

Whether a call sends page images off-machine follows one rule (implemented identically in OCR dispatch and extraction):

- **Explicit `model=` ⇒ local.** Naming a model runs it in-process, and the default client is suppressed for the whole call — nothing nested can backfill a remote endpoint. Shorthand engines (`glm_ocr`, `dots`, `chandra`, `paddlevl`) resolve to pinned local models and get the same guarantee.
- **Explicit `client=` ⇒ remote,** to exactly that client.
- **The default client (`natural_pdf.set_default_client(...)`) applies only when *both* `model=` and `client=` are unset.**

```python {.skip-execution}
import natural_pdf
from openai import OpenAI

natural_pdf.set_default_client(OpenAI(), model="gpt-5-mini")

page.apply_ocr(engine="vlm")                           # remote: default client fills the gap
page.apply_ocr(engine="vlm", model="zai-org/GLM-OCR")  # LOCAL: explicit model wins
page.apply_ocr(engine="glm_ocr")                       # LOCAL: shorthand pins a local model
page.extract(Schema, engine="vlm", client=my_client)   # remote, to that client
```

Audit question "did this document leave the machine?" — only if you passed `client=`, or called a VLM feature with no model and no client while a default client was set. Everything else (every named model, every classic OCR/layout/QA/classification engine) is local inference. Details: [Engines and Models](../concepts/engines-and-models.md).

## Page representation for prompts: `to_llm()`

`page.to_llm()` returns a text-only structural digest of a page (dimensions, text-layer diagnostics, layout preview with element boundaries, style tiers, alignment, lines, rectangles) built for pasting into an LLM context. No model runs; no image is produced.

Current surface:

- `detail=` — `"brief"` | `"standard"` (default) | `"full"`: controls layout-preview lines, style-tier caps, and sample counts.
- `include_hints=` — `"none"` (default) | `"descriptive"` | `"api"`: appends next-step guidance, either descriptive or as natural-pdf API calls.
- `max_chars=` — cap on final output length (default `6000`; `None` disables).

Invalid mode strings raise `ValueError` instead of silently degrading. With the `quality` extra installed (`pip install "natural-pdf[quality]"`), the text-layer section includes a dictionary-based garble rate for detecting corrupt/OCR text layers.

```python
print(page.to_llm(detail="brief"))
```
