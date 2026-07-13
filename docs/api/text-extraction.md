# Text extraction contracts

Natural PDF has four `extract_text()` families. Each family defines its signature
and help text once, so IDE completion is consistent without advertising options a
host cannot use.

| Family | Hosts | Additional options |
| --- | --- | --- |
| Spatial | `Page`, `Region`, `RectangleElement` | layout, exclusions, bidi |
| Scalar | `TextElement` | newline and whitespace transforms |
| Ordered aggregate | `PDF`, `PageCollection`, `Flow`, `FlowRegion`, `FlowRegionCollection` | spatial options plus `separator` |
| Selected aggregate | `ElementCollection`, `FlowElementCollection` | literal selection join plus `separator` |

Line, image, and generic non-text elements intentionally expose only
`extract_text() -> ""`. `FlowElement.extract_text()` is also a no-argument proxy;
use `flow_element.physical_object` when you deliberately need its host-specific
options.

## Spatial extraction

```python
text = page.extract_text(
    layout=False,
    apply_exclusions=True,
    newlines=True,
    whitespace="preserve",
    strip=True,
    bidi=True,
    content_filter=None,
)
```

The same options and defaults mean the same thing on a `Page`, `Region`, or
rectangle-backed spatial host.

- `newlines=True` preserves canonical `\n` line breaks.
- `newlines=False` replaces each line break with one space.
- A string such as `newlines=" | "` is used as the exact replacement.
- `whitespace="normalize"` collapses horizontal whitespace within each line. It
  never removes line boundaries.
- `strip` trims only the overall result boundaries.
- `content_filter` accepts a regex, compiled regex, sequence of regexes, or a
  per-character predicate. Regex matches are removed; predicates return `True`
  for characters to retain. Invalid filters fail closed.

Advanced layout settings live in one typed value instead of being duplicated
across every method signature:

```python
from natural_pdf import TextLayoutOptions

text = page.extract_text(
    layout=TextLayoutOptions(
        x_density=8,
        y_density=14,
        x_tolerance=4,
    )
)
```

`layout=False` is the default on spatial and ordered aggregate hosts. An ordered aggregate lays out each
page or region independently; it never sends characters from multiple pages
through one page-shaped layout operation.

## Aggregates and selections

PDFs and page collections use `"\n"` as their natural boundary. Flow-family
objects use `"\n\n"`. Passing `separator=None` selects that natural boundary;
passing `""` deliberately joins without one.

```python
document_text = pdf.extract_text()
custom_text = pdf.extract_text(separator="\n\n---\n\n")
```

Ordered aggregates preserve empty members, so an empty middle page still has
both page boundaries. They transform and filter each leaf before joining;
separators are exact and regexes never match across source boundaries. Therefore:

```python
pdf.extract_text() == "\n".join(page.extract_text() for page in pdf.pages)
pdf.pages.extract_text() == pdf.extract_text()
```

Selected collections preserve the caller's stored order and duplicates. They do
not perform layout reconstruction, sorting, or deduplication, and omit non-text or
empty contributions:

```python
words = region.find_all("text", overlap="partial")
text = words.extract_text(separator=" ")
```

A `PDFCollection` deliberately does not flatten document boundaries and has no
`extract_text()` or `extract_text_result()` method. Use
`pdfs.extract_each_text()` to get one string per document, then choose a
document-level representation yourself:

```python
texts = pdfs.extract_each_text(
    layout=False,
    apply_exclusions=True,
    newlines=True,
    whitespace="preserve",
    strip=True,
    bidi=True,
)
```

## Provenance

`extract_text()` always returns `str`. Use `extract_text_result()` when source
mapping is required:

```python
result = pdf.extract_text_result(layout=True)
print(result.text)
for segment in result.segments:
    print(segment.output_start, segment.output_end, segment.source)
```

The result API accepts acquisition options only (`layout`, exclusions, and an
aggregate separator). Text transforms are intentionally absent until their
character mappings can be updated without ambiguity. Segment offsets include the
exact aggregate separator, while separator ranges themselves have no source.

Structured extraction follows the same rule when `citations=True`: do not pass
`newlines`, `whitespace`, `strip`, `bidi`, or `content_filter`. Those transforms
change character offsets, so Natural PDF rejects them before acquiring or sending
text rather than returning misaligned citations. They remain available when
citations are disabled.

Third-party hosts that implement only `extract_text() -> str` remain usable for
ordinary structured extraction. If citations are requested, Natural PDF warns and
continues without element citations. Implement `extract_text_result()` and return
an `ExtractedText` value to provide provenance.

## Breaking-contract migration

The cleanup intentionally removes aliases and data-dependent options:

| Removed | Replacement |
| --- | --- |
| `use_exclusions=` | `apply_exclusions=` |
| `preserve_line_breaks=False` | `newlines=False` |
| `preserve_whitespace=` | `whitespace="preserve"` or `"normalize"` |
| `page_separator=` | `separator=` |
| top-level layout tuning arguments | `layout=TextLayoutOptions(...)` |
| `extract_text(return_textmap=True)` | `extract_text_result()` |
| `region.extract_text("words", overlap=...)` | `region.find_all("text", overlap=...).extract_text()` |
| `pdf.extract_text(selector=...)` | `pdf.find_all(selector).extract_text()` |
| `strip_final` / `strip_empty` | `strip`; handle line removal explicitly |
| `extract_text(debug=...)` / `extract_text(debug_exclusions=...)` | normal logging; `debug_exclusions` remains available on exclusion-inspection methods such as `Page.get_elements()` |

Unknown or wrong-family arguments now raise `TypeError` immediately, including on
empty hosts. Layout failures raise `TextExtractionError` and retain the backend
exception as their cause; extraction never silently switches to a different text
representation.
