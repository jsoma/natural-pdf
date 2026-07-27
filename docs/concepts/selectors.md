---
fixture: pdfs/01-practice.pdf
---

# Selectors

Natural PDF selectors look like CSS, but they aren't CSS — they're a small grammar tuned to PDF geometry. This page is the map: the exact shape of the grammar, every pseudo-class and operator the engine actually registers, and where the vocabulary comes from (and doesn't).

## The grammar

A selector is: an optional **element type**, then any number of **`[attribute]` blocks** and **`:pseudo-class`es**, in any order. Alternatives join with `|` or `,` (OR).

```
text:bold[size>=10]:contains("Total")
   │    │        │         │
   │    │        │         └─ pseudo-class with argument (quoted)
   │    │        └─ attribute comparison, in brackets
   │    └─ boolean pseudo-class, after a colon
   └─ element type
```

How it differs from CSS, concretely:

- **Colons and brackets are not interchangeable.** Text-content tests, style flags, and position filters are pseudo-classes (`:contains(...)`, `:bold`, `:first`); numeric and string comparisons on element properties are bracket attributes (`[size>12]`, `[fontname*="Bold"]`). `text.bold`, `text[contains="X"]`, and `text(size>12)` are all invalid.
- **No combinators.** There is no descendant/child syntax (`div p`, `>`); PDFs have no tree. Spatial relationships are pseudo-classes (`:below(...)`) or navigation methods.
- **Quoting:** pseudo-class arguments and string attribute values take single or double quotes. Chain clauses by concatenation, no spaces needed: `text:bold[size>=14]`.
- **OR exists, AND is implicit.** Everything you chain must all hold; `sel1|sel2` or `sel1,sel2` matches either. OR selectors can't contain the spatial pseudo-classes.
- A leading `*` means "any element type": `*[width>100]`.

```python
from natural_pdf import PDF

pdf = PDF("pdfs/01-practice.pdf")
page = pdf.pages[0]

len(page.find_all('text:contains("Critical")|text:contains("Serious")'))
```

## Element types

`text` matches word/char elements. `line`, `rect`, `image`, `curve` match graphic primitives. `region` matches layout-detected or manually added regions (anything with a `region_type`); `checkbox`, `table`, `table_cell`, `form_cell` and other named region types work as types too. Type names are matched case-insensitively with spaces/underscores/hyphens normalized, so `table-row`, `table_row`, and `table row` are equivalent.

## The registered pseudo-classes

This list is read from the actual registries (`natural_pdf/selectors/_clauses.py`, `natural_pdf/selectors/parser.py`). Three kinds exist, and the distinction matters because they run at different stages.

**Per-element filters** — test each element on its own:

| Pseudo-class | Matches when |
|---|---|
| `:contains("s")` | element text contains `s` (exact substring; see flags below) |
| `:regex("pat")` | element text matches the regex |
| `:startswith("s")` / `:starts-with("s")` | text starts with `s` |
| `:endswith("s")` / `:ends-with("s")` | text ends with `s` |
| `:bold`, `:italic`, `:not-bold`, `:not-italic` | font style flags |
| `:strike` (`:strikethrough`, `:strikeout`) | struck-through text |
| `:underline` (`:underlined`) | underlined text |
| `:highlight` (`:highlighted`) | highlighted text |
| `:horizontal`, `:vertical` | line orientation |
| `:checked`, `:unchecked` | checkbox state |
| `:empty`, `:not-empty` | whitespace-only vs. has text |
| `:first-child`, `:last-child` | position among a parent's children (regions with children) |
| `:not(selector)` | inner selector does *not* match |

**Fuzzy-match pseudo-classes** — filter *and* re-rank by score:

| Pseudo-class | What it does |
|---|---|
| `:ocr("s")` | OCR-confusion-tolerant substring match; superset of `:contains`. Threshold 0.75, tune inline: `:ocr("Total@0.6")`. Best score first. |
| `:closest("s")` | Jaro-Winkler similarity ranking; `:closest("Total@0.8")` sets a floor. Best match first. |

**Collection-level (post) pseudo-classes** — run *after* matching, on the result list:

| Pseudo-class | Effect |
|---|---|
| `:first` / `:last` | keep only the first / last match (in reading order) |
| `:nth(i)` | keep the i-th match, 0-indexed, negatives from the end |
| `:slice(start, stop)` | Python slice semantics; also `:slice(stop)` |
| `:limit(n)` | first n matches |

**Relational (spatial) pseudo-classes** — compare against a reference element found by another selector:

| Pseudo-class | Keeps elements that are |
|---|---|
| `:above(sel)` | entirely above the first match of `sel` |
| `:below(sel)` | entirely below it |
| `:left-of(sel)` / `:right-of(sel)` | entirely to its left / right |
| `:near(sel)` | within 50 pt center-to-center (override: `find_all(..., near_threshold=120)`) |

```python
page.find_all('text:bold:nth(1)')[0].text   # second bold element on the page
```

Search flags are keyword arguments on `find`/`find_all`, not selector syntax: `case=False` for case-insensitive `:contains`/`:closest`, `regex=True` to treat the `:contains` argument as a regex.

## Attribute operators

Inside `[...]`: a property name, an operator, a value. Bare `[name]` tests presence (value is not None).

| Operator | Meaning | Notes |
|---|---|---|
| `=` | equal | colors compared with Delta E ≤ 2; booleans accept `true/1/yes` |
| `!=` | not equal | |
| `>` `>=` `<` `<=` | numeric comparison | non-numeric values never match |
| `~=` | approximately equal | colors: Delta E ≤ 20. **Numeric literals are rejected** — `[size~=10]` raises with "use explicit ranges such as `[width>1][width<4]`" |
| `^=` / `$=` | string prefix / suffix | |
| `*=` | string contains | case-insensitive for `fontname`, case-sensitive otherwise |

Value handling worth knowing:

- Attribute names use hyphens or underscores (`[non-stroking-color=...]` → the `non_stroking_color` property).
- `x0`, `y0`, `x1`, `y1` read from the element's bbox. Any other name is looked up as a Python attribute on the element — the selector engine doesn't maintain a whitelist, so `[size>12]`, `[width>100]`, `[source=ocr]`, `[confidence>=0.8]` all work when elements carry those properties.
- Color attributes (`color`, `non_stroking_color`, `fill`, `stroke`, …) parse names, hex, and rgb tuples: `[color~=red]`, `[fill=#ff0000]`.
- **Aggregates:** the value can be computed from the whole candidate set — `min()`, `max()`, `avg()`, `median()`, `mode()`, `closest(...)` (colors), with optional arithmetic:

```python
page.find('text[size=max()]').text        # the largest text on the page
```

```python
[el.text for el in page.find_all('text[size>=max()-2]')][:3]
```

## The `region[type=...]` vocabulary problem

`region[type=table]` only matches regions that exist — run `page.analyze_layout()` (or add regions yourself) first, or every region selector returns empty. That's the easy mistake. The subtler one:

**The set of valid `type=` values depends on which layout engine produced the regions.** There is no universal vocabulary. Values are normalized (lowercase, spaces/underscores → hyphens), and you can match either the raw or the normalized name, but each engine emits its own set:

- **YOLO** (`analyze_layout()` default, DocLayout-YOLO): raw classes `title`, `plain text`, `abandon`, `figure`, `figure_caption`, `table`, `table_caption`, `table_footnote`, `isolate_formula`, `formula_caption` — normalized for selectors to `title`, `text`, `figure`, `table`, `caption`, `footnote`, `formula`, `unknown` (the three caption variants all become `caption`; `abandon` — headers/footers — becomes `unknown`).
- **TATR** (`analyze_layout('tatr')`, Table Transformer): `table`, `table-row`, `table-column`, `table-column-header`, `table-projected-row-header`, `table-spanning-cell`. Nothing else — TATR sees only table anatomy.
- **Other engines, other vocabularies.** `analyze_layout('doclayout')` (PP-DocLayoutV3) emits classes like `doc_title`, `paragraph_title`, `aside_text`, `chart`, `seal`, `vision_footnote`, `algorithm` — twenty-plus types; `paddle`, `surya`, and the VLM engine each have their own sets again.

So `region[type=paragraph_title]` silently matches nothing after a YOLO run — the selector is valid, the vocabulary isn't. When a region selector comes up empty: `page.find_all('region').inspect()` shows the `type` values that actually exist, and `region[model=yolo]` filters by which engine created a region.

```python
page.find_all('region')   # nothing yet — no layout analysis has run
```

## Reading order and what comes back

`find_all` returns matches sorted top-to-bottom, left-to-right (pass `reading_order=False` for document order), *then* applies collection pseudo-classes — so `:first` means "first in reading order." Fuzzy pseudo-classes (`:ocr`, `:closest`) instead sort by match quality. `find` is `find_all(...)[0]`-with-`None`: **it returns `None` on no match**, so guard before chaining.

Everything on this page is also a machine surface: the same registries that populate these tables accept third-party additions via `natural_pdf.selectors.registry` (`register_pseudo`, `register_attribute`, `register_post_pseudo`, `register_relational_pseudo`), so an installed plugin may extend this vocabulary.
