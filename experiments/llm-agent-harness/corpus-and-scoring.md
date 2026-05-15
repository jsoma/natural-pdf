# Corpus And Scoring

The corpus should start small and deliberately cover extraction patterns, then grow into real-world failures from `pdfs/` and `bad-pdfs/submissions`.

## Corpus Layers

### Layer 1: Synthetic Curriculum

Generate or curate small PDFs where the expected extraction is unambiguous.

Recommended task types:

- label/value pairs
- repeated label/value blocks
- ruled tables
- borderless aligned tables
- multi-line table cells
- checkboxes
- repeated headers/footers
- multi-page continuing tables
- scanned OCR-only pages
- low-confidence OCR text
- tiny text tables
- multi-column prose
- RTL/multilingual tables

Synthetic tasks are useful because we can create variants cheaply:

- rename labels
- shift coordinates
- add headers/footers
- add extra rows
- reorder sections
- change fonts
- add OCR noise
- add trap substitutions

### Layer 2: Existing Local PDFs

Seed with PDFs already used by tests and benchmarks:

- `pdfs/01-practice.pdf`
- `pdfs/01-practice-trap.pdf`
- `pdfs/guides-expenses-sample.pdf`
- `pdfs/Atlanta_Public_Schools_GA_sample.pdf`
- `pdfs/Atlanta_Public_Schools_GA_sample-trap.pdf`
- `pdfs/m27.pdf`
- `pdfs/m27-trap.pdf`
- `pdfs/needs-ocr.pdf`
- `pdfs/multicolumn.pdf`
- `pdfs/use-of-force-raw.pdf`
- `pdfs/hebrew-table.pdf`
- `pdfs/tiny-text-tables.pdf`
- `pdfs/pak-ks-expenses.pdf`

Many of these already have expert extraction code in `benchmark/configs/` or tests.

### Layer 3: Bad PDF Submissions

Use `bad-pdfs/submissions` and the submission CSV to build tasks from real user goals.

The CSV provides:

- extraction goal
- source/context
- why the PDF is bad
- language/script
- OCR/handwriting hints

This metadata can become task text and tags. Expected outputs may need to be produced manually or with expert `natural-pdf` solutions.

### Layer 4: badpdfs.com Expert Examples

The public site contains worked examples and method usage metadata. Use it as a retrieval/evaluation source when available:

- task statement
- original PDF
- expert code
- methods used
- explanation

Do not let the model see the exact expert solution during normal evaluation. Use it for:

- expected method tags
- offline scoring
- example retrieval in dedicated retrieval variants
- human comparison

## Task Manifest Fields

```yaml
id: string
pdf: path
pages: list[int] | all
goal: string
output_format: json | csv | text
expected: path
schema: path | inline
tags: list[string]
known_pitfalls: list[string]
expert_solution: path
variant_of: string | null
allow_ocr: bool
allow_vision: bool
max_iterations: int
timeout_seconds: int
```

## Scoring

Use layered scoring instead of one pass/fail bit.

### Output Correctness

- exact field match where possible
- normalized text match for whitespace and quote variants
- numeric/date normalization where appropriate
- table shape
- row count
- column alignment
- required missing/extra fields

### Code Quality

This is not style scoring. It should measure reproducibility risk.

Track:

- hardcoded coordinates
- hardcoded page numbers
- raw regex count
- unguarded `.find(...).right()` chains
- use of `page.extract_text()` followed by ad hoc parsing
- use of appropriate `natural-pdf` APIs
- whether code closes PDFs
- whether code handles empty results

### Generalization

Run the same code against:

- trap PDF
- shifted synthetic variant
- extra-row variant
- alternate font variant
- page subset/multipage variant

Score both original and variant. A solution that only passes the exact fixture should not be treated as robust.

### Interaction Cost

Track:

- number of attempts
- number of probes
- number of failed subprocess runs
- token/context estimate
- wall time

## Initial Grading Strategy

Start simple:

1. Require `extract(pdf_path)` to return JSON-serializable data.
2. Normalize strings with whitespace collapse and curly quote normalization.
3. Compare expected fields recursively.
4. Compare tables by header-normalized rows.
5. Record code-risk metrics with simple AST and text scans.

Add more sophisticated grading only after the first tasks are running.
