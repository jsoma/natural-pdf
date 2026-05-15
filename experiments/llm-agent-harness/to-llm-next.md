# `.to_llm()` Next Steps

`.to_llm()` should help an agent decide what to inspect next and what extraction strategy is plausible. It should not dump the document into context or try to replace `extract_text()`, `show()`, selector probes, or targeted region inspection.

## Current Problems

1. Layout preview can explode on dense pages.
   - Example: `pdfs/use-of-force-raw.pdf` produced roughly 100k characters because one tiny-text row contained thousands of elements.
   - Any preview section needs hard caps by line count, character count, and elements per line.

2. The default output mixes summary, examples, and diagnostics.
   - Useful for humans, but not ideal for an agent deciding the next action.
   - The agent needs routing signals and suggested probes more than complete samples.

3. Suggestions are too shallow.
   - Current hints map broad observations to methods.
   - Better hints should propose next probes and explain what result would confirm or reject an approach.

4. It violates the original cheap-compute promise.
   - `render_pixel_histogram()` calls `to_image()`, so default `.to_llm()` can render.
   - Rendering should be opt-in or only included if previously computed.

5. It does not expose uncertainty well.
   - The output says what was observed, but not which observations are weak, noisy, capped, or potentially misleading.

## Proposed Shape

Introduce explicit modes instead of one ever-growing text representation:

```python
page.to_llm(mode="overview")
page.to_llm(mode="strategy")
page.to_llm(mode="probes")
page.to_llm(mode="detail", target="tables")
page.to_llm(mode="region", region=...)
```

The existing `detail="brief|standard|full"` can remain, but the harness should test whether task-oriented modes work better.

## Mode Definitions

### `overview`

Purpose: cheap routing.

Includes:
- page dimensions
- word count and source breakdown
- image coverage
- line/rect counts
- top style tiers with capped samples
- likely page type tags: scanned, ruled form, ruled table, borderless table, dense tiny text, repeated sections

Excludes:
- full layout preview
- pixel histogram
- long text samples
- method suggestions

### `strategy`

Purpose: tell the agent which extraction families are plausible.

Includes:
- observations
- confidence level for each observation
- risks
- 3-5 candidate strategies

Example:

```text
LIKELY STRATEGIES
  high: ruled table/form via vector lines
    evidence: 10 horizontal, 5 vertical lines; rows align with text
    probe: page.extract_table().to_df().head()
  medium: label/value extraction for header fields
    evidence: bold labels at x=50 followed by regular text on same row
    probe: page.find('text:contains("Site")').right(until="text").extract_text()
```

### `probes`

Purpose: produce runnable inspection snippets, not final extraction code.

Includes bounded snippets such as:

```python
print(page.find_all("text:bold").extract_each_text()[:20])
print(page.extract_table().to_df().head())
print(page.find("text:contains('Site')").right(until="text").extract_text())
```

These probes should be short, deterministic, and easy for the harness to run.

### `detail`

Purpose: focused view for a selected hypothesis.

Targets:
- `target="tables"`
- `target="forms"`
- `target="checkboxes"`
- `target="ocr"`
- `target="repeated_sections"`
- `target="multicolumn"`

Each target should expose relevant evidence only.

## Hard Bounds

Every mode should enforce:

- max total characters
- max section characters
- max layout preview characters
- max elements per row
- max rows
- max examples per style/cluster

Proposed defaults:

```python
max_chars=6000
max_section_chars=1500
max_preview_lines=12
max_preview_line_chars=240
max_elements_per_preview_line=20
max_examples=8
```

When output is capped, say so explicitly:

```text
LAYOUT PREVIEW
  capped: 12 of 318 detected lines, 20 elements per line
```

## Suggestion Style

Test at least three variants:

1. No suggestions: observations only.
2. Descriptive suggestions: extraction family, no method names.
3. API suggestions: concrete `natural-pdf` method names and probes.

Suggestions should avoid pretending to know the answer. Prefer:

```text
try this probe
```

over:

```text
use this method
```

## Near-Term Implementation Tasks

1. Add character caps to `render_layout_preview()`.
2. Move pixel histogram behind an explicit `include_rendered_diagnostics=True` flag.
3. Add a cheap page-type classifier based on existing parsed primitives.
4. Add `mode="overview"` and `mode="probes"` while keeping the existing signature compatible.
5. Add tests with:
   - `pdfs/01-practice.pdf`
   - `pdfs/use-of-force-raw.pdf`
   - `pdfs/needs-ocr.pdf`
   - `pdfs/multicolumn.pdf`
6. Make the harness compare old `.to_llm()` and new modes before replacing defaults.
