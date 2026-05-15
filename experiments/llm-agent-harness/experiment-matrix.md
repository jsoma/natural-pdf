# Experiment Matrix

The point of this matrix is to compare guidance strategies, not models alone. A weaker model with the right feedback loop may outperform a stronger model given an overstuffed context dump.

## Primary Question

Which guidance style makes LLMs most likely to write correct, deterministic `natural-pdf` code?

## Guidance Variants

### A. Docs Only

Inputs:
- task statement
- `docs/llms.txt`
- `docs/for-llms/common-patterns.md`

Tools:
- Python execution only

Purpose:
- baseline for whether concise docs are enough.

### B. Docs + Shell Exploration

Inputs:
- same as A

Tools:
- `rg`, `find`, Python snippets, direct `natural-pdf` calls

Purpose:
- tests the "normal coding loop" baseline.
- keeps model out of a specialized tool frame.

### C. Current `.to_llm()`

Inputs:
- docs
- current `page.to_llm(detail="standard", include_hints="none")`

Purpose:
- baseline for current implementation.
- expected risk: large context and noisy layout preview.

### D. Bounded Overview

Inputs:
- docs
- new `page.to_llm(mode="overview")`

Purpose:
- tests whether compact structural routing is enough.

### E. Bounded Overview + Descriptive Suggestions

Inputs:
- docs
- `page.to_llm(mode="strategy", suggestions="descriptive")`

Purpose:
- tests guidance without API-name anchoring.

### F. Bounded Overview + API Suggestions

Inputs:
- docs
- `page.to_llm(mode="strategy", suggestions="api")`

Purpose:
- tests explicit method recommendations.
- expected upside: faster path to correct APIs.
- expected risk: model blindly follows a wrong suggestion.

### G. Probe Suggestions

Inputs:
- docs
- `page.to_llm(mode="probes")`

Tools:
- can run suggested probes and inspect results.

Purpose:
- tests hand-holding without producing final code.

### H. Narrow Tools

Tools:
- selector sample
- table preview
- OCR status
- region text
- page overview

Purpose:
- tests whether strict tools improve search or force default-mode behavior.

### I. Expert Example Retrieval

Inputs:
- docs
- 1-3 retrieved examples from badpdfs/local cookbook/benchmark configs

Purpose:
- tests whether analogy beats structural descriptions.
- retrieval can be by tags, method names, or task embedding later.

### J. Hybrid Tutor

Inputs/tools:
- bounded overview
- probe suggestions
- expert examples only after first failed attempt

Purpose:
- tests whether staged help avoids over-conditioning first attempts.

## Metrics

Per episode:

- final correctness score
- first-attempt score
- number of iterations to pass
- execution failures
- runtime
- generated code length
- number of hardcoded coordinates
- number of raw regex operations
- number of `natural-pdf` idioms used
- use of OCR/layout/table APIs where appropriate
- whether code generalizes to a trap/variant PDF

Aggregate:

- pass rate by tag
- median iterations to pass
- common failure modes by guidance variant
- token/context cost
- time-to-first-valid-code

## Failure Modes To Track

- wrong selector syntax
- `NoneType` after failed `find()`
- table rows shifted across columns
- values include adjacent label/location text
- checkboxes inferred visually instead of using geometry
- headers/footers included
- OCR not applied to scanned pages
- excessive magic coordinates
- direct LLM extraction instead of deterministic code
- code works on one PDF but fails on trap/variant

## Initial Matrix

Run the first pass on three tasks only:

| Task | Tags | Variants |
| --- | --- | --- |
| `practice_health_inspection` | label/value, ruled table, checkbox | A, B, C, D, F, G |
| `guides_expenses` | borderless/guide table | A, B, C, D, F, G |
| `needs_ocr_text` | scanned/OCR | A, B, D, F, G |

Once the runner is stable, expand to real bad-pdf tasks and synthetic variants.
