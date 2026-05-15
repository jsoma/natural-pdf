# Roadmap

## Stage 0: Stabilize The Question

Decision:
The goal is not "better direct LLM PDF extraction." The goal is "LLMs write better deterministic `natural-pdf` extraction code."

Deliverables:

- this planning folder
- first task manifest format
- first scoring contract

## Stage 1: Fix `.to_llm()` Footguns

Scope:

- add hard caps to layout preview
- make rendered diagnostics opt-in
- add tests for dense tiny text and scanned pages

Success:

- `page.to_llm()` on `pdfs/use-of-force-raw.pdf` stays under a configured character cap
- `page.to_llm()` does not render by default
- existing `tests/test_to_llm_features.py` still passes or is updated intentionally

## Stage 2: Minimal Local Harness

Build:

- task manifest loader
- subprocess runner for `extract(pdf_path)`
- JSON result capture
- recursive scorer
- artifact directory

Use hand-written "agent outputs" first, including one bad solution and one expert solution, before adding live model calls.

Success:

- can run one task end to end
- produces score report and saved artifacts
- catches a deliberately wrong extraction

## Stage 3: First Agent Loop

Build:

- prompt renderer
- guidance config
- iteration loop
- feedback formatter
- support for a single model/provider path

Start with three tasks:

- `practice_health_inspection`
- `guides_expenses`
- `needs_ocr_text`

Success:

- can compare at least three guidance variants on the same tasks
- records every submitted code version and subprocess result

## Stage 4: Guidance Matrix

Run:

- docs only
- docs + shell exploration
- current `.to_llm()`
- bounded overview
- API suggestions
- probe suggestions

Success:

- evidence table showing pass rate, first-attempt score, iterations, and failure modes
- clear recommendation for the next `.to_llm()` default or mode

## Stage 5: Synthetic Variant Generator

Build simple generated PDFs for:

- label/value form
- ruled table
- borderless table
- OCR-only page

Generate variants with shifted positions, renamed labels, extra rows, and trap substitutions.

Success:

- can test whether model-written code generalizes beyond one exact fixture

## Stage 6: Bad PDF Integration

Use:

- `bad-pdfs/Bad PDF Submission form_Submissions_2025-06-22.csv`
- `bad-pdfs/submissions`
- selected badpdfs.com expert examples

Build task manifests from real user goals and expert solutions.

Success:

- at least 10 real-world tasks tagged by failure mode
- at least 3 tasks with expert solution comparison

## Stage 7: Productize What Wins

Only after harness evidence:

- change `.to_llm()` defaults
- add new probe APIs
- update `docs/for-llms/common-patterns.md`
- add a user-facing "agent guide" if it demonstrably improves results

Avoid productizing a tool just because it seems elegant. Productize the loop and guidance that wins in the harness.
