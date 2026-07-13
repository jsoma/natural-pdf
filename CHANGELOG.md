# Changelog

## Unreleased

### Breaking changes

- OCR now has three explicit modes: recognition (the default), detection via
  `detect_only=True`, and custom function mode via `function=`. The old
  `ocr_function=` spelling is removed and rejected. `engine` is the only
  positional argument after the host; `function` and all other controls are
  keyword-only.
- OCR mutation APIs now use explicit `replace="ocr" | "all" | "none"` modes;
  boolean replacement values are rejected. The default is `"ocr"`.
- Detection refreshes geometry without deleting recognized or native text.
- `PDF.apply_ocr` accepts `pages=` and `show_progress=`; `PDFCollection.apply_ocr`
  additionally accepts `max_workers=`.
- OCR on guide grids is scoped through `guides.cells`, `guides.rows`, and
  `guides.columns`; use `guides.cells.plan_ocr(...)` and
  `guides.cells.apply_ocr(...)`. `Guides.apply_ocr` is no longer public.
- OCR corrections are explicit with `ElementCollection.correct_ocr(...)`.
- Region, Flow, guide-window, and comparison replacement is scoped to the
  processed geometry. `detect_only=True` never removes native or recognized
  text.

### Fixes

- Kept edited words, characters, selectors, extraction, and Flow views coherent;
  removed stale Flow text/element caches.
- Applied exclusions to the image actually sent to OCR and made pure OCR
  extraction non-mutating.
- Included effective OCR options, crop/exclusion geometry, and runtime identity
  in cache/engine reuse decisions, avoiding unsafe reuse when identity is unknown.

## 0.6.1 - 2026-04-09

Patch release focused on correctness, stability, and internal cleanup.

- Stabilized core state semantics around exclusions, cached pages, and partial-close behavior.
- Tightened selector execution and extraction service contracts for more consistent branch execution, dispatch, and mode resolution.
- Refactored guides generation and grid internals without changing the public guides/table API.
- Fixed release-blocking regressions in closest-string utilities, lazy page loading, and `to_llm()` garble-rate reporting.
- Added regression coverage around selectors, extraction wrappers, OCR/text updates, and lazy-page exclusion behavior.
