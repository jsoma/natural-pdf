# Changelog

## Unreleased

### Breaking changes

- Text extraction now has four explicit, shared signature families: spatial,
  scalar, ordered aggregate, and selected aggregate. Common options have one
  spelling and meaning on every host that advertises them.
- `extract_text()` always returns `str`. Use `extract_text_result()` for raw
  text plus source spans, text maps, and word provenance.
- Spatial extraction defaults to `layout=False`; advanced layout controls now
  live in `TextLayoutOptions`. Layout failures raise `TextExtractionError`
  instead of silently returning a different representation.
- Removed text-extraction aliases and overloaded modes, including
  `use_exclusions=`, `page_separator=`, `preserve_line_breaks=`, selector/word
  modes, and `return_textmap=` on host methods. `PDFCollection` now exposes
  `extract_each_text()` rather than flattening document boundaries.
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
- Removed the inert `PDF(reading_order=...)` option. Remaining `PDF` constructor
  settings are keyword-only, so an old positional value cannot silently bind to
  a different option. Text ordering continues to use the existing geometric
  behavior.

### Fixes

- Text filtering, newline handling, whitespace normalization, stripping, bidi,
  exclusions, and separators now follow one validated order. Aggregate filters
  run per leaf, so they cannot consume structural separators or cross source
  boundaries.
- Citation provenance now uses exact result spans, including arbitrary
  separators and repeated identical lines.
- Kept edited words, characters, selectors, extraction, and Flow views coherent;
  removed stale Flow text/element caches.
- Applied exclusions to the image actually sent to OCR and made pure OCR
  extraction non-mutating.
- Included effective OCR options, crop/exclusion geometry, and runtime identity
  in cache/engine reuse decisions, avoiding unsafe reuse when identity is unknown.
- Validate and materialize complete classic and VLM OCR payloads before caching,
  extraction, or replacement; malformed provider output now raises `OCRError`
  without partially mutating text or table regions.
- Propagate table-cell extraction failures with row/column context instead of
  turning failed cells into blanks.
- Keep text classification fail-closed on extraction errors; only genuinely
  empty text falls back to vision, with a warning.
- Keep successful primary structured extraction results when the optional
  citation/confidence pass fails, while discarding partial metadata and emitting
  a `RuntimeWarning`.
- Declared `openpyxl` in the `export` extra, exposed `set_option` from the package
  root, and removed obsolete pytest asyncio configuration.

## 0.6.1 - 2026-04-09

Patch release focused on correctness, stability, and internal cleanup.

- Stabilized core state semantics around exclusions, cached pages, and partial-close behavior.
- Tightened selector execution and extraction service contracts for more consistent branch execution, dispatch, and mode resolution.
- Refactored guides generation and grid internals without changing the public guides/table API.
- Fixed release-blocking regressions in closest-string utilities, lazy page loading, and `to_llm()` garble-rate reporting.
- Added regression coverage around selectors, extraction wrappers, OCR/text updates, and lazy-page exclusion behavior.
