# Changelog

## Unreleased

### Breaking changes

- Exporters now raise `ExportError` (instead of bare `RuntimeError`) and fail
  closed: a page that cannot be processed raises with page context instead of
  being silently skipped (`create_searchable_pdf` accepts `on_error="skip"` to
  opt back into skipping). Exporter options are keyword-only and validated
  (`pages`, `legend_scope`, `output_format`, `method`, `dpi`). All PDF outputs
  are written atomically (temp file + rename). `pikepdf.PasswordError`
  propagates from `create_original_pdf` as documented.
- `export_training_data(overwrite=True)` refuses to delete a non-empty
  directory it did not create (a `.natural-pdf-export` marker file identifies
  previous exports). CSV metadata columns are now `x0, top, x1, bottom`
  (previously mislabeled `y0`/`y1` for values that were top/bottom).
- Classification is fail-closed end to end: malformed or non-numeric provider
  payloads raise `ClassificationError` instead of silently producing
  `category=None`; batch paths (`pdf.classify_pages`, `classify_all`) raise on
  result-count mismatches and on content errors other than genuinely empty
  documents; an unrecognized model id raises instead of downloading the model
  to probe its mode; `multi_label=True` is rejected in vision mode (the
  pipeline silently ignored it). Classification signatures are keyword-only
  after `labels`, and `Region.classify`/`Element.classify` have real
  signatures. `device=` now works (it previously crashed with a duplicate
  keyword or was silently ignored).
- `to_llm()` validates `detail` and `include_hints` and rejects unknown values;
  all five builders (page/region/collection/element/pdf) accept `max_chars`
  and cap output; `pdf.to_llm()` bounds page iteration with `max_pages=50` by
  default. Layout-extraction failures inside `to_llm` degrade to plain text
  with an explicit marker instead of raising or silently changing shape.
- Deskew: `deskew_kwargs` are validated per engine (unknown keys raise);
  `grayscale=` is honored instead of silently ignored; deskew/skew-detection
  signatures are keyword-only; a `None` detection result (blank page) is
  cached instead of re-running the sweep on every access; the projection
  engine's sweep parameters are tunable via `deskew_kwargs`.
- Search: `pdf.search()` embedding caches are invalidated when page text
  changes (previously results were silently stale after `apply_ocr()` etc.);
  `PDFCollection.search()` reuses per-PDF caches; `top_k < 1` and empty
  queries raise `ValueError`.
- Directional methods reject a number in the cross-direction mode slot:
  `.below(width=200)` / `.above(width=200)` / `.left(height=50)` /
  `.right(height=50)` now raise `TypeError` (the value was silently treated
  like `"element"`, producing a plausible but wrong region). `width` on
  above/below and `height` on left/right are mode strings (`"full"` /
  `"element"`); the numeric extent parameter is the other one.
- Removed the inert `tolerance`/`row_tolerance` parameters from
  `Guides.from_content`, `add_content`, `from_headers_and_row_anchors`, and
  `page.extract_table_guided` — they were accepted and never used.
- Removed `page.filter_elements()` (raised `NameError` on any call; nothing
  used it) and `page.annotate_checkboxes()` plus the `CheckboxAnnotator`
  widget (its drawn boxes never reached Python; it always returned zero
  regions).
- `ElementCollection.viewer()` no longer accepts `title=` and raises
  `ValueError` for empty collections instead of returning `None`; it now
  works again (it had been passing arguments a zero-argument `Page.viewer()`
  rejected, silently returning `None`). `Page.viewer()` accepts `resolution`,
  `elements_to_render`, and `include_attributes`, shows excluded elements
  (it is a debugging view), and `Region.viewer()` raises on render failure
  instead of returning `None`.
- `extract(engine="vlm")` now honors `client=` and
  `natural_pdf.set_default_client()` with the same semantics as
  `apply_ocr(engine="vlm")` (an explicit `client=` is used; the default
  client applies only when neither `model=` nor `client=` is passed), and
  runs through the shared VLM client stack — gaining MLX model support and
  the same processor pixel caps as OCR. The internal `HFVLMAdapter`
  (`natural_pdf.extraction.vlm_adapter`) was removed; previously a `client=`
  passed with `engine="vlm"` was silently ignored.
- Removed import paths (internal modules with no documented API):
  `natural_pdf.ocr.ocr_manager` (re-export shim), `natural_pdf.ocr.ocr_factory`
  (`OCRFactory` — superseded by the engine registry), the per-capability
  `natural_pdf.engine_registry.{checkbox,classification,deskew,guides,layout,
  selectors}` submodules (their registration functions remain importable from
  `natural_pdf.engine_registry`), and `natural_pdf.templates`. The
  `natural_pdf.analyzers.{guides,layout,checkbox}` packages are permanent
  re-export shims for the new top-level packages; `GuidesOcrResult` remains
  importable there as an alias of `GuideOCRResult` (renamed earlier with a
  changed field shape).
- `natural_pdf.set_default_client()` can no longer receive document images
  from calls that pin a model: `extract(engine="vlm", model=...)`,
  `apply_ocr(engine="vlm", model=...)`, and shorthand VLM OCR engines
  (`glm_ocr`, ...) now always run locally unless a client is passed
  explicitly. Previously the shared generate() path backfilled the global
  default client, sending pages to a remote endpoint despite an explicitly
  local model.
- `export_training_data(overwrite=True)` builds into a staging directory and
  replaces the previous export only after the new one succeeds — validation
  errors, empty sources, and mid-build failures leave the prior export
  untouched (previously it was deleted up front).
- `ElementCollection.viewer()` raises `ValueError` for collections spanning
  multiple pages (previously every element was overlaid on the first page).

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

- Searchable PDF text layers no longer double-escape `&`, `<`, and `>`
  (previously every affected word contained literal `&amp;`), and words on the
  same visual line are grouped into one hOCR line again, restoring inter-word
  spaces in extracted text. Both regressions are covered by new round-trip
  tests.
- `pdf.to_llm()` headers show the source filename (previously always the
  literal "PDF"); typography summaries count strikethrough text (the counter
  read a nonexistent attribute and was permanently zero); checkbox regions in
  `.inspect()` get their `state` column (the guard was unreachable); region
  stat summaries run for layout-detected regions.
- The garble-rate diagnostics dependencies (`pyspellchecker`, `langdetect`)
  are included in `natural-pdf[all]` via the `quality` extra; the feature was
  previously unreachable on every documented install path. The reported
  language is labeled "assumed from script" when langdetect did not run.
- `HocrTransform` deprecation warning no longer crashes the logging machinery;
  an hOCR file with no `ocr_page` element raises `HocrTransformError` instead
  of `AttributeError`; URL downloads in exporters have timeouts.
- `optional_imports` distinguishes installed-but-broken packages from missing
  ones instead of reporting both as "not installed".
- `.describe()` reports text sources whenever any text is non-native, so a
  fully-OCR'd page says so; broken element properties render as `<error>` in
  `.inspect()` instead of an empty cell; `_color_to_hex` honors its "hex or
  None" contract with consistent casing.
- Guide-grid OCR resolves engine, languages, min_confidence, device, and
  resolution from the host page/region instead of from freshly constructed
  probe regions, and windowed OCR honors the host's exclusions — a Region
  configured for e.g. French at 333 DPI no longer silently OCRs in English
  at 150 DPI with masked content exposed. Auto window batching enforces
  `max_area_px` on the initial row window (individually compliant cells can
  no longer merge past the pixel budget).
- `PDFCollection.classify_all()` and `PDF.classify_pages()` resolve the
  classification engine once (previously twice, creating duplicate instances
  under context-scoped caching) and route kwargs correctly: `device=` and
  other engine options reach the engine, `resolution=` reaches content
  rendering. Mismatched label/score array lengths in provider payloads raise
  `ClassificationError` instead of silently truncating; unsupported classic
  OCR payload shapes raise `OCRError` instead of becoming an empty page.
- `to_llm(max_chars=N)` is a hard cap even for tiny N (the truncation suffix
  no longer overflows the limit, and empty-collection output is capped too).
- Search argument validation (empty query, `top_k < 1`) applies before the
  empty-collection early return, so empty and populated collections reject
  the same inputs; embedding-cache fingerprints are length-prefixed so texts
  containing separator bytes cannot collide across page boundaries.
- The interactive viewer emits valid markup: element hit-target `<div>`s
  live in the elements layer and highlight `<rect>`s inside the SVG
  (previously divs inside `<svg>` aborted browser SVG parsing, breaking
  click highlighting). Exporter temp files use unique per-call names so
  concurrent writers to the same output path cannot delete each other's
  work.
- The `register_*_engine` helpers forward the engine-lifecycle controls
  (`lifetime`, `cache_key`), so third-party engines registered through them
  can opt into singleton reuse.


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
