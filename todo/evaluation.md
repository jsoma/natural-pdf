# Evaluation TODO List

This markdown collects open tasks and per-PDF follow-ups discovered during the automated evaluation suite.  Feel free to edit.

## Global engineering tasks

- [ ] Text-formatting detection (underline / strike-through association)
- [X] Visual colour-blob detection for OCR documents (use `page.detect_blobs()`)
- [ ] Dense-text / char-overlap handling (<6 pt fonts)
- [ ] **OCR-only mode** – allow `page.apply_ocr(engine=..., ignore_native=True)` to discard corrupted native text layers
- [ ] Graceful error handling for exclusions & spatial navigation when elements missing
- [ ] Streaming / chunked processing for 10 k+-page PDFs
- [X] Selector parser enhancement: support `*` universal selector (e.g. `*[width>100]`)

## Per-PDF follow-up queue

| Submission ID | Key issues observed | Suggested next step |
|---------------|--------------------|---------------------|
| ja6EqV1 | Tiny 2 pt fonts; table columns merge | Re-render at higher scale; adjust char_margin |
| obR1XQb | Needs anchor filter to isolate 300 3RD ST animal rows | Use text-layer + header filter, exclusion of other addresses |
| zxyRByM | Asset schedule tables – clean but repeated header every page | De-dup headers when concatenating; demo `page.to_dataframe()` |
| eqQ4N7q | Election-result table single-page | Confirm anchor under "REGISTERED VOTERS" optional |
| q4DXYk8 | Left narrative column causes table shift | Add exclusion for right narrative before TATR |
| ODXl8aR | Declarations form – key/value extraction | Show `.right(until)` anchor pairs; no location pages present |
| lbODDK6 | Bilingual two-column; requires vertical rule anchor | Flow example with vertical split; regex Ethiopic filter |
| Xxogz9j | Scanned Japanese; needs Paddle OCR + line detection | Document OCR language hint; show `detect_lines()` grid |
| eqrZ5yq | 4-page Annex 6 table | Use Flow stacking example; verify new docs |
| 44J0yXY | Goal unspecified → placeholder snippet; LLM failed to infer task | Ensure `description` → `goal` mapping; add template-check in collator |
| 5BjDYeM | Complex newspaper layout; no extraction goal | Detect news-page vs data table; propose column grouping demo |
| 7RKyoKL | Tamil NDC – tables on later pages but prompt left unknown | Teach LLM to look beyond first pages or ask for target pages |
| A72GaWl | Bus route map PDF – graphics heavy, OCR unlikely | Flag as map/diagram; advise `page.show()` + manual annotation |
| Ekl7KXB | 1790 census – curly-brace totals confuse LLM; unknown goal | Add curly-brace table heuristics; demonstrate `regex_rows()` |

_Add rows as evaluation proceeds. Use short notes; detailed summaries live in each `eval_results/<ID>/summary.json`_

# Documentation & API roadmap (new)

- [ ] Write an advanced tutorial on **exclusion-first** workflows – headers, footers, watermarks – before any extraction (see `docs/fix-messy-tables` new section).
- [ ] Add a Flows example for stitching **multi-page tables** (already added to `docs/reflowing-pages`). Verify performance on 100+ page docs.
- [ ] Expand anchor-based extraction docs: `.find(...).below(until=…)`, emphasise `until=` over pixel paddings.

# Selector / Navigation enhancements

- [X] Up-tree selection: allow `.parent('<selector>')` to move from a text element to its containing table (proof-of-concept: `table = page.find(text="header").parent('table')`).
- [REJECTED IDEA] Convenience helper `find_table(text_contains=..., model='tatr|yolo')` that wraps the parent trick.
- [ ] Investigate `.associate_marks()` API for linking rect/line marks (underline/strike) to nearby text; prototype algorithm in evaluation scripts.

# Geometry / Region helpers

- [X] **Percentage coordinates for `region()`**: allow `left="50%"`, `width="30%"`, etc., so callers can specify areas relative to page size without manual arithmetic.
- [ ] **Region union helper**: expose `Region.union(*others)` that returns the minimal bounding rectangle covering multiple regions.  Required for multi-anchor workflows and caption-plus-figure extraction.
- [ ] Investigate polygon support so a `Region` can be defined by an arbitrary convex hull; today all regions are axis-aligned rectangles so you must approximate with the bounding box.

## Rejected geometry helpers
- [REJECTED IDEA] `figure.save(...)` shortcut.  You can already call `fig_region.save_image('fig.png')` or use the global `page.save_image()` helper.

# Helper utilities

- [REJECTED IDEA] `extract_tables(pages=..., merge_headers=True, as_df=True)` high-level wrapper (renamed from previous `extract_all_tables`). Should internally use anchor/filter strategy, not "every page is a table" assumption.
- [X] Provide `page.to_dataframe(headers="first")` sugar for simple table extraction lists. Implemented as page.extract_table().to_df()
- [REJECTED IDEA] `extract_tables(kind='tatr')` flag. Prefer the explicit `region.extract_table(extractor='tatr')` with a clear anchor instead of magic defaults.
- [REJECTED IDEA] `.find_table()` shorthand (duplicate of earlier rejection).
- [REJECTED IDEA] `script=` parameter for selector filters – use Python lambdas or `.filter()` chaining instead.
- [REJECTED IDEA] `stream_pages()` helper – just `for page in pdf.pages:` covers the use-case.
- [REJECTED IDEA] Timeline grid visualiser – out of scope for core library; consider external viz notebooks.
- [REJECTED IDEA] RTL flipping for tables – pandas handles bidi text, table direction determined by PDF layout.
- [DONE] First-row-as-header convenience – implemented via `TableResult.to_df(header='first')` / `skiprows=`.

# LLM prompt / eval improvements

- [ ] Update enrichment prompt to penalise blanket "run TATR" plans unless evidence warrants it.
- [ ] Add post-processing pipeline that flags suggestions still using pixel coords and suggests anchor refactor.
- [ ] Encourage use of the new `TableResult` API (`tbl.df` / `tbl.to_df()`) – many suggestions still unpack raw lists then build a DataFrame manually.
- [ ] Discourage placeholder anchors like `'AnchorText'`, `'Texts'`, `'Also'`; require the model to guess a plausible real anchor or leave a TODO comment at **goal time**, not code-time.
- [ ] When `Thoughts` mention `scanned_image`, ensure `page.apply_ocr()` appears **before** any `page.find()` calls; flag omissions.
- [ ] Prefer `.parent('table')` after anchoring on header text over global `page.find('table')` + positional index – reduces false positives on pages with multiple tables.
- [ ] Replace fallback `page.analyze_layout()` / `'YOLO fallback'` jargon with explicit extractor names (`'detectron'`, `'tatr'`) or anchor-based extraction; adjust prompt.
- [ ] Auto-detect "Goal tag: unknown" in summaries and surface as evaluation warning so we can enrich the metadata or tweak the LLM system prompt.

# Stress-Test Harness Roadmap (new)

> These items extend the bad-PDF evaluation suite so a single nightly run exercises **performance, correctness, and ergonomics** across the whole Natural-PDF API.
>
> Each sub-item should log metrics or produce assertions so we can flag regressions automatically.

- **Size & performance metrics**
  - [ ] Include a 10 k-page PDF and a 1 GB scanned book.  Record wall-clock time, pages/sec, peak RAM and written-image MB.
  - [ ] Add an image-heavy PDF (hundreds of inline PNG/JPEG) and track cache memory release.

- **Concurrency / re-entrancy**
  - [ ] Run N=10 PDFs concurrently (ThreadPool / asyncio) to expose thread-safety bugs in glyph cache, OCR pool, etc.

- **Robust error paths**
  - [ ] Feed password-protected, zero-byte, and corrupt-xref PDFs. Assert clean `NaturalPDFError` with helpful message.

- **Language & script coverage**
  - [ ] Add PDFs that exercise **right-to-left scripts** (Arabic, Hebrew) including pages with mixed RTL/LTR tables (e.g. Arabic text with Western digits).
  - [ ] Add complex scripts (Khmer/Burmese) and vertical Japanese + Latin footers.
  - [X] Verify `page.extract_text()` returns Unicode in logical reading order for RTL paragraphs; assert a known Arabic snippet appears without reversed glyph order.
  - [ ] Table direction check: ensure `TableResult.to_df()` preserves column order when headers are RTL and Pandas renders them right-aligned.
  - [ ] LLM task: list every script on the first 3 pages and highlight any bidirectional reordering issues it sees compared to `extract_text()` output.

- **Layout primitives**
  - [ ] Forms with check-marks/key-value pairs.
  - [ ] SVG-heavy technical drawings.
  - [ ] Newspaper columns with ads interrupting tables.
  - [ ] LLM prompt: provide three selectors to isolate only checked boxes.

- **Geometry helper coverage**
  - [ ] Synthetic docs that require percentage `region()`, `Region.union()`, and multi-page table stitching to succeed.

- **Table semantics**
  - [ ] Verify `.to_df(header='first')` sets correct dtypes.
  - [ ] Wide table dtype inference; de-duplicate repeated headers across pages.
  - [ ] LLM task: output JSON schema of resulting DataFrame.

- **CLI / API smoke tests**
  - [ ] Invoke every public call with default kwargs on a small page; flag default-value regressions.

- **Documentation fitness**
  - [ ] Randomly pick 3 PDFs, drop docs + summary into a temp dir, prompt LLM to write an extraction script using *only* `docs/quick-reference`.
  - [ ] Score compilation success and runtime warnings; surface doc gaps automatically.
