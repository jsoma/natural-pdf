# Performance Experiment Synthesis

Status: initial experiment framework and candidate patches are in place. Full
screening and confirmation matrices should be generated under
`experiments/performance/results/`.

## Current Baseline Signals

- Tiny text table extraction is the pathological outlier. The scratch profile
  showed `real:tiny-text-tables` at about 138 seconds with `pdfplumber` calling
  `char_in_bbox` about 38 million times.
- Page materialization is broad impact. Profiles for multipage and election
  workflows show page creation and `ElementManager.load_elements` near the top,
  with decoration detection visible in table-heavy PDFs.
- Repeated selectors and exclusions are real but workload-dependent. Atlanta
  shows hundreds of repeated `find_all` and exclusion filtering calls.
- Rendering appears inside extraction-style workflows and should be measured as
  its own track.
- Cache candidates must be labeled as repeated-work improvements. They are not
  production recommendations unless they also improve cold single-use cases.

## Candidate Patch Matrix

| Track | Candidate | Type | Primary question |
|---|---|---|---|
| control | `noop/baseline_control` | control | Does patch plumbing preserve baseline behavior? |
| tiny text | `tiny_text/prefilter_pdfplumber_chars` | structural | Can prefiltering reduce pdfplumber cell assignment work? |
| tiny text | `tiny_text/tuned_text_strategy` | comparator | Do stricter text settings reduce work without breaking correctness? |
| tiny text | `tiny_text/text_engine_for_text_strategy` | structural comparator | Can Natural PDF text tables avoid pdfplumber text-strategy blowups? |
| page materialization | `page_materialization/defer_decorations` | structural upper bound | How much eager decoration detection costs plain extraction? |
| page materialization | `page_materialization/skip_char_elements` | structural upper bound | How much eager char `TextElement` wrapping costs? |
| selectors | `selectors/exact_query_cache` | cache-only | How much repeated-query time is cacheable? |
| selectors | `selectors/skip_exclusions_upper_bound` | structural upper bound | How much selector time could exclusion-mask work recover? |
| guide/table | `guide_table/skip_alt_text_scan` | structural upper bound | How much table extraction pays for alt-text scanning? |
| guide/table | `guide_table/default_word_cells` | structural | Can guide-built cells use the recent fast word assignment path by default? |
| guide/table | `guide_table/vector_lines_only` | structural comparator | How much render-driven guide line detection can vector lines avoid? |
| rendering | `rendering/render_cache` | cache-only | How much repeated rendering is present in workflows? |
| workflow reuse | `workflow_pdf_reuse/pennsylvania_pdf_cache` | cache/workflow | How much benchmark reopen cost affects election extraction? |
| vector | `vector/decorations_ephemeral` | structural | Can per-call arrays speed decoration detection without reuse? |
| vector | `vector/decorations_page_store` | structural/reuse | Does reusing prepared-char arrays beat per-call decoration arrays? |
| vector | `vector/spatial_char_filter_ephemeral` | structural | Can rectangular char filtering use vector masks safely? |
| vector | `vector/region_overlap_page_store` | structural | Can rectangular region overlap and exclusion checks use vector masks? |
| vector | `vector/table_word_assignment` | structural | Can guide table cells assign words with `searchsorted`? |
| vector | `vector/simple_selector_fast_path` | structural | Can common simple selectors filter/sort with arrays? |
| vector | `vector/lazy_text_elements_upper_bound` | structural upper bound | How much eager char `TextElement` creation can lazy materialization recover? |

## Recommended Screening Commands

The predefined matrix runner can execute baseline-vs-patch pairs:

```bash
uv run python experiments/performance/run_matrix.py --mode screening
uv run python experiments/performance/run_matrix.py --mode screening --track selectors
uv run python experiments/performance/run_matrix.py --dry-run
```

Use one-iteration screening without profiler or tracemalloc:

```bash
uv run python scripts/perf_natural_pdf.py \
  --output experiments/performance/results/screening-<candidate> \
  --experiment-label screening-<candidate> \
  --patch experiments/performance/patches/<track>/<candidate>/patch.py \
  --iterations 1 --warmups 0 --no-tracemalloc --profile-slowest 0 \
  --cases <track-specific-cases>
```

Use confirmation only for candidates that pass correctness and improve at least
one named workload:

```bash
uv run python scripts/perf_natural_pdf.py \
  --output experiments/performance/results/confirm-<candidate> \
  --experiment-label confirm-<candidate> \
  --patch experiments/performance/patches/<track>/<candidate>/patch.py \
  --iterations 5 --warmups 1 --no-tracemalloc --profile-slowest 0 \
  --cases <track-specific-cases>
```

Run profiling separately for finalists with `--profile-slowest`.

## Production Recommendation Rules

- Prefer structural fixes that improve cold single-use runs.
- Treat cache-only wins as diagnostic evidence unless repeated-use is the target
  workflow.
- Reject candidates that materially regress `real:01-practice`.
- Require targeted correctness tests for each candidate before recommending a
  production implementation.

## Screening Results 2026-05-10

These are one-iteration screening runs, so treat exact percentages as directional.
The strongest result is structural and guide-related: avoid pixel line detection
when vector lines are already available.

| Track | Candidate | Result | Recommendation |
|---|---|---:|---|
| guide/table | `vector_lines_only` | `micro:guide-table:01-practice` 210.7ms -> 46.0ms; `real:0500000US42001` 1038.1ms -> 843.2ms | Promote to production design: prefer vector/auto-vector guide line detection before rendering. |
| page materialization | `defer_decorations` | `real:0500000US42001` 1821.8ms -> 987.0ms; `real:01-practice` 214.8ms -> 141.6ms | Investigate a lazy/opt-in decoration model. Do not blanket-disable decorations. |
| page materialization | `skip_char_elements` | `micro:page-materialize:m27` 153.5ms -> 71.4ms; smaller wins on real workflows | Use as evidence for lazy char element creation. Not safe as-is. |
| tiny text | `text_engine_for_text_strategy` | 28090.4ms -> 13847.8ms but returned 0 table rows | Reject as implemented; speedup is not useful without table correctness. |
| tiny text | `prefilter_pdfplumber_chars` | 28090.4ms -> 28251.8ms; table shape preserved | Reject; no speed benefit. |
| tiny text | `tuned_text_strategy` | 28090.4ms -> 30619.8ms; shape changed from 58 to 69 columns | Reject; slower and likely less correct. |
| guide/table | `default_word_cells` | Small wins on election/multipage, regression on guides-expenses; mock contract test failed | Do not change default globally; consider explicit opt-in or narrower auto mode. |
| guide/table | `skip_alt_text_scan` | No meaningful improvement | Deprioritize. |
| selectors | `exact_query_cache` | Small 0.9-10.7% wins | Diagnostic only; cache invalidation makes it a weak production route. |
| selectors | `skip_exclusions_upper_bound` | No speedup and broke Atlanta workflow | Reject as a direction. |
| rendering | `render_cache` | Small 2.8-8.5% wins | Diagnostic only; prefer avoiding unnecessary renders. |
| workflow reuse | `pennsylvania_pdf_cache` | 1.4% wall-time win | Not a library priority. |

Targeted correctness checks:

- `vector_lines_only`: `tests/test_guides_partial.py` and
  `tests/test_guides_extract_table.py` passed under the patch.
- `defer_decorations`: `tests/test_tables_integration.py` and
  `tests/test_words_vs_find_all_text.py` passed under the patch.
- `default_word_cells`: failed an existing guide contract test because it changes
  the default forwarded `cell_extract` and `cell_newlines` arguments.

Suggested production order:

1. Prefer vector guide line detection when vector lines exist; avoid pixel/render
   line detection unless needed.
2. Design lazy decoration detection or an extraction-mode fast path that avoids
   eager strike/underline/highlight work when those attributes are unused.
3. Design lazy char `TextElement` materialization so word-heavy workflows do not
   pay char-wrapper cost up front.
4. Continue tiny-table work with a new, correctness-preserving structural
   candidate: construct table cells from detected/explicit guide geometry and
   assign words directly, rather than routing text-strategy tables through
   pdfplumber or Natural PDF's current text engine wholesale.

## Vectorization Experiment Addendum

The vector candidates are designed to separate structural cold-run wins from
derived-array reuse wins:

- `decorations_ephemeral` and `decorations_page_store` compare one-off arrays
  against page-level reuse for the measured decoration cost.
- `spatial_char_filter_ephemeral` targets region text extraction and tiny-text
  layout paths without relying on query-result caches.
- `region_overlap_page_store` and `simple_selector_fast_path` target repeated
  selector/navigation workloads while reporting fast-path and fallback counts.
- `table_word_assignment` tests `searchsorted`-based guide cell assignment.
- `lazy_text_elements_upper_bound` keeps raw char dicts but delays char
  `TextElement` construction until chars are actually requested.

Vector metrics in `baseline.json` report array builds, timings, byte counts,
mask allocation estimates, and fast-path/fallback counts. Reject any candidate
that gets its win only by breaking correctness, regresses `real:01-practice`
materially, or grows memory without a documented benefit.

## Vector Screening Results 2026-05-10

All vector candidates installed and ran through smoke workloads. Targeted
correctness checks passed for the relevant decoration, selector/spatial, table,
and lazy-materialization test groups.

| Candidate | Screening result | Confirmation result | Recommendation |
|---|---:|---:|---|
| `lazy_text_elements_upper_bound` | broad wins: `micro:page-materialize:m27` 109.7ms -> 69.6ms; `real:multipage-table` 323.2ms -> 230.5ms | confirmed: `micro:page-materialize:m27` +54.0%, `real:01-practice` +22.5%, `real:multipage-table` +24.3%, RSS +0.4-6.1% | First production track: lazy native char `TextElement` materialization with raw char dict retention. |
| `region_overlap_page_store` | `real:Atlanta_Public_Schools_GA_sample` 212.3ms -> 180.4ms; neutral/regressive elsewhere | confirmed: Atlanta +20.7%, `real:01-practice` -1.2%, other workloads neutral/slightly negative | Implement only for exclusion-heavy rectangular workflows, not as a broad selector rewrite. |
| `decorations_page_store` | mixed: `real:multipage-table` +13.8%, `real:guides-expenses-sample` +3.9%, `real:0500000US42001` -1.5% | confirmed: `real:multipage-table` +13.7%, `real:0500000US42001` +4.1%, `real:01-practice` -2.1%, `micro:page-materialize:m27` -2.2% | Keep as a targeted follow-up for line/rect-heavy pages; not first. |
| `decorations_ephemeral` | fast on some real cases, regressed `micro:page-materialize:m27` by 18.6% | not confirmed | Reject broad per-call arrays; array construction cost is too visible. |
| `spatial_char_filter_ephemeral` | regressed tiny text and m27 text extraction; allocated ~10-20MB mask/array bytes on tiny workloads | not confirmed | Reject current approach; build arrays once or avoid this path for one-off char filtering. |
| `simple_selector_fast_path` | regressed common selector workflows despite fast-path hits | not confirmed | Reject current approach; object/list conversion and array builds cost more than simple predicates save. |
| `table_word_assignment` | no fast-path hit in current guide workflows; timings neutral/noisy | not confirmed | Defer until workflows use `cell_extract="words"` or a direct guide-word table path exists. |

Updated production order:

1. Lazy native char `TextElement` materialization.
2. Rectangular exclusion/region masks for selector-heavy workflows like Atlanta.
3. Targeted vectorized decoration detection only after the lazy char work, and
   only for pages where line/rect candidate counts make the math favorable.
4. Do not pursue one-off spatial char masks, simple selector vectorization, or
   table word assignment from these prototypes without a narrower workload.

## Deeper Vector Follow-Up 2026-05-10

I inspected the first pages of the noisy cases:

- `tiny-text-tables.pdf` / `use-of-force-raw.pdf`: pure dense text, about
  107k chars and no rects/lines. This is a bad fit for one-off spatial masks
  unless arrays already exist; array construction can dominate.
- `Atlanta_Public_Schools_GA_sample.pdf`: section-heavy, repeated selectors and
  exclusions. This is a good fit for derived array reuse around rectangular
  exclusion/region checks.
- `pak-ks-expenses.pdf`: dense rect/table page, about 4k chars and 561 rects.
  It broadens materialization/table coverage beyond the original PDFs.
- `sample-bop-policy-restaurant.pdf`: line-heavy policy page. It checks that
  vector/region changes do not only work on table PDFs.

Additional candidates:

| Candidate | What changed | Confirmation result | Conclusion |
|---|---|---:|---|
| `region_overlap_memoized_arrays` | Memoizes derived arrays for identical element lists; does not cache query results | Atlanta +29.4%, `micro:region-navigation:m27` +20.0%, `warm:repeated-page:m27` +15.4%, `real:01-practice` -0.2% | Better than rebuilding arrays every overlap/exclusion call. Production version needs a page element-version invalidation scheme. |
| `simple_selector_page_store` | Reuses page arrays for simple page-level selectors | Small wins on 01-practice/Atlanta, but regressions on repeated m27 and pak-ks selectors | Still not worth it. Most repeated selector time is in complex selectors like aggregate size or text contains; simple array predicates do not move enough work. |
| fixed `spatial_char_filter_ephemeral` | Early-return when there is no target/exclusion filtering | Removed the huge tiny-text array-build regression; remaining timings are noisy/no-op | Keep the early-return lesson, but do not pursue one-off char masks as a production track. |
| expanded `lazy_text_elements_upper_bound` | Added `pak-ks`, policy-lines, tiny-text materialization cases | Confirmed broad wins: 01-practice +27.6%, guides +22.3%, multipage +33.3%, pak-ks +15.2%, policy +17.8%, m27 materialize +57.1%, tiny-text materialize +28.6% | Strongest production candidate by far. |

Why the noisy/regressive candidates behaved that way:

- `simple_selector_fast_path` regressed because it rebuilt arrays many times and
  often fell back anyway. `simple_selector_page_store` fixed most build overhead,
  but still did not help enough because the common repeated selectors are not
  simple boolean/geometry predicates.
- `spatial_char_filter_ephemeral` originally regressed because it vectorized a
  path that often had nothing to filter. After the early return, it mostly
  disappears from timing, which means this is not the limiting problem.
- `table_word_assignment` did not hit because the representative guide workflows
  are not currently using `cell_extract="words"`. This should be retested only
  after a direct guide-word table path exists.
- `decorations_page_store` is plausible for line/rect-heavy pages, but lazy char
  materialization removes a larger and more universal cost first.

Updated recommendation:

1. Implement lazy native char `TextElement` materialization first.
2. Then implement a versioned derived-array layer for rectangular
   region/exclusion filtering, focused on selector-heavy workflows.
3. Revisit decoration vectorization after lazy chars, because its cost/benefit
   changes once char wrappers are no longer eagerly created.
4. Do not prioritize simple selector vectorization, one-off spatial char masks,
   or guide table word assignment from the current evidence.

## Lazy Char Implementation Comparison 2026-05-12

I split lazy native char materialization into four monkey-patch candidates:

| Candidate | Implementation shape | Production read |
|---|---|---|
| `lazy_chars_minimal_field` | Prepared native char dicts live on `ElementManager`; `store["chars"]` stays empty until char APIs materialize wrappers. | Smallest patch, but production lifecycle behavior is less explicit. |
| `lazy_chars_proxy` | A `LazyCharElementList` proxy stands in for `store["chars"]`; `len()` is cheap, iteration/indexing materializes wrappers. | Compatibility experiment only. `ElementStore.replace()` list-copies collections, so a real proxy design would need explicit store support. |
| `lazy_chars_raw_store` | Prepared char dicts live in a separate manager raw store; the element store is treated as the materialized view. | Best production shape: explicit backing data, normal char APIs, clear invalidation/mutation boundary. |
| `lazy_chars_columnar_store` | Builds a structure-of-arrays view over raw char dicts during page population, while still deferring wrappers. | Useful for later vector work, but too much upfront tax for this patch. |

Confirmation run:
`experiments/performance/results/lazy_chars/20260512T085855Z-matrix-summary.json`.
Settings: one warmup, five measured iterations, no profiler/tracemalloc.

Wall-time median improvement versus same-run baseline:

| Case | Minimal field | Proxy | Raw store | Columnar store |
|---|---:|---:|---:|---:|
| `real:01-practice` | +23.0% | +22.6% | +22.0% | +20.8% |
| `real:guides-expenses-sample` | +21.7% | +20.6% | +20.5% | +17.6% |
| `real:0500000US42001` | +0.2% | +10.1% | +9.0% | -3.9% |
| `real:multipage-table` | +16.4% | +21.6% | +21.6% | +16.9% |
| `real:pak-ks-expenses` | +11.1% | +10.7% | +9.2% | +9.6% |
| `real:policy-lines` | +13.1% | +14.9% | +35.0% | +19.3% |
| `micro:page-materialize:m27` | +46.4% | +48.8% | +47.1% | +46.1% |
| `micro:page-materialize:tiny-text` | +30.3% | +32.1% | +31.3% | +24.7% |
| `micro:page-materialize:pak-ks-expenses` | +11.1% | +10.5% | +9.1% | +6.8% |
| `warm:repeated-page:m27` | +20.6% | +15.5% | +16.2% | +17.6% |

Char-touch cases show the deferred cost:

| Case | Minimal field | Proxy | Raw store | Columnar store |
|---|---:|---:|---:|---:|
| `micro:page-chars:m27` | -0.5% | -1.2% | -0.4% | -4.4% |
| `micro:page-chars:tiny-text` | +0.2% | +0.6% | +0.9% | -6.4% |
| `micro:find-all-char:m27` | -24.8% | -24.2% | -25.2% | -5.4% |

Vector metrics explain the tradeoff:

- Deferring chars avoids wrapping 3,850 native chars on `m27` and 107,001 chars
  on `tiny-text-tables` unless char APIs are requested.
- On-demand wrapper materialization costs about 59-65ms median on `m27` and
  about 1.9s median on `tiny-text-tables`; this is essentially the cost the
  normal workflows stop paying up front.
- The columnar store builds arrays eagerly: about 7ms and 354KB on `m27`, but
  about 389-396ms and 9.8MB on `tiny-text-tables`. That explains its weaker
  tiny-text result and makes it a poor fit for the first lazy-char patch.
- `find_all("char")` regresses because the conservative prototype materializes
  chars during selector execution and then mutates the store with
  `ElementStore.set("chars", ...)`. Production can likely reduce that penalty by
  treating materialization as a view fill rather than a logical element mutation,
  but that should be a targeted implementation detail, not a reason to change
  `ElementStore.replace()` globally.

Correctness checks under all four patches:

- `tests/test_core/test_text_layer.py`
- `tests/test_words_vs_find_all_text.py`
- `tests/test_directional_boundary_precision.py`
- `tests/test_strikethrough_detection.py`
- `tests/test_highlight_detection.py`

All four candidates passed this 12-test targeted subset. `uv run pytest` could
not be used because the local `uv.toml` contains `exclude-newer = "1 week"`,
which this uv version rejects as a date; tests were run with `python -m pytest`.

Recommendation:

1. Implement the raw-store lazy char design first.
2. Do not change `ElementStore.replace()` for this work; keep its list-copy
   behavior as the default safety contract.
3. If production needs a proxy or non-copy lazy collection later, add an explicit
   narrow store method such as `set_lazy("chars", collection)` with invalidation
   tests instead of changing `replace()` semantics.
4. Keep columnar/page-array storage separate from lazy char materialization.
   Build arrays lazily only when a vector path can reuse them enough to pay for
   the upfront memory and construction cost.
