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
| selector shortcuts | `selector_shortcuts/find_contains_first` | structural | Can simple `find("text:contains(...)")` return the first match without full collection allocation? |
| selector shortcuts | `selector_shortcuts/find_aggregate_first` | structural | Can simple aggregate anchors like `text[size=max()]` return the first winner in one scan? |
| selector shortcuts | `selector_shortcuts/find_combined_first` | structural | Does combining contains and aggregate first-match shortcuts help real workflows? |
| region first | `region_first/typed_pool` | structural | Can `Region.find_all` run selectors against a region-prefiltered typed pool? |
| region first | `region_first/word_pool_only` | structural | Is a narrower text-only region-first path safer and faster? |
| region first | `region_first/no_exclusion_typed_pool` | structural | Does a no-exclusion-only region-first fast path avoid the Atlanta exclusion trap? |
| wildcard shortcuts | `wildcard_shortcuts/no_chars_upper_bound` | unsafe upper bound | How much broad any-element APIs pay by materializing chars they may not need? |
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

## Shortcut Experiment Addendum 2026-05-12

I added three more families of one-shot shortcut experiments because repeated
page iteration and exclusions are probably not the common case:

- `selector_shortcuts/*`: early-return variants for common `find(...)` anchors.
- `region_first/*`: variants that apply region geometry before selector
  execution.
- `wildcard_shortcuts/no_chars_upper_bound`: an unsafe upper bound for broad
  any-element paths that should not need native chars.

Screening runs:

- `experiments/performance/results/shortcuts/20260512T213751Z-matrix-summary.json`
- `experiments/performance/results/shortcuts/20260512T213825Z-*`

All seven new candidates were screened. `find_aggregate_first` was neutral
outside `real:policy-lines`, so it was only carried forward as part of
`find_combined_first`.

Confirmation runs:

- `experiments/performance/results/shortcuts_confirm_contains/20260512T213943Z-matrix-summary.json`
- `experiments/performance/results/shortcuts_confirm_combined/20260512T213943Z-matrix-summary.json`
- `experiments/performance/results/shortcuts_confirm_region_no_excl/20260512T213943Z-matrix-summary.json`
- `experiments/performance/results/shortcuts_confirm_wildcard/20260512T213943Z-matrix-summary.json`

Confirmation results, median wall time:

| Candidate | Strongest wins | Regressions / limits | Read |
|---|---:|---:|---|
| `selector_shortcuts/find_contains_first` | `real:policy-lines` +18.8%, `micro:find-anchors:01-practice` +12.6%, `real:01-practice` +3.5% | `micro:region-navigation:m27` -13.7%, `real:0500000US42001` -5.1%, `real:guides-expenses-sample` -4.1% | Too volatile as a standalone patch. The fast path hits, but fallback/check overhead can dominate mixed workflows. |
| `selector_shortcuts/find_combined_first` | `real:policy-lines` +5.2%, `micro:find-anchors:atlanta` +4.2%, `real:0500000US42001` +2.7%, `real:01-practice` +1.4% | `micro:region-navigation:m27` -1.0%; Atlanta neutral | Plausible low-risk cleanup, but not high leverage. It is a small cold-path improvement, not the next major optimization. |
| `region_first/no_exclusion_typed_pool` | `real:01-practice` +6.7%, `micro:region-navigation:m27` +1.1% | `real:hebrew-table` -6.8%, `real:Atlanta_Public_Schools_GA_sample` -4.1%, `real:policy-lines` -2.9% | Do not promote broadly. The no-exclusion guard is necessary, but region-prefiltering still adds overhead on many normal pages. |
| `wildcard_shortcuts/no_chars_upper_bound` | `micro:find-all-any:m27` +49.4%, `micro:get-elements:m27` +43.9%, `micro:get-elements:tiny-text` +32.6%, `real:policy-lines` +7.0% | `real:01-practice` -9.9%; unsafe semantics because chars disappear from broad pools | Useful design signal only. Add an explicit internal layout-element pool or `include_chars=False` path if broad APIs show up in real workflows. Do not change `get_all_elements()` semantics. |

Instrumentation signals:

- `find_combined_first` hit 80 contains fast paths and 30 aggregate fast paths
  across the confirmation matrix, but also had 280 exclusion fallbacks. That
  explains why it helps simple cold anchors but barely moves Atlanta.
- `region_first/no_exclusion_typed_pool` had 455 exclusion fallbacks and only 70
  fast-path region queries. The broader `typed_pool` and `word_pool_only`
  variants were rejected in screening because Atlanta regressed by about 50-56%
  when callable exclusions were evaluated before selector narrowing.
- `wildcard_shortcuts/no_chars_upper_bound` skipped 573,505 raw chars across the
  confirmation run. That is the reason the broad all-element microcases improved
  so much, even after production lazy char materialization.

Correctness checks:

`selector_shortcuts/find_combined_first` and
`region_first/no_exclusion_typed_pool` both passed this targeted subset under the
patch:

- `tests/test_selector_expressions.py`
- `tests/test_words_vs_find_all_text.py`
- `tests/test_multipage_directional.py`
- `tests/test_directional_boundary_precision.py`
- `tests/test_guides_partial.py`

`27 passed` for each patch. The wildcard patch was not correctness-qualified
because it intentionally changes wildcard/all-element semantics.

Recommendation:

1. Do not make region-first selector execution the next production task. The
   experiments show that it is easy to move work earlier and make exclusion-heavy
   flows worse.
2. Keep `find_combined_first` as a small, testable follow-up only after higher
   leverage work. It is likely safe, but the median wins are low single digits.
3. Treat wildcard/no-chars as the most interesting shortcut signal. The
   production version should be explicit, such as an internal
   `get_layout_elements(include_chars=False)` or selector pool used by layout,
   dissolve, and rendering helpers that do not need chars. It should not alter
   public `page.get_elements()` or `find_all("*")` behavior.
4. The next real optimization should still be structural and common-path:
   either vector/line guide detection if it is not already productionized, or an
   explicit no-char internal pool for broad layout operations if profiling shows
   those paths in user workflows.

## Internal No-Char Pool Implementation 2026-05-12

Follow-up profiling found a concrete safe use case for the wildcard/no-char
signal: describe/inspect paths. Those paths already discard character elements
as too granular, but `page.inspect()` and `page.describe()` were still reaching
wildcard/all-element pools that materialized native char `TextElement`s first.

Before implementation, a focused `cProfile` run for `page.inspect()` on
`m27.pdf` showed:

- `page.inspect()` total: about 96ms after page object creation.
- `SelectorService.find_all("*")` / selector pool resolution: about 84ms.
- `ElementManager.get_all_elements()` and `_materialize_chars()`: about 75ms.

Local timing before the patch, after page object creation:

| Case | Before | After | Read |
|---|---:|---:|---|
| `page.describe()` on `m27.pdf` | 70.0ms | 1.6ms | Removed char wrapper construction from describe body. |
| `page.inspect()` on `m27.pdf` | 80.9ms | 1.8ms | Same. |
| `page.describe()` on `tiny-text-tables.pdf` | 2378.2ms | 47.0ms | Avoids constructing 107k char wrappers only to ignore them. |

Production implementation:

- `ElementManager.get_all_elements(include_chars=True)` keeps existing default
  behavior.
- `Page._get_elements(include_chars=False)` and `Region._get_elements(...)`
  provide internal opt-in no-char pools.
- Page, region, PDF, page collection, and PDF collection describe/inspect paths
  use the internal no-char pool.
- Public broad APIs remain unchanged: `page.get_elements()`, `page.find_all("*")`,
  `page.chars`, and `page.find_all("char")` still include/materialize chars.

Focused harness run after implementation:
`/tmp/npdf-no-char-internal/baseline.json`.

Median wall times:

| Case | Median |
|---|---:|
| `micro:describe:m27` | 112.4ms |
| `micro:inspect:m27` | 111.7ms |
| `micro:describe:tiny-text` | 4144.5ms |
| `micro:get-elements:m27` | 197.9ms |
| `micro:find-all-any:m27` | 204.8ms |

The harness includes PDF open and page materialization, so tiny-text remains
dominated by raw page loading. The describe/inspect body no longer materializes
chars; the public broad APIs still do, as intended.

Validation:

- `tests/test_lazy_char_materialization.py`
- `tests/test_words_vs_find_all_text.py`
- `tests/test_attr_method.py`
- `tests/test_trim_sparse_content.py`
- `tests/realpdf/test_element_manager_compat.py`

Result: `53 passed`.

## Guide Line Detection Experiments

Question: optimize both guide line paths without guessing. The vector path
should avoid repeated Python work if that matters; the pixel path should avoid
unnecessary render/detection work.

New isolated workloads:

- `micro:guide-lines-vector-both:01-practice`
- `micro:guide-lines-vector-both:policy-lines`
- `micro:guide-lines-pixels-both:01-practice`
- `micro:guide-lines-pixels-horizontal:01-practice`
- `micro:guide-lines-pixels-vertical:01-practice`

Candidates tested:

| Candidate | Result | Recommendation |
|---|---:|---|
| `guide_lines/direct_both_axis` | `micro:guide-lines-pixels-both:01-practice` 383.3ms -> 227.4ms (+40.7%); vector both-axis +1.7-2.3%; `micro:guide-table:01-practice` +2.6%; `real:guides-expenses-sample` +1.1% | Best production candidate. Structural one-pass fix, not cache-first. |
| `guide_lines/pixel_both_axis_detect_cache` | Pixel both-axis 391.7ms -> 208.8ms (+46.7%); no guide-table benefit | Confirms duplicate pixel detection is the cost, but do not implement as a cache. |
| `guide_lines/pixel_axis_specific` | Single-axis pixel detection +5.5-6.2%; guide-table and guides-expenses neutral | Viable secondary cleanup if we want single-axis calls to only add requested-orientation detected lines. |
| `guide_lines/vector_numpy_classify` | Vector cases neutral/regressive; changed guide counts on policy-lines | Reject. Array setup and semantic drift are not worth it. |
| `guide_lines/vector_single_pass_cache` | Vector cases neutral/slightly regressive | Reject. Re-collecting vector lines is not the bottleneck. |

Interpretation:

- Pixel detection is the real guide-line opportunity. Current both-axis guide
  construction can run the render/detect path once per axis. Removing that
  duplicate pass is a clear cold single-use win.
- The vector path is already cheap enough that NumPy classification and
  collection caching do not pay for themselves on the measured PDFs.
- The direct both-axis prototype preserves guide counts after matching current
  vector line-selection semantics. For pixel detection, it changes page side
  effects in a likely-better way: one pass with both `max_lines_h` and
  `max_lines_v` leaves detected line elements for both orientations, while the
  current two-call `replace=True` sequence leaves only the final call's source
  lines. Production work should codify the desired side effect with a test.

Validation:

- `guide_lines/direct_both_axis` under
  `tests/test_guides.py`, `tests/test_guides_integration.py`,
  `tests/test_guides_generation_dedup.py`,
  `tests/test_guides_extract_table.py`, and `tests/test_guides_partial.py`:
  `57 passed`.
- `guide_lines/pixel_axis_specific` under the same suite: `57 passed`.

Production follow-up implemented:

- Added a provider-level `detect_both()` path with a default two-call fallback.
- Implemented `LinesGuidesEngine.detect_both()` so line guide generation shares
  vector line collection or pixel line detection across both axes.
- Updated `Guides.from_lines(axis="both")` and `Guides.add_lines(axis="both")`
  to use the shared path.
- Added tests that pixel both-axis guide generation calls `detect_lines()` once
  and passes both `max_lines_h` and `max_lines_v`.

Post-production confirmation medians:

| Case | Median |
|---|---:|
| `micro:guide-lines-pixels-both:01-practice` | 214.9ms |
| `micro:guide-lines-vector-both:01-practice` | 37.5ms |
| `micro:guide-lines-vector-both:policy-lines` | 112.7ms |
| `micro:guide-table:01-practice` | 49.2ms |
| `real:guides-expenses-sample` | 257.3ms |

## Pixel Line Detection Experiments

Question: after eliminating duplicate both-axis guide detection, can the pixel
threshold/projection pipeline itself get faster?

Production cleanup implemented first:

- `_detect_lines_projection()` no longer renders a second copy of the page only
  to recover image dimensions. `_get_image_for_detection()` already rendered the
  processed image, so the projection path now derives dimensions from that image.
- Post-cleanup `micro:guide-lines-pixels-both:01-practice` render count dropped
  from median `2` to median `1`.
- Post-cleanup medians:

| Case | Median |
|---|---:|
| `micro:guide-lines-pixels-both:01-practice` | 192.6ms |
| `micro:guide-lines-pixels-horizontal:01-practice` | 191.3ms |
| `micro:guide-lines-pixels-vertical:01-practice` | 192.9ms |

Production uint8 profile pass implemented next:

- `_find_lines_on_image_data()` now keeps the existing adaptive/Otsu threshold
  and `find_peaks` behavior, but profiles the `uint8` binary mask directly with
  `np.count_nonzero()` instead of building normalized float copies for each
  axis.
- RGB grayscale conversion now uses an integer luminance approximation, avoiding
  an intermediate float dot-product array.
- Production medians after this pass:

| Case | Median |
|---|---:|
| `micro:guide-lines-pixels-both:01-practice` | 176.9ms |
| `micro:guide-lines-pixels-horizontal:01-practice` | 153.2ms |
| `micro:guide-lines-pixels-vertical:01-practice` | 150.6ms |
| `micro:guide-table:01-practice` | 48.1ms |
| `real:guides-expenses-sample` | 251.3ms |

Experiment candidates:

| Candidate | Confirmed result | Recommendation |
|---|---:|---|
| `pixel_detection/uint8_profile_pipeline` | Pixel both-axis +16.2%; horizontal +16.4%; vertical +0.4%; guide-table +7.4%; guides-expenses +9.1% | Promoted to production. It keeps the same broad algorithm and removes float/binary copies. |
| `pixel_detection/cv2_adaptive_pipeline` | Pixel both-axis +66.9%; horizontal +49.7%; vertical +52.0%; guides-expenses +3.6%; guide-table -16.6% | High-upside candidate, but needs broader PDF checks because threshold semantics differ and one mixed workflow regressed. |
| `pixel_detection/fast_topk_peaks` | Pixel both-axis +22.1%; horizontal +19.2%; vertical +40.2%; guide-table +10.7%; guides-expenses +1.6% | Useful comparator. Higher semantic risk because it changes peak selection. |
| `pixel_detection/resolution_150` | Explicit pixel workloads +21.8-38.5%; guide-table -20.5%; guides-expenses -1.5% | Keep as opt-in/tuning, not a default. |
| `pixel_detection/resolution_120` | Explicit pixel workloads +37.4-46.6%; guide-table -19.1%; guides-expenses -2.4% | Too aggressive for default; useful only as an explicit speed/quality tradeoff. |

Validation:

- Production no-duplicate-render path under
  `tests/test_perf_natural_pdf.py`, `tests/test_guides.py`,
  `tests/test_guides_integration.py`, `tests/test_guides_generation_dedup.py`,
  `tests/test_guides_extract_table.py`, and `tests/test_guides_partial.py`:
  `66 passed`.
- `uint8_profile_pipeline`, `cv2_adaptive_pipeline`, and `fast_topk_peaks`
  each passed targeted guide tests under monkey patch:
  `tests/test_guides.py::test_pixel_based_line_detection`,
  both pixel `detects_once` tests, `tests/test_guides_integration.py`,
  `tests/test_guides_extract_table.py`, and `tests/test_guides_partial.py`.
- Production uint8 path passed the same targeted guide group:
  `21 passed`.
- Coordinate spot-check against the old float-mask implementation matched for
  `01-practice.pdf`, `guides-expenses-sample.pdf`, and `multipage-table.pdf`
  across both-axis, horizontal-only, and vertical-only pixel guide calls with
  `max_lines=5`.

Interpretation:

- The live duplicate-render cleanup and uint8 profile pass are the safest pixel
  wins and should stay.
- OpenCV is probably the ceiling for this path, but it should get a larger PDF
  coverage pass before production because adaptive threshold behavior is not
  identical.
- Lowering resolution is a parameter/tuning story, not a structural default.

## Render Optimization Notes

Question: are there `.render()` optimization opportunities outside extraction?

Production cleanup implemented:

- `resolve_crop_bbox()` no longer calls `content_bbox_fn()` for `crop=False`
  or explicit `crop_bbox`.
- This removes accidental content-bbox calculation from plain `page.render()`.
  On dense pages, that calculation could materialize every native char element
  before doing a clean full-page render.

Measured effect:

| Case | Before | After | Read |
|---|---:|---:|---|
| Harness `optional:render:01-practice` total | 66.6ms | 42.1ms | Total includes PDF/page setup; render phase dropped from 28.4ms to 4.8ms. |
| `01-practice.pdf` page render, 150 DPI | 10.5ms | 6.7ms | Removes small accidental bbox work. |
| `m27.pdf` page render, 150 DPI | 18.2ms | 6.8ms | Removes dense element-pool work. |
| `guides-expenses-sample.pdf` page render, 150 DPI | 21.0ms | 6.9ms | Same. |
| `tiny-text-tables.pdf` page render, 150 DPI | 392.0ms | 72.8ms | Avoids constructing ~107k char wrappers only to render clean page image. |

Validation:

- `tests/test_crop_utils.py`, `tests/test_crop_enhancements.py`,
  `tests/test_region_show_crop_highlights.py`,
  `tests/test_element_show_crop_highlights.py`, `tests/test_highlight_offset.py`,
  `tests/test_from_images.py`, and `tests/test_extraction_text_and_vision.py`:
  `54 passed`.

Remaining render candidates:

| Candidate | Evidence | Recommendation |
|---|---:|---|
| Direct pypdfium full-page render | Very similar to current pdfplumber path: 01-practice 6.55ms vs 6.25ms, m27 6.81ms vs 7.97ms, tiny-text 73.67ms vs 69.60ms at 150 DPI | Not worth replacing broadly. Current pdfplumber path already uses pypdfium underneath and handles page details. |
| Direct pypdfium crop rendering | Cropped 150 DPI render improved 01-practice 6.42ms -> 1.52ms, m27 6.48ms -> 1.87ms, tiny-text 68.24ms -> 44.35ms, but output dimensions differed by 1-2px in spot checks | Good experiment candidate for region/crop-heavy workflows, but needs careful coordinate/rounding compatibility work before production. |
| Exact-width render when `width=` is provided | `page.render(width=300)` improved about 24.4ms -> 4.0ms on 01-practice and 98.7ms -> 53.9ms on tiny-text when rendering at exact width resolution | Potential opt-in or behavior cleanup. Current code renders at at least base DPI and downsamples for quality; changing default is a quality/performance contract decision. |
| Render cache | Prior experiment showed only small 2.8-8.5% wins | Keep diagnostic/cache-only. Avoid production cache unless a workflow demonstrably repeats identical renders and memory is bounded. |

Direct crop follow-up:

- Added experiment patch `rendering/direct_pdfium_crop`.
- Focused render/crop/highlight suite passed under the patch:
  `54 passed`.
- Full pytest suite passed under the patch after fixing an unrelated mocked-page
  compatibility issue in `Page._get_all_elements_raw()`:
  `1758 passed, 41 skipped, 1 xpassed`.
- Pixel diffs remain non-identical because direct pypdfium crop rounds the crop
  bitmap independently. The fastest production path would need explicit output
  size/rounding rules rather than assuming byte-for-byte identity with the
  current full-page-render-then-crop path.

## Optimization Pass Status And TODO

Accomplished:

- Built the developer perf harness and experiment matrix runner.
- Added baseline/profile/menu artifacts for non-AI, non-OCR workloads.
- Added experiment support for monkey-patch candidates and patch metadata.
- Implemented and productionized lazy native char `TextElement`
  materialization.
- Screened vectorized decoration, spatial filtering, selector, table, and lazy
  char candidates.
- Added one-shot shortcut experiments for first-match selectors, region-first
  selector execution, and broad no-char pools.
- Implemented internal no-char pools for describe/inspect while preserving
  public broad element API semantics.
- Normalized guide line detection defaults to `detection_method="auto"` and made
  auto routing explicit via warnings that state whether vector or pixel detection
  was selected.
- Tested five guide-line optimization candidates across isolated vector, pixel
  both-axis, pixel single-axis, and guide-table workloads.
- Productionized the direct both-axis guide line path so pixel line detection is
  not run once per axis.
- Removed the second render inside projection line detection and tested five
  pixel-threshold/projection alternatives.
- Removed accidental content-bbox/char materialization from plain uncropped
  renders.

Current production improvements:

- Word/text extraction workflows no longer eagerly pay native char wrapper cost
  unless char APIs are requested.
- Page/region/PDF/collection describe and inspect paths no longer materialize
  char wrappers that they immediately ignore.
- `Guides.from_lines(axis="both")` and `Guides.add_lines(axis="both")` share
  one provider call; the built-in line engine now runs one vector/pixel line
  detection pass for both axes.
- Pixel projection line detection no longer renders twice in one detection pass.
- Pixel projection profiling now avoids normalized float mask copies and profiles
  the `uint8` binary mask directly.
- Plain `page.render(crop=False)` no longer computes content bounds or
  materializes chars before rendering.

Do not prioritize:

- Broad region-first selector execution. It regressed Atlanta because it pushed
  exclusion and region filtering before selective selector matching.
- Query-result caching as a first production fix. It helps repeated workflows
  but does not address one-shot structural cost and creates invalidation
  complexity.
- One-off vector masks for char filtering. Array construction cost often eats
  the benefit.
- Simple selector vectorization in its current form. Most real selector cost is
  in text contains, aggregates, exclusions, and workflow shape rather than simple
  boolean predicates.

Worth checking next:

1. Decide whether to productionize single-axis pixel detection as
   requested-axis-only. The experiment showed +5.5-6.2% on explicit single-axis
   pixel line workloads and passed guide tests, but it changes page side effects
   from "add both orientations" to "add the requested orientation only."
2. Run a larger OpenCV thresholding coverage pass. It is the high-upside path,
   but threshold semantics differ from the current scipy/numpy implementation.
3. Prototype direct pypdfium crop rendering for `crop_bbox` / region render
   workloads, with explicit output-size and pixel-coordinate compatibility tests.
4. Decide whether `width=` should get an explicit fast/quality mode. Exact-width
   rendering is much faster for thumbnails, but lower quality than render-high
   then downsample.
5. Decoration detection after lazy chars: rerun decoration profiles now that char
   wrappers are lazy. If decoration annotation still dominates line/rect-heavy
   pages, test a narrower vectorized decoration implementation.
6. Tiny text table extraction: continue separately from normal workloads. The
   hard cost is still pdfplumber text-strategy/table behavior over very dense
   tiny glyphs.
7. Broad internal no-char pools beyond describe/inspect: inspect/profile
   remaining internal uses of `get_elements()` / `find_all("*")`, especially
   viewer overlays, section helpers, parent/next/prev navigation, and collection
   split helpers. Only convert paths that explicitly ignore chars.
8. `find_combined_first`: optional small cleanup for simple `find()` anchor
   lookups. It passed targeted tests but only produced low single-digit median
   workflow wins.

Recommended next implementation order:

1. Run broader coordinate/count validation if we want to consider the OpenCV
   threshold path; otherwise leave pixel detection at the current uint8 profile
   production implementation.
2. Prototype direct crop rendering if region/crop render workflows matter.
3. Re-profile decoration detection and decide whether to implement a targeted
   lazy/vector decoration path.
4. Re-enter tiny-text table extraction with correctness-preserving alternatives
   to pdfplumber text-strategy assignment.
5. Add more internal no-char call sites only when profiles show broad element
   pools are hot and the call site ignores chars.
