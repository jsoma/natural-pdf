# Natural PDF Performance Experiments

This directory is the developer-only workspace for performance research. Use it
to compare monkey-patch prototypes before changing production code.

The goal is measurement first:

1. Define the user-visible path or expensive phase.
2. Build one or more isolated patch candidates.
3. Run baseline and patched cases with the same harness settings.
4. Check correctness before trusting a timing win.
5. Update `experiments/performance/synthesis.md` with findings and next steps.

Do not treat cache wins as structural wins unless the target workflow is
actually repeated work. Prefer cold single-use improvements when choosing
production changes.

## Directory Layout

`scripts/perf_natural_pdf.py` is the benchmark harness. It discovers workloads,
loads optional patch modules, records timing/counters/environment data, and writes
reports.

`experiments/performance/run_matrix.py` runs known candidates as
baseline-vs-patch pairs.

`experiments/performance/patches/<track>/<candidate>/` contains experiment
patches. Each candidate should have:

- `patch.py`: monkey-patch implementation.
- `README.md`: hypothesis, implementation notes, cases, correctness checks, and
  current result summary.

`experiments/performance/results/` contains generated run artifacts. Results are
useful locally, but do not assume they should be committed.

`experiments/performance/synthesis.md` is the rolling decision document. Keep it
high signal: what was tested, what won, what regressed, and what should happen
next.

`experiments/performance/vector_metrics.py` provides optional experiment metrics
for prototypes that build arrays or masks.

## Patch Contract

A patch module must define `install()`. It may also define `METADATA`.

```python
from contextlib import contextmanager

METADATA = {
    "track": "example_track",
    "candidate": "example_candidate",
    "cache_only": False,
    "hypothesis": "Short claim this patch tests.",
}


@contextmanager
def install():
    original = SomeClass.some_method

    def replacement(self, *args, **kwargs):
        return original(self, *args, **kwargs)

    SomeClass.some_method = replacement
    try:
        yield
    finally:
        SomeClass.some_method = original
```

Rules for patches:

- Keep patches experiment-only. Do not change public `natural_pdf` APIs.
- Restore every monkey-patched attribute in `finally`.
- Make unsupported cases fall back to current behavior unless the candidate is
  intentionally an upper-bound prototype.
- Label cache-only candidates in `METADATA` and in the candidate README.
- Avoid network, OCR, AI, VLM, semantic search, and model-loading paths.
- Keep patch directories small and local to one idea.

For no-op plumbing, see
`experiments/performance/patches/noop/baseline_control/patch.py`.

## Adding A Candidate

1. Pick a track and candidate name:

   `experiments/performance/patches/<track>/<candidate>/`

2. Add `README.md` with this shape:

   ```markdown
   # Candidate Name

   Track: `<track>`

   Hypothesis:
   What should get faster and why.

   Implementation:
   What is monkey-patched, what falls back, and what semantics are preserved.

   Cases:
   Which workloads should improve, which should not regress, and why.

   Correctness:
   Targeted tests or manual checks required before trusting timings.

   Results:
   Fill in after screening and confirmation.
   ```

3. Add `patch.py` following the patch contract.

4. Add the candidate to `CANDIDATES` in
   `experiments/performance/run_matrix.py` if it should be part of repeatable
   matrix runs.

5. Run a no-op or dry run first if the patch shape is complex.

## Choosing Cases

List available harness cases:

```bash
python scripts/perf_natural_pdf.py --list-cases
```

Choose cases based on what the patch claims:

- Common cold workflows: `real:01-practice`, `real:guides-expenses-sample`,
  `real:multipage-table`, `real:0500000US42001`,
  `real:Atlanta_Public_Schools_GA_sample`.
- Phase workloads: `micro:page-materialize:m27`,
  `micro:guide-table:01-practice`, `micro:region-navigation:m27`.
- Repeated workloads: `micro:repeated-selectors:atlanta`,
  `warm:repeated-page:m27`, `warm:repeated-page:atlanta`.
- Pathological workloads: `real:tiny-text-tables`, `micro:tiny-text-layout`.
  Run these separately from normal varied cases.
- Rendering workloads: pass `--include-render` only when studying render paths.
- `.to_llm()` export workloads: pass `--include-to-llm`; this must not call a
  model.

If no existing workload exercises the path, add a small focused workload in
`discover_workloads()` in `scripts/perf_natural_pdf.py`. Keep the workload
non-AI, non-OCR, deterministic, and explicit about cold vs warm behavior.

## Running Experiments

The commands below use `python` directly. In a clean repo environment, `uv run
python ...` is also fine.

Screening is fast and rejects bad ideas:

```bash
python scripts/perf_natural_pdf.py \
  --output experiments/performance/results/<run-id>-baseline \
  --experiment-label screening-<track>-<candidate>-baseline \
  --cases <comma-separated-cases> \
  --iterations 1 --warmups 0 --no-tracemalloc --profile-slowest 0

python scripts/perf_natural_pdf.py \
  --output experiments/performance/results/<run-id>-patched \
  --experiment-label screening-<track>-<candidate>-patched \
  --patch experiments/performance/patches/<track>/<candidate>/patch.py \
  --cases <comma-separated-cases> \
  --iterations 1 --warmups 0 --no-tracemalloc --profile-slowest 0
```

Confirmation uses stable timing settings for plausible winners:

```bash
python scripts/perf_natural_pdf.py \
  --output experiments/performance/results/<run-id>-confirmed-baseline \
  --experiment-label confirmation-<track>-<candidate>-baseline \
  --cases <comma-separated-cases> \
  --iterations 5 --warmups 1 --no-tracemalloc --profile-slowest 0

python scripts/perf_natural_pdf.py \
  --output experiments/performance/results/<run-id>-confirmed-patched \
  --experiment-label confirmation-<track>-<candidate>-patched \
  --patch experiments/performance/patches/<track>/<candidate>/patch.py \
  --cases <comma-separated-cases> \
  --iterations 5 --warmups 1 --no-tracemalloc --profile-slowest 0
```

Profile finalists separately because profiling changes timings:

```bash
python scripts/perf_natural_pdf.py \
  --output experiments/performance/results/<run-id>-profile \
  --experiment-label profile-<track>-<candidate> \
  --patch experiments/performance/patches/<track>/<candidate>/patch.py \
  --cases <slow-or-interesting-cases> \
  --iterations 1 --warmups 0 --profile-slowest 2 --profile-top-n 20
```

Run the predefined matrix:

```bash
python experiments/performance/run_matrix.py --list-candidates
python experiments/performance/run_matrix.py --mode screening --track <track>
python experiments/performance/run_matrix.py --mode confirmation --track <track> --candidate <candidate>
python experiments/performance/run_matrix.py --mode profile --track <track> --candidate <candidate>
```

Use `--dry-run` on `run_matrix.py` to inspect commands without executing them.

## Output Artifacts

Every harness run writes:

- `baseline.json`: environment, git SHA, dirty status, patch metadata, case
  metrics, operation counters, vector metrics, and optional pytest durations.
- `profile-summary.md`: cProfile summaries by cumulative time, self time, and
  call count for profiled cases.
- `improvement-menu.md`: generated candidate menu from that run.
- Raw `.prof` files when profiling is enabled.

`run_matrix.py` also writes `<run-id>-matrix-summary.json` with baseline and
patched command statuses.

Key fields in `baseline.json`:

- `experiment_label`
- `git`
- `patches`
- `cases[].timing.wall_ms`
- `cases[].timing.cpu_ms`
- `cases[].tracemalloc_peak_kib`
- `cases[].rss_peak_kib`
- `cases[].operation_counts`
- `cases[].vector_metrics`

## Interpreting Results

Use medians for decisions. Use min and max to spot noise.

Classify a candidate as:

- Structural common-path win: improves cold real workflows without relying on
  repeated calls.
- Repeated-use win: helps warm or repeated cases only.
- Pathology win: helps tiny text, rendering, or another special case without
  normal-case regression.
- Reject: correctness failure, small-PDF regression over about 5 percent, memory
  growth over about 15 percent without a clear offsetting win, or behavior
  changes.

If a result is noisy:

- Re-run confirmation.
- Isolate the slowest case.
- Run baseline and patched outputs close together in time.
- Disable profiling and tracemalloc for timing comparisons.
- Check whether PDF opening, page materialization, or one expensive optional path
  is dominating the claimed win.

Cache candidates should be marked as amortized repeated-work improvements unless
they also improve cold single-use runs.

## Correctness Checks

Run targeted tests before calling a candidate viable. Pick the smallest suite
that covers the touched behavior.

Common targeted suites:

```bash
python -m pytest -q tests/test_selector_expressions.py tests/test_words_vs_find_all_text.py
python -m pytest -q tests/test_multipage_directional.py
python -m pytest -q tests/test_guides_partial.py tests/test_guides_extract_table.py tests/test_tables_integration.py
python -m pytest -q tests/test_tiny_text_tables.py tests/test_tiny_text_tables_table.py
python -m pytest -q tests/test_strikethrough_detection.py tests/test_highlight_detection.py
python -m pytest -q tests/test_crop_utils.py tests/test_render_direct_crop.py
```

For a production change, run the broader affected suite and record the exact
command and result in `synthesis.md`.

## Updating The Synthesis

Add a short entry to `experiments/performance/synthesis.md` after each completed
experiment group:

```markdown
## <Track Or Date>

Question:
What decision this experiment was meant to answer.

Candidates:
- `<track>/<candidate>`: short implementation summary.

Benchmarks:
| Candidate | Cases | Median result | Memory | Correctness | Decision |
| --- | --- | ---: | ---: | --- | --- |

Findings:
What improved, what regressed, and why.

Recommendation:
Production path, follow-up experiment, or rejection.
```

Also update each candidate README with result links and any important caveats.

## Clean Session Handoff

To start a fresh session on a new performance idea, point it here and give a
specific question. A useful prompt shape is:

```text
Read experiments/performance/README.md and experiments/performance/synthesis.md.
Create or extend an experiment for <specific path>. Add at least two candidate
patches if there are multiple plausible approaches. Run screening, targeted
correctness tests, and confirmation for plausible winners. Update the candidate
READMEs and synthesis with results, tradeoffs, and a production recommendation.
Do not change production code unless the result is very obvious and requested.
```

The session should first inspect local implementation and existing candidates,
then propose or create patches, then benchmark. It should not jump straight to a
production rewrite.
