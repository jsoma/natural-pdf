# Natural PDF Performance Experiments

This directory holds developer-only monkey-patch experiments for comparing
performance ideas before changing production code.

Run patches through the perf harness:

```bash
uv run python scripts/perf_natural_pdf.py \
  --output experiments/performance/results/<run-id> \
  --experiment-label <label> \
  --patch experiments/performance/patches/<track>/<candidate>/patch.py \
  --cases <comma-separated cases> \
  --iterations 1 --warmups 0 --no-tracemalloc --profile-slowest 0
```

Use screening runs first, then confirmation runs with one warmup and five
measured iterations. Keep tiny-text runs separate from varied baseline runs.

To run the predefined baseline-vs-patch matrix:

```bash
uv run python experiments/performance/run_matrix.py --mode screening
uv run python experiments/performance/run_matrix.py --mode confirmation --track selectors
uv run python experiments/performance/run_matrix.py --list-candidates
```
