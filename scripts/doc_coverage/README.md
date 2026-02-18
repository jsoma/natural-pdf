# Documentation Coverage Tool

Measures which parts of the natural-pdf library's public API are demonstrated in documentation code samples.

## How It Works

1. **Builds an API catalog** by introspecting the `natural_pdf` module for public classes and methods
2. **Extracts code samples** from Markdown (`.md`) and Jupyter notebooks (`.ipynb`) in the docs folder
3. **Analyzes** which API methods appear in the code samples
4. **Reports** coverage statistics and identifies gaps

## Quick Start

```bash
# Recommended: Quick check with gap analysis
python -m scripts.doc_coverage --exclude-internal --deduplicate --gaps

# CI integration (fails if coverage drops below threshold)
python -m scripts.doc_coverage --exclude-internal --deduplicate --fail-under 20 --quiet

# Full HTML report for detailed review
python -m scripts.doc_coverage --exclude-internal --deduplicate --html coverage.html
```

## Key Options

| Option | Description | When to Use |
|--------|-------------|-------------|
| `--exclude-internal` | Exclude internal methods (`get_*`, `set_*`, `is_*`, `has_*`, etc.) | Always recommended - focuses on user-facing API |
| `--deduplicate` | Count unique method names instead of `Class.method` pairs | Recommended - avoids counting same method on multiple classes |
| `--gaps` | Show gap analysis with priority documentation needs | When planning what to document next |
| `--fail-under N` | Exit with error if coverage < N% | CI pipelines |
| `--quiet` | Suppress terminal output | CI pipelines |
| `--html FILE` | Generate interactive HTML report | Deep dive into coverage details |
| `--show-all-uncovered` | List all uncovered methods | Finding specific missing methods |

## Example Output

### Standard Report
```
Building API catalog for natural_pdf...
Scanning docs/ for code samples...

=== DOCUMENTATION COVERAGE REPORT ===
Coverage: 44/216 unique methods (20.4%)

Covered methods: 44
  PDF.ask, PDF.close, PDF.extract_text, Page.analyze_layout,
  Page.apply_ocr, Page.ask, Page.create_region, Page.find, ...

Uncovered methods: 172
  Flow.add_span, Flow.adjust_reading_order, Flow.break_at,
  FlowRegion.above, FlowRegion.analyze_layout, ...
```

### Gap Analysis (`--gaps`)
```
============================================================
GAP ANALYSIS: Priority Documentation Needs
============================================================

[High Priority] Common methods with NO examples:
  • export (6 classes): PDF, Page, Region +3 more
  • highlight (6 classes): PDF, Page, Region +3 more
  • inspect (6 classes): PDF, Page, Region +3 more
  • render (6 classes): PDF, Page, Region +3 more

[Per-Class Coverage]
  ✗ DocumentQA: 0/3 (0%)
      Missing: ask, ask_batch, ask_pages +0 more
  ✗ Flow: 0/19 (0%)
      Missing: add_span, adjust_reading_order, break_at +16 more
  ○ Page: 21/84 (25%)
  ...
```

### HTML Report (`--html`)

Generates an interactive HTML file with:
- Sortable tables of covered/uncovered methods
- Per-class breakdowns
- Search and filter capabilities

## Understanding the Metrics

- **Coverage %**: Percentage of unique public methods that appear in at least one code sample
- **Covered methods**: Methods found in documentation code blocks
- **Uncovered methods**: Methods in the public API with no documentation examples

### Per-Class Coverage Symbols
- ✓ (50%+) - Well documented
- ○ (20-49%) - Partially documented
- ✗ (<20%) - Needs attention

## CI Integration

Add to your CI workflow to prevent documentation drift:

```yaml
- name: Check documentation coverage
  run: |
    python -m scripts.doc_coverage \
      --exclude-internal \
      --deduplicate \
      --fail-under 20 \
      --quiet
```

## Workflow for Improving Coverage

1. **Identify gaps**: Run with `--gaps` to see priority areas
2. **Write documentation**: Add code examples for undocumented methods
3. **Verify improvement**: Re-run the tool to confirm coverage increased
4. **Commit**: Include coverage delta in commit message

```bash
# Before: 20.4% (44/216)
python -m scripts.doc_coverage --exclude-internal --deduplicate

# ... add documentation ...

# After: 25.0% (54/216)
python -m scripts.doc_coverage --exclude-internal --deduplicate
git commit -m "docs: add batch processing examples (+10 methods)"
```

## All Options

```
Usage: python -m scripts.doc_coverage [OPTIONS]

Options:
  --docs PATH                     Documentation directory [default: docs/]
  --module TEXT                   Python module to analyze [default: natural_pdf]
  --output PATH                   JSON output file [default: doc_coverage.json]
  --html PATH                     HTML report output path
  --fail-under FLOAT              Minimum coverage threshold
  --include-inherited/--no-include-inherited
                                  Include inherited methods [default: no]
  --exclude-internal/--no-exclude-internal
                                  Exclude internal methods [default: no]
  --include-submodules/--no-include-submodules
                                  Include submodule classes [default: yes]
  --deduplicate/--no-deduplicate  Count unique method names [default: no]
  --show-all-uncovered            Show all uncovered methods
  --gaps                          Show gap analysis
  --quiet                         Suppress terminal output
  --help                          Show this message and exit
```
