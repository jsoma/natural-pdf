"""Command-line interface for documentation coverage tool."""

import sys
from collections import Counter
from pathlib import Path

import click

from .analyzers import build_api_catalog, calculate_coverage
from .extractors import CodeSample, extract_from_markdown, extract_from_notebook
from .reporters import print_report, write_html, write_json


def _print_gap_analysis(
    catalog: dict[str, list[str]],
    samples: list[CodeSample],
    deduplicate: bool,
) -> None:
    """Print gap analysis showing priority documentation needs."""
    # Get non-deduplicated coverage to analyze per-class
    result = calculate_coverage(samples, catalog, deduplicate=False)

    # Count how many classes have each method
    method_class_count: Counter[str] = Counter()
    for cls, methods in catalog.items():
        for m in methods:
            method_class_count[m] += 1

    click.echo("\n" + "=" * 60)
    click.echo("GAP ANALYSIS: Priority Documentation Needs")
    click.echo("=" * 60)

    # Find common methods (on 3+ classes) that are undocumented
    click.echo("\n[High Priority] Common methods with NO examples:")
    common_undoc = []
    for method, class_count in method_class_count.most_common():
        if class_count >= 3:
            total_hits = sum(
                result.method_hits.get(f"{cls}.{method}", 0)
                for cls in catalog
                if method in catalog[cls]
            )
            if total_hits == 0:
                classes = [cls for cls in catalog if method in catalog[cls]]
                common_undoc.append((method, class_count, classes))

    if common_undoc:
        for method, count, classes in common_undoc[:10]:
            class_str = ", ".join(classes[:3])
            if len(classes) > 3:
                class_str += f" +{len(classes) - 3} more"
            click.echo(f"  • {method} ({count} classes): {class_str}")
        if len(common_undoc) > 10:
            click.echo(f"  ... and {len(common_undoc) - 10} more")
    else:
        click.echo("  None - all common methods have examples!")

    # Per-class coverage
    click.echo("\n[Per-Class Coverage]")
    for cls in sorted(catalog.keys()):
        methods = catalog[cls]
        covered = [m for m in methods if result.method_hits.get(f"{cls}.{m}", 0) > 0]
        uncovered = [m for m in methods if result.method_hits.get(f"{cls}.{m}", 0) == 0]
        pct = len(covered) / len(methods) * 100 if methods else 0

        # Color code by coverage
        if pct >= 50:
            status = "✓"
        elif pct >= 20:
            status = "○"
        else:
            status = "✗"

        click.echo(f"  {status} {cls}: {len(covered)}/{len(methods)} ({pct:.0f}%)")

        # Show top missing methods for low-coverage classes
        if pct < 30 and uncovered:
            missing = ", ".join(sorted(uncovered)[:5])
            if len(uncovered) > 5:
                missing += f" +{len(uncovered) - 5} more"
            click.echo(f"      Missing: {missing}")

    click.echo()


@click.command()
@click.option(
    "--docs",
    default="docs/",
    help="Documentation directory to scan for code samples.",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@click.option(
    "--module",
    default="natural_pdf",
    help="Python module to analyze for public API.",
)
@click.option(
    "--output",
    default="doc_coverage.json",
    help="JSON output file path.",
    type=click.Path(path_type=Path),
)
@click.option(
    "--html",
    default=None,
    help="Optional HTML report output path.",
    type=click.Path(path_type=Path),
)
@click.option(
    "--fail-under",
    default=0.0,
    type=float,
    help="Exit with error if coverage is below this percentage.",
)
@click.option(
    "--include-inherited/--no-include-inherited",
    default=False,
    help="Include inherited methods in API catalog.",
)
@click.option(
    "--exclude-internal/--no-exclude-internal",
    default=False,
    help="Exclude internal-looking methods (get_, set_, is_, has_, etc.).",
)
@click.option(
    "--include-submodules/--no-include-submodules",
    default=True,
    help="Include classes from submodules (TableResult, ElementCollection, etc.).",
)
@click.option(
    "--deduplicate/--no-deduplicate",
    default=False,
    help="Count unique method names instead of Class.method pairs (recommended).",
)
@click.option(
    "--show-all-uncovered",
    is_flag=True,
    default=False,
    help="Show all uncovered methods in terminal output.",
)
@click.option(
    "--gaps",
    is_flag=True,
    default=False,
    help="Show gap analysis: common undocumented methods and per-class coverage.",
)
@click.option(
    "--quiet",
    is_flag=True,
    default=False,
    help="Suppress terminal output (still writes JSON/HTML).",
)
def main(
    docs: Path,
    module: str,
    output: Path,
    html: Path | None,
    fail_under: float,
    include_inherited: bool,
    exclude_internal: bool,
    include_submodules: bool,
    deduplicate: bool,
    show_all_uncovered: bool,
    gaps: bool,
    quiet: bool,
) -> None:
    """Analyze documentation coverage for a Python library.

    Scans documentation files for Python code samples and measures
    which parts of the library's public API are demonstrated.
    """
    # 1. Build API catalog
    if not quiet:
        click.echo(f"Building API catalog for {module}...")

    try:
        catalog = build_api_catalog(
            module,
            include_inherited=include_inherited,
            include_submodules=include_submodules,
            exclude_internal=exclude_internal,
        )
    except ImportError as e:
        click.echo(f"Error: Could not import module '{module}': {e}", err=True)
        sys.exit(1)

    if not catalog:
        click.echo(f"Warning: No public classes found in {module}", err=True)

    # 2. Extract code samples
    if not quiet:
        click.echo(f"Scanning {docs} for code samples...")

    samples: list[CodeSample] = []

    for md_file in docs.rglob("*.md"):
        samples.extend(extract_from_markdown(md_file))

    for nb_file in docs.rglob("*.ipynb"):
        samples.extend(extract_from_notebook(nb_file))

    if not samples:
        click.echo("Warning: No code samples found in documentation", err=True)

    # 3. Calculate coverage
    result = calculate_coverage(samples, catalog, deduplicate=deduplicate)

    # 4. Output reports
    if not quiet:
        print_report(result, show_all_uncovered=show_all_uncovered)

    write_json(result, output)
    if not quiet:
        click.echo(f"JSON report written to {output}")

    if html:
        write_html(result, html)
        if not quiet:
            click.echo(f"HTML report written to {html}")

    # 5. Gap analysis
    if gaps and not quiet:
        _print_gap_analysis(catalog, samples, deduplicate)

    # 6. CI gate
    if fail_under > 0 and result.coverage_percent < fail_under:
        click.echo(
            f"\nError: Coverage {result.coverage_percent:.1f}% is below "
            f"threshold {fail_under}%",
            err=True,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
