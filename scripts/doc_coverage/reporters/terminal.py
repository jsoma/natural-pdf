"""Terminal output for documentation coverage results."""

from ..analyzers.matcher import CoverageResult


def print_report(
    result: CoverageResult,
    show_all_uncovered: bool = False,
    max_uncovered: int = 20,
) -> None:
    """Print coverage report to terminal.

    Args:
        result: Coverage analysis result.
        show_all_uncovered: Show all uncovered methods (vs. truncated list).
        max_uncovered: Maximum uncovered methods to show if not showing all.
    """
    try:
        from rich.console import Console
        from rich.table import Table

        _print_rich_report(result, show_all_uncovered, max_uncovered)
    except ImportError:
        _print_plain_report(result, show_all_uncovered, max_uncovered)


def _print_rich_report(
    result: CoverageResult,
    show_all_uncovered: bool,
    max_uncovered: int,
) -> None:
    """Print report using rich formatting."""
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    console = Console()

    # Header
    color = (
        "green"
        if result.coverage_percent >= 70
        else "yellow" if result.coverage_percent >= 50 else "red"
    )
    console.print()
    console.print(
        Panel(
            f"[bold {color}]{result.coverage_percent:.1f}%[/bold {color}] documentation coverage\n"
            f"{result.covered_methods}/{result.total_methods} methods documented\n"
            f"{result.samples_analyzed} code samples analyzed",
            title="Documentation Coverage Report",
        )
    )

    # Top covered methods
    if result.method_hits:
        top_hits = sorted(
            [(m, c) for m, c in result.method_hits.items() if c > 0],
            key=lambda x: x[1],
            reverse=True,
        )[:10]

        if top_hits:
            console.print("\n[bold]Most documented methods:[/bold]")
            table = Table(show_header=True, header_style="bold")
            table.add_column("Method", style="cyan")
            table.add_column("Examples", justify="right", style="green")

            for method, count in top_hits:
                table.add_row(method, str(count))

            console.print(table)

    # Uncovered methods
    if result.uncovered:
        console.print(f"\n[yellow]Undocumented methods ({len(result.uncovered)}):[/yellow]")

        # Check if methods are in "Class.method" format or just method names
        has_class_prefix = any("." in m for m in result.uncovered)

        if has_class_prefix:
            # Group by class
            by_class: dict[str, list[str]] = {}
            for method in result.uncovered:
                cls, name = method.split(".", 1)
                by_class.setdefault(cls, []).append(name)

            shown = 0
            for cls in sorted(by_class.keys()):
                if not show_all_uncovered and shown >= max_uncovered:
                    break
                methods = by_class[cls]
                for m in sorted(methods):
                    if not show_all_uncovered and shown >= max_uncovered:
                        break
                    console.print(f"  - [dim]{cls}.[/dim]{m}")
                    shown += 1
        else:
            # Just method names (deduplicated mode)
            shown = 0
            for method in sorted(result.uncovered):
                if not show_all_uncovered and shown >= max_uncovered:
                    break
                console.print(f"  - {method}")
                shown += 1

        remaining = len(result.uncovered) - shown
        if remaining > 0:
            console.print(f"  [dim]... and {remaining} more[/dim]")

    # Unparseable files
    if result.unparseable:
        console.print(f"\n[red]Unparseable files ({len(result.unparseable)}):[/red]")
        for path in result.unparseable[:5]:
            console.print(f"  - {path}")
        if len(result.unparseable) > 5:
            console.print(f"  ... and {len(result.unparseable) - 5} more")

    console.print()


def _print_plain_report(
    result: CoverageResult,
    show_all_uncovered: bool,
    max_uncovered: int,
) -> None:
    """Print report using plain text formatting."""
    print()
    print("=" * 50)
    print("Documentation Coverage Report")
    print("=" * 50)
    print(f"\nCoverage: {result.coverage_percent:.1f}%")
    print(f"  {result.covered_methods}/{result.total_methods} methods documented")
    print(f"  {result.samples_analyzed} code samples analyzed")

    # Top covered
    if result.method_hits:
        top_hits = sorted(
            [(m, c) for m, c in result.method_hits.items() if c > 0],
            key=lambda x: x[1],
            reverse=True,
        )[:10]

        if top_hits:
            print("\nMost documented methods:")
            for method, count in top_hits:
                print(f"  {method}: {count}")

    # Uncovered
    if result.uncovered:
        print(f"\nUndocumented methods ({len(result.uncovered)}):")
        shown = result.uncovered if show_all_uncovered else result.uncovered[:max_uncovered]
        for method in shown:
            print(f"  - {method}")
        remaining = len(result.uncovered) - len(shown)
        if remaining > 0:
            print(f"  ... and {remaining} more")

    # Unparseable
    if result.unparseable:
        print(f"\nUnparseable files ({len(result.unparseable)}):")
        for path in result.unparseable[:5]:
            print(f"  - {path}")

    print()
