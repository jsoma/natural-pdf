"""HTML report output for documentation coverage."""

from datetime import datetime
from pathlib import Path

from ..analyzers.matcher import CoverageResult


def write_html(result: CoverageResult, path: str | Path) -> None:
    """Write coverage result to an HTML file.

    Args:
        result: Coverage analysis result.
        path: Output file path.
    """
    # Group methods by class
    by_class: dict[str, dict[str, int]] = {}
    for method, count in result.method_hits.items():
        cls, name = method.split(".", 1)
        by_class.setdefault(cls, {})[name] = count

    # Calculate per-class coverage
    class_stats = []
    for cls in sorted(by_class.keys()):
        methods = by_class[cls]
        total = len(methods)
        covered = sum(1 for c in methods.values() if c > 0)
        pct = (covered / total * 100) if total > 0 else 0
        class_stats.append(
            {
                "name": cls,
                "total": total,
                "covered": covered,
                "percent": pct,
                "methods": methods,
            }
        )

    # Generate HTML
    html = _generate_html(result, class_stats)
    Path(path).write_text(html)


def _generate_html(result: CoverageResult, class_stats: list[dict]) -> str:
    """Generate HTML report content."""
    coverage_color = (
        "#22c55e"
        if result.coverage_percent >= 70
        else "#eab308" if result.coverage_percent >= 50 else "#ef4444"
    )

    # Build class rows
    class_rows = ""
    for stat in class_stats:
        color = (
            "#22c55e"
            if stat["percent"] >= 70
            else "#eab308" if stat["percent"] >= 50 else "#ef4444"
        )
        method_list = "".join(
            f'<div class="method {"covered" if count > 0 else "uncovered"}">'
            f'{name} <span class="count">({count})</span></div>'
            for name, count in sorted(stat["methods"].items())
        )
        class_rows += f"""
        <div class="class-card">
            <div class="class-header">
                <span class="class-name">{stat["name"]}</span>
                <span class="class-coverage" style="color: {color}">
                    {stat["covered"]}/{stat["total"]} ({stat["percent"]:.0f}%)
                </span>
            </div>
            <div class="methods">{method_list}</div>
        </div>
        """

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Documentation Coverage Report</title>
    <style>
        * {{ box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            max-width: 1200px;
            margin: 0 auto;
            padding: 2rem;
            background: #f9fafb;
            color: #1f2937;
        }}
        h1 {{ margin-bottom: 0.5rem; }}
        .summary {{
            background: white;
            border-radius: 8px;
            padding: 1.5rem;
            margin-bottom: 2rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        .coverage-badge {{
            font-size: 3rem;
            font-weight: bold;
            color: {coverage_color};
        }}
        .stats {{
            color: #6b7280;
            margin-top: 0.5rem;
        }}
        .class-card {{
            background: white;
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 1rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        .class-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 0.5rem;
            padding-bottom: 0.5rem;
            border-bottom: 1px solid #e5e7eb;
        }}
        .class-name {{
            font-weight: 600;
            font-size: 1.1rem;
        }}
        .class-coverage {{
            font-weight: 500;
        }}
        .methods {{
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
        }}
        .method {{
            font-size: 0.85rem;
            padding: 0.25rem 0.5rem;
            border-radius: 4px;
            font-family: monospace;
        }}
        .method.covered {{
            background: #d1fae5;
            color: #065f46;
        }}
        .method.uncovered {{
            background: #fee2e2;
            color: #991b1b;
        }}
        .count {{
            opacity: 0.7;
        }}
        .timestamp {{
            color: #9ca3af;
            font-size: 0.85rem;
            margin-top: 2rem;
        }}
    </style>
</head>
<body>
    <h1>Documentation Coverage Report</h1>
    <div class="summary">
        <div class="coverage-badge">{result.coverage_percent:.1f}%</div>
        <div class="stats">
            {result.covered_methods} of {result.total_methods} methods documented
            &bull; {result.samples_analyzed} code samples analyzed
        </div>
    </div>

    <h2>Coverage by Class</h2>
    {class_rows}

    <div class="timestamp">
        Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    </div>
</body>
</html>
"""
