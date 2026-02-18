"""JSON report output for documentation coverage."""

import json
from pathlib import Path

from ..analyzers.matcher import CoverageResult


def write_json(result: CoverageResult, path: str | Path) -> None:
    """Write coverage result to a JSON file.

    Args:
        result: Coverage analysis result.
        path: Output file path.
    """
    data = {
        "summary": {
            "total_methods": result.total_methods,
            "covered_methods": result.covered_methods,
            "coverage_percent": round(result.coverage_percent, 2),
            "samples_analyzed": result.samples_analyzed,
        },
        "methods": result.method_hits,
        "uncovered": result.uncovered,
        "unparseable_files": result.unparseable,
    }

    Path(path).write_text(json.dumps(data, indent=2, sort_keys=True))
