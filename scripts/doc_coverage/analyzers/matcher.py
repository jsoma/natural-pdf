"""Match extracted method calls to API catalog entries."""

from dataclasses import dataclass, field

from ..extractors.base import CodeSample
from .ast_walker import MethodCall, extract_calls

# Variable name to class mapping heuristics
# Keys can be exact names or patterns
VAR_TO_CLASS: dict[str, str] = {
    # PDF class
    "pdf": "PDF",
    "doc": "PDF",
    "document": "PDF",
    # Page class
    "page": "Page",
    "p": "Page",
    "pg": "Page",
    "first_page": "Page",
    "current_page": "Page",
    # Element classes - map to TextElement since Element is not in API catalog
    "element": "TextElement",
    "el": "TextElement",
    "elem": "TextElement",
    "text_element": "TextElement",
    "header": "TextElement",
    "title": "TextElement",
    "label": "TextElement",
    "value": "TextElement",
    "text": "TextElement",
    "line": "LineElement",
    "rect": "RectElement",
    "char": "CharElement",
    # Region class
    "region": "Region",
    "r": "Region",
    "area": "Region",
    "content": "Region",
    "section": "Region",
    "block": "Region",
    # Collections
    "collection": "ElementCollection",
    "elements": "ElementCollection",
    "items": "ElementCollection",
    "results": "ElementCollection",
    "matches": "ElementCollection",
    "found": "ElementCollection",
    "bold_text": "ElementCollection",
    "all_bold": "ElementCollection",
    "headers": "ElementCollection",
    "titles": "ElementCollection",
    "pages": "PageCollection",
    # Table-related
    "table": "TableResult",
    "tbl": "TableResult",
    "table_result": "TableResult",
    "df": "DataFrame",  # pandas, not our API
    # Extraction result-related
    "result": "StructuredDataResult",
    "answer": "StructuredDataResult",
    # Flow-related
    "flow": "Flow",
    "flow_region": "FlowRegion",
}

# Suffix patterns to class mappings
# If variable ends with these suffixes, map to corresponding class
VAR_SUFFIX_TO_CLASS: dict[str, str] = {
    "_region": "Region",
    "_area": "Region",
    "_section": "Region",
    "_block": "Region",
    "_content": "Region",
    "_element": "TextElement",
    "_elem": "TextElement",
    "_el": "TextElement",
    "_text": "TextElement",
    "_header": "TextElement",
    "_title": "TextElement",
    "_label": "TextElement",
    "_page": "Page",
    "_pages": "PageCollection",
    "_collection": "ElementCollection",
    "_elements": "ElementCollection",
    "_table": "TableResult",
    "_flow": "Flow",
}

# Patterns for receiver names that indicate specific types
RECEIVER_PATTERNS: dict[str, str] = {
    # Subscript patterns
    "pages[]": "Page",
    "pdf.pages[]": "Page",
    "doc.pages[]": "Page",
    # Attribute patterns
    "pdf.pages": "PageCollection",
    "doc.pages": "PageCollection",
    # Method result patterns (what type does method return?)
    "find()": "TextElement",
    "find_all()": "ElementCollection",
    "below()": "Region",
    "above()": "Region",
    "left()": "Region",
    "right()": "Region",
    "extract_table()": "TableResult",
    "ask()": "StructuredDataResult",
    "apply_ocr()": "ElementCollection",
    "analyze_layout()": "ElementCollection",
    "create_region()": "Region",
}


@dataclass
class CoverageResult:
    """Result of documentation coverage analysis.

    Attributes:
        total_methods: Total number of public API methods.
        covered_methods: Number of methods with at least one example.
        coverage_percent: Coverage as a percentage.
        method_hits: Dict mapping "Class.method" to example count.
        uncovered: List of methods with no examples.
        unparseable: List of file paths that failed parsing.
        samples_analyzed: Total number of code samples analyzed.
    """

    total_methods: int
    covered_methods: int
    coverage_percent: float
    method_hits: dict[str, int] = field(default_factory=dict)
    uncovered: list[str] = field(default_factory=list)
    unparseable: list[str] = field(default_factory=list)
    samples_analyzed: int = 0


def resolve_call(call: MethodCall, catalog: dict[str, list[str]]) -> str | None:
    """Resolve a method call to a fully-qualified API method.

    Args:
        call: The method call to resolve.
        catalog: API catalog mapping class names to method lists.

    Returns:
        Fully-qualified method name ("Class.method") or None if unresolved.
    """
    receiver = call.receiver

    # Try exact match in VAR_TO_CLASS
    if receiver in VAR_TO_CLASS:
        cls = VAR_TO_CLASS[receiver]
        if cls in catalog and call.method in catalog[cls]:
            return f"{cls}.{call.method}"

    # Try pattern matching for more complex receivers
    if receiver in RECEIVER_PATTERNS:
        cls = RECEIVER_PATTERNS[receiver]
        if cls in catalog and call.method in catalog[cls]:
            return f"{cls}.{call.method}"

    # Handle method chaining patterns
    for pattern, cls in RECEIVER_PATTERNS.items():
        if receiver.endswith(pattern):
            if cls in catalog and call.method in catalog[cls]:
                return f"{cls}.{call.method}"

    # Try suffix-based matching for variable names like content_region, header_element
    for suffix, cls in VAR_SUFFIX_TO_CLASS.items():
        if receiver.endswith(suffix):
            if cls in catalog and call.method in catalog[cls]:
                return f"{cls}.{call.method}"

    # Fallback: find unique method across all classes
    matches = []
    for cls, methods in catalog.items():
        if call.method in methods:
            matches.append(cls)

    if len(matches) == 1:
        return f"{matches[0]}.{call.method}"

    # Ambiguous or not in API
    return None


def calculate_coverage(
    samples: list[CodeSample],
    catalog: dict[str, list[str]],
    deduplicate: bool = False,
) -> CoverageResult:
    """Calculate documentation coverage from code samples.

    Args:
        samples: List of code samples extracted from documentation.
        catalog: API catalog mapping class names to method lists.
        deduplicate: If True, count unique method names instead of Class.method pairs.
            This treats `Page.extract_text` and `Region.extract_text` as the same method.

    Returns:
        CoverageResult with coverage statistics.
    """
    # Build set of all public methods
    all_methods = {f"{cls}.{method}" for cls, methods in catalog.items() for method in methods}

    hits: dict[str, int] = {m: 0 for m in all_methods}
    unparseable: set[str] = set()

    for sample in samples:
        calls = extract_calls(sample.source)

        # Track unparseable files (has content but no calls extracted)
        if not calls and sample.source.strip():
            # Could be unparseable or just no method calls
            # Only mark as unparseable if it looks like it should have calls
            if "." in sample.source and "(" in sample.source:
                unparseable.add(sample.file_path)

        for call in calls:
            resolved = resolve_call(call, catalog)
            if resolved and resolved in hits:
                hits[resolved] += 1

    if deduplicate:
        # Deduplicate: count unique method names, not Class.method pairs
        # A method is covered if ANY class's version has examples
        unique_methods: set[str] = set()
        for cls, methods in catalog.items():
            unique_methods.update(methods)

        # Aggregate hits by method name
        method_hits: dict[str, int] = {m: 0 for m in unique_methods}
        for full_method, count in hits.items():
            _, method_name = full_method.split(".", 1)
            method_hits[method_name] += count

        covered = [m for m, count in method_hits.items() if count > 0]
        uncovered = sorted([m for m, c in method_hits.items() if c == 0])
        total = len(unique_methods)
        covered_count = len(covered)

        return CoverageResult(
            total_methods=total,
            covered_methods=covered_count,
            coverage_percent=(covered_count / total * 100) if total > 0 else 0.0,
            method_hits=method_hits,
            uncovered=uncovered,
            unparseable=sorted(unparseable),
            samples_analyzed=len(samples),
        )

    # Default: count Class.method pairs separately
    covered = [m for m, count in hits.items() if count > 0]
    total = len(all_methods)
    covered_count = len(covered)

    return CoverageResult(
        total_methods=total,
        covered_methods=covered_count,
        coverage_percent=(covered_count / total * 100) if total > 0 else 0.0,
        method_hits=hits,
        uncovered=sorted([m for m, c in hits.items() if c == 0]),
        unparseable=sorted(unparseable),
        samples_analyzed=len(samples),
    )
