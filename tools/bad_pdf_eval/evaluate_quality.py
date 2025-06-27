"""Evaluate the quality of LLM enrichment suggestions.

This script analyzes the code suggestions to identify:
- Use of modern features (Guides API, extract_table)
- Avoidance of anti-patterns (placeholder text, unnecessary TATR)
- Practical, working code
"""

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List


def analyze_code_quality(code: str) -> Dict[str, Any]:
    """Analyze a code suggestion for quality indicators."""

    quality = {
        "uses_guides": bool(
            re.search(r"from natural_pdf\.analyzers import Guides|Guides\(|Guides\.", code)
        ),
        "uses_tatr": bool(re.search(r'analyze_layout\([\'"]tatr[\'"]?\)', code)),
        "uses_extract_table": bool(re.search(r"\.extract_table\(\)", code)),
        "has_placeholder_text": bool(
            re.search(r'[\'"](?:AnchorText|Texts|Also|HeaderText|TableHeader)[\'"]', code)
        ),
        "uses_real_text": bool(
            re.search(
                r'[\'"](?!AnchorText|Texts|Also|HeaderText|TableHeader)[^\'"{}\[\]]+[\'"]', code
            )
        )
        and not bool(
            re.search(r'[\'"](?:AnchorText|Texts|Also|HeaderText|TableHeader)[\'"]', code)
        ),
        "uses_snap_to_whitespace": bool(re.search(r"snap_to_whitespace", code)),
        "uses_parent_navigation": bool(re.search(r"\.parent\(", code)),
        "uses_until": bool(re.search(r"until\s*=", code)),
        "has_ocr_call": bool(re.search(r"apply_ocr\(", code)),
        "removes_text_layer": bool(re.search(r"remove_text_layer\(|text_layer\s*=\s*False", code)),
    }

    # Calculate score based on quality indicators
    score = 0
    if quality["uses_guides"]:
        score += 3  # Major positive: modern approach
    if quality["uses_tatr"]:
        score += 1  # Minor positive: valid for complex layouts
    if quality["uses_extract_table"]:
        score += 2  # Positive: using singular method
    if quality["uses_real_text"]:
        score += 2  # Positive: using actual anchors
    if quality["uses_snap_to_whitespace"]:
        score += 2  # Positive: modern Guides feature
    if quality["uses_parent_navigation"]:
        score += 1  # Positive: robust navigation
    if quality["uses_until"]:
        score += 1  # Positive when appropriate: precise region selection

    quality["score"] = score
    quality["max_score"] = 12

    return quality


def analyze_difficult_elements(elements: List[str]) -> Dict[str, int]:
    """Count types of difficult elements identified."""

    patterns = {
        "tiny_font": r"tiny.*font|small.*font|font.*size|<\s*\d+\s*pt",
        "rtl_language": r"arabic|hebrew|rtl|right.*to.*left",
        "scanned": r"scanned|image.*only|no.*text.*layer",
        "complex_layout": r"column|multi.*column|layout",
        "handwritten": r"handwritten|hand.*written",
        "redacted": r"redact",
    }

    counts = defaultdict(int)
    for element in elements:
        element_lower = element.lower()
        for category, pattern in patterns.items():
            if re.search(pattern, element_lower):
                counts[category] += 1

    return dict(counts)


def evaluate_submission(submission_path: Path) -> Dict[str, Any]:
    """Evaluate a single submission's enrichment quality."""

    summary_path = submission_path / "summary.json"
    if not summary_path.exists():
        return None

    data = json.loads(summary_path.read_text())

    result = {
        "submission_id": data.get("submission_id", submission_path.name),
        "has_doc_enrichment": bool(data.get("code_suggestion")),
        "doc_code_quality": None,
        "difficult_elements_analysis": None,
        "page_code_quality": [],
    }

    # Analyze document-level code
    if data.get("code_suggestion"):
        result["doc_code_quality"] = analyze_code_quality(data["code_suggestion"])

    # Analyze difficult elements
    if data.get("difficult_elements"):
        result["difficult_elements_analysis"] = analyze_difficult_elements(
            data["difficult_elements"]
        )

    # Analyze page-level code
    for page in data.get("pages", []):
        if page.get("code_suggestion"):
            page_quality = analyze_code_quality(page["code_suggestion"])
            page_quality["page_number"] = page.get("page_number")
            result["page_code_quality"].append(page_quality)

    return result


def main():
    """Analyze all submissions and generate quality report."""

    eval_dir = Path("eval_results")
    results = []

    for submission_dir in eval_dir.iterdir():
        if submission_dir.is_dir() and (submission_dir / "summary.json").exists():
            result = evaluate_submission(submission_dir)
            if result:
                results.append(result)

    # Aggregate statistics
    stats = {
        "total_submissions": len(results),
        "with_doc_enrichment": sum(1 for r in results if r["has_doc_enrichment"]),
        "using_guides": 0,
        "using_tatr": 0,
        "using_placeholders": 0,
        "avg_quality_score": 0,
        "difficult_elements_breakdown": defaultdict(int),
    }

    all_scores = []
    for result in results:
        if result["doc_code_quality"]:
            quality = result["doc_code_quality"]
            if quality["uses_guides"]:
                stats["using_guides"] += 1
            if quality["uses_tatr"]:
                stats["using_tatr"] += 1
            if quality["has_placeholder_text"]:
                stats["using_placeholders"] += 1
            all_scores.append(quality["score"])

        if result["difficult_elements_analysis"]:
            for elem_type, count in result["difficult_elements_analysis"].items():
                stats["difficult_elements_breakdown"][elem_type] += count

    if all_scores:
        stats["avg_quality_score"] = sum(all_scores) / len(all_scores)

    # Generate report
    print("\n=== Natural PDF Evaluation Quality Report ===\n")
    print(f"Total submissions analyzed: {stats['total_submissions']}")
    print(f"With document enrichment: {stats['with_doc_enrichment']}")
    print(f"\nCode Quality Metrics:")
    print(
        f"  Using Guides API: {stats['using_guides']} ({stats['using_guides']/stats['with_doc_enrichment']*100:.1f}%)"
    )
    print(
        f"  Using TATR: {stats['using_tatr']} ({stats['using_tatr']/stats['with_doc_enrichment']*100:.1f}%)"
    )
    print(
        f"  Using placeholders: {stats['using_placeholders']} ({stats['using_placeholders']/stats['with_doc_enrichment']*100:.1f}%)"
    )
    print(f"  Average quality score: {stats['avg_quality_score']:.1f}/12")

    print(f"\nDifficult Elements Identified:")
    for elem_type, count in sorted(
        stats["difficult_elements_breakdown"].items(), key=lambda x: x[1], reverse=True
    ):
        print(f"  {elem_type}: {count}")

    # Save detailed results
    output_path = eval_dir / "quality_analysis.json"
    with open(output_path, "w") as f:
        json.dump({"stats": stats, "detailed_results": results}, f, indent=2)

    print(f"\nDetailed results saved to: {output_path}")


if __name__ == "__main__":
    main()
