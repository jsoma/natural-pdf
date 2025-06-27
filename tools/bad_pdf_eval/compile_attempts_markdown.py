#!/usr/bin/env python3
"""Compile multi-try enrichment attempts into a single Markdown report.

For every `summary.json` produced by the retry enrichment pipeline, this script
collects all attempts (initial + retries) and writes a human-readable markdown
file that shows, *per PDF*, every attempt alongside its quality score.

Example
-------
$ python -m tools.bad_pdf_eval.compile_attempts_markdown \
    --output eval_results/attempts_progress.md

The resulting markdown looks like::

    # Attempts Progress Report
    ## obe1Vq5 — obe1Vq5.pdf
    ### Attempt 0 (Score: 3/12)
    ...
    ### Attempt 1 (Score: 6/12)
    ...

This file can then be fed into an LLM for meta-analysis of score improvements
and guidance quality.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List

# Re-use the same constants as other evaluation utilities ---------------------
ROOT = Path(__file__).resolve().parent.parent.parent  # repo root
EVAL_DIR = ROOT / "eval_results"

# ---------------------------------------------------------------------------


def iter_summary_paths(submission: str | None) -> Iterable[Path]:
    """Yield all summary.json paths (optionally filtered by submission ID)."""
    if submission:
        p = EVAL_DIR / submission / "summary.json"
        if not p.exists():
            raise FileNotFoundError(
                f"No summary.json found for submission '{submission}' – expected {p}"
            )
        yield p
    else:
        yield from EVAL_DIR.glob("*/summary.json")


def load_summary(path: Path) -> dict:
    """Return the parsed JSON for the given summary path."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc


def build_markdown_for_summary(submission_id: str, summary: dict) -> str:
    """Return a markdown string for a single submission (all attempts)."""
    pdf_name = Path(summary.get("pdf", "")).name or "<unknown.pdf>"
    header = f"## {submission_id} — {pdf_name}"

    attempts: List[dict] = sorted(summary.get("attempts", []), key=lambda d: d.get("attempt", 0))
    if not attempts:
        return header + "\n\n_No attempts recorded – run the enrichment retry pipeline first._\n"

    sections: List[str] = [header]

    for att in attempts:
        num = att.get("attempt", "?")
        score = att.get("score", "?")
        tp = att.get("thought_process", "").strip()
        code = att.get("code_suggestion", "").rstrip()

        sections.append(f"### Attempt {num} (Score: {score}/12)")

        if tp:
            sections.append("**Thought Process**")
            # indent each line with > for blockquote formatting
            quoted_tp = "\n".join(f"> {line}" for line in tp.splitlines())
            sections.append(quoted_tp)

        if code:
            sections.append("```python")
            sections.append(code)
            sections.append("```")

    return "\n\n".join(sections)


def compile_report(paths: Iterable[Path]) -> str:
    """Aggregate individual submission markdown into one report."""
    pieces: List[str] = ["# Attempts Progress Report", ""]
    for p in sorted(paths):
        submission_id = p.parent.name
        summary = load_summary(p)
        pieces.append(build_markdown_for_summary(submission_id, summary))
        pieces.append("---")  # horizontal rule between PDFs
    return "\n\n".join(pieces).rstrip("-\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile multi-retry attempts into markdown.")
    parser.add_argument(
        "--output",
        type=Path,
        default=EVAL_DIR / "attempts_progress.md",
        help="Destination .md file (default: eval_results/attempts_progress.md).",
    )
    parser.add_argument("--submission", help="Only compile a single submission ID.")
    args = parser.parse_args()

    summary_paths = list(iter_summary_paths(args.submission))
    if not summary_paths:
        raise SystemExit("No summary.json files found.")

    md = compile_report(summary_paths)
    args.output.write_text(md, encoding="utf-8")
    print(
        f"[ok] Wrote markdown report to {args.output.relative_to(ROOT)} (covers {len(summary_paths)} PDFs)"
    )


if __name__ == "__main__":
    main()
