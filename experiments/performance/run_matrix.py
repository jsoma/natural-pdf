#!/usr/bin/env python3
"""Run baseline-vs-patch performance experiment matrices."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
HARNESS = REPO_ROOT / "scripts" / "perf_natural_pdf.py"
PATCH_ROOT = REPO_ROOT / "experiments" / "performance" / "patches"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "experiments" / "performance" / "results"


@dataclass(frozen=True)
class Candidate:
    track: str
    name: str
    patch: Path
    cases: str

    @property
    def label(self) -> str:
        return f"{self.track}-{self.name}"


CANDIDATES = [
    Candidate(
        "tiny_text",
        "prefilter_pdfplumber_chars",
        PATCH_ROOT / "tiny_text" / "prefilter_pdfplumber_chars" / "patch.py",
        "real:tiny-text-tables,micro:tiny-text-layout,real:guides-expenses-sample",
    ),
    Candidate(
        "tiny_text",
        "tuned_text_strategy",
        PATCH_ROOT / "tiny_text" / "tuned_text_strategy" / "patch.py",
        "real:tiny-text-tables,real:guides-expenses-sample",
    ),
    Candidate(
        "tiny_text",
        "text_engine_for_text_strategy",
        PATCH_ROOT / "tiny_text" / "text_engine_for_text_strategy" / "patch.py",
        "real:tiny-text-tables,real:guides-expenses-sample",
    ),
    Candidate(
        "page_materialization",
        "defer_decorations",
        PATCH_ROOT / "page_materialization" / "defer_decorations" / "patch.py",
        "real:multipage-table,real:0500000US42001,real:01-practice,micro:page-materialize:m27",
    ),
    Candidate(
        "page_materialization",
        "skip_char_elements",
        PATCH_ROOT / "page_materialization" / "skip_char_elements" / "patch.py",
        "real:multipage-table,real:0500000US42001,real:01-practice,micro:page-materialize:m27",
    ),
    Candidate(
        "selectors",
        "exact_query_cache",
        PATCH_ROOT / "selectors" / "exact_query_cache" / "patch.py",
        "real:Atlanta_Public_Schools_GA_sample,micro:repeated-selectors:atlanta,warm:repeated-page:m27,real:01-practice",
    ),
    Candidate(
        "selectors",
        "skip_exclusions_upper_bound",
        PATCH_ROOT / "selectors" / "skip_exclusions_upper_bound" / "patch.py",
        "real:Atlanta_Public_Schools_GA_sample,micro:repeated-selectors:atlanta,warm:repeated-page:m27,real:01-practice",
    ),
    Candidate(
        "guide_table",
        "skip_alt_text_scan",
        PATCH_ROOT / "guide_table" / "skip_alt_text_scan" / "patch.py",
        "real:guides-expenses-sample,micro:guide-table:01-practice,real:0500000US42001,real:multipage-table",
    ),
    Candidate(
        "guide_table",
        "default_word_cells",
        PATCH_ROOT / "guide_table" / "default_word_cells" / "patch.py",
        "real:guides-expenses-sample,micro:guide-table:01-practice,real:0500000US42001,real:multipage-table",
    ),
    Candidate(
        "guide_table",
        "vector_lines_only",
        PATCH_ROOT / "guide_table" / "vector_lines_only" / "patch.py",
        "real:guides-expenses-sample,micro:guide-table:01-practice,real:0500000US42001,real:multipage-table",
    ),
    Candidate(
        "rendering",
        "render_cache",
        PATCH_ROOT / "rendering" / "render_cache" / "patch.py",
        "real:0500000US42001,micro:guide-table:01-practice",
    ),
    Candidate(
        "workflow_pdf_reuse",
        "pennsylvania_pdf_cache",
        PATCH_ROOT / "workflow_pdf_reuse" / "pennsylvania_pdf_cache" / "patch.py",
        "real:0500000US42001",
    ),
    Candidate(
        "vector",
        "decorations_ephemeral",
        PATCH_ROOT / "vector" / "decorations_ephemeral" / "patch.py",
        "real:01-practice,real:guides-expenses-sample,real:multipage-table,real:0500000US42001,micro:page-materialize:m27",
    ),
    Candidate(
        "vector",
        "decorations_page_store",
        PATCH_ROOT / "vector" / "decorations_page_store" / "patch.py",
        "real:01-practice,real:guides-expenses-sample,real:multipage-table,real:0500000US42001,micro:page-materialize:m27",
    ),
    Candidate(
        "vector",
        "spatial_char_filter_ephemeral",
        PATCH_ROOT / "vector" / "spatial_char_filter_ephemeral" / "patch.py",
        "micro:extract-text-simple:m27,micro:extract-text-layout:m27,micro:tiny-text-layout,real:tiny-text-tables,real:policy-lines",
    ),
    Candidate(
        "vector",
        "region_overlap_page_store",
        PATCH_ROOT / "vector" / "region_overlap_page_store" / "patch.py",
        "real:01-practice,real:Atlanta_Public_Schools_GA_sample,real:policy-lines,micro:region-navigation:m27,micro:repeated-selectors:atlanta,warm:repeated-page:m27",
    ),
    Candidate(
        "vector",
        "region_overlap_memoized_arrays",
        PATCH_ROOT / "vector" / "region_overlap_memoized_arrays" / "patch.py",
        "real:01-practice,real:Atlanta_Public_Schools_GA_sample,real:policy-lines,micro:region-navigation:m27,micro:repeated-selectors:atlanta,warm:repeated-page:m27",
    ),
    Candidate(
        "vector",
        "table_word_assignment",
        PATCH_ROOT / "vector" / "table_word_assignment" / "patch.py",
        "real:guides-expenses-sample,micro:guide-table:01-practice,real:0500000US42001,real:multipage-table",
    ),
    Candidate(
        "vector",
        "simple_selector_fast_path",
        PATCH_ROOT / "vector" / "simple_selector_fast_path" / "patch.py",
        "real:01-practice,real:Atlanta_Public_Schools_GA_sample,real:pak-ks-expenses,micro:repeated-selectors:atlanta,micro:repeated-selectors:tiny-text,micro:repeated-selectors:pak-ks-expenses,warm:repeated-page:m27,warm:repeated-page:atlanta",
    ),
    Candidate(
        "vector",
        "simple_selector_page_store",
        PATCH_ROOT / "vector" / "simple_selector_page_store" / "patch.py",
        "real:01-practice,real:Atlanta_Public_Schools_GA_sample,real:pak-ks-expenses,micro:repeated-selectors:atlanta,micro:repeated-selectors:tiny-text,micro:repeated-selectors:pak-ks-expenses,warm:repeated-page:m27,warm:repeated-page:atlanta",
    ),
    Candidate(
        "vector",
        "lazy_text_elements_upper_bound",
        PATCH_ROOT / "vector" / "lazy_text_elements_upper_bound" / "patch.py",
        "real:01-practice,real:guides-expenses-sample,real:multipage-table,real:0500000US42001,real:pak-ks-expenses,real:policy-lines,micro:page-materialize:m27,micro:page-materialize:tiny-text,micro:page-materialize:pak-ks-expenses,warm:repeated-page:m27",
    ),
    Candidate(
        "lazy_chars",
        "minimal_field",
        PATCH_ROOT / "vector" / "lazy_chars_minimal_field" / "patch.py",
        "real:01-practice,real:guides-expenses-sample,real:multipage-table,real:0500000US42001,real:pak-ks-expenses,real:policy-lines,micro:page-materialize:m27,micro:page-materialize:tiny-text,micro:page-materialize:pak-ks-expenses,micro:page-chars:m27,micro:page-chars:tiny-text,micro:find-all-char:m27,warm:repeated-page:m27",
    ),
    Candidate(
        "lazy_chars",
        "proxy",
        PATCH_ROOT / "vector" / "lazy_chars_proxy" / "patch.py",
        "real:01-practice,real:guides-expenses-sample,real:multipage-table,real:0500000US42001,real:pak-ks-expenses,real:policy-lines,micro:page-materialize:m27,micro:page-materialize:tiny-text,micro:page-materialize:pak-ks-expenses,micro:page-chars:m27,micro:page-chars:tiny-text,micro:find-all-char:m27,warm:repeated-page:m27",
    ),
    Candidate(
        "lazy_chars",
        "raw_store",
        PATCH_ROOT / "vector" / "lazy_chars_raw_store" / "patch.py",
        "real:01-practice,real:guides-expenses-sample,real:multipage-table,real:0500000US42001,real:pak-ks-expenses,real:policy-lines,micro:page-materialize:m27,micro:page-materialize:tiny-text,micro:page-materialize:pak-ks-expenses,micro:page-chars:m27,micro:page-chars:tiny-text,micro:find-all-char:m27,warm:repeated-page:m27",
    ),
    Candidate(
        "lazy_chars",
        "columnar_store",
        PATCH_ROOT / "vector" / "lazy_chars_columnar_store" / "patch.py",
        "real:01-practice,real:guides-expenses-sample,real:multipage-table,real:0500000US42001,real:pak-ks-expenses,real:policy-lines,micro:page-materialize:m27,micro:page-materialize:tiny-text,micro:page-materialize:pak-ks-expenses,micro:page-chars:m27,micro:page-chars:tiny-text,micro:find-all-char:m27,warm:repeated-page:m27",
    ),
]


MODE_ARGS = {
    "screening": [
        "--iterations",
        "1",
        "--warmups",
        "0",
        "--no-tracemalloc",
        "--profile-slowest",
        "0",
    ],
    "confirmation": [
        "--iterations",
        "5",
        "--warmups",
        "1",
        "--no-tracemalloc",
        "--profile-slowest",
        "0",
    ],
    "profile": [
        "--iterations",
        "1",
        "--warmups",
        "0",
        "--profile-slowest",
        "2",
        "--profile-top-n",
        "20",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=sorted(MODE_ARGS), default="screening")
    parser.add_argument("--track", help="Only run candidates for this track.")
    parser.add_argument("--candidate", help="Only run one candidate name.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--list-candidates", action="store_true")
    return parser.parse_args()


def selected_candidates(args: argparse.Namespace) -> list[Candidate]:
    candidates = CANDIDATES
    if args.track:
        candidates = [candidate for candidate in candidates if candidate.track == args.track]
    if args.candidate:
        candidates = [candidate for candidate in candidates if candidate.name == args.candidate]
    return candidates


def run_command(command: list[str], *, dry_run: bool) -> dict[str, object]:
    if dry_run:
        return {"returncode": 0, "command": command, "dry_run": True}
    completed = subprocess.run(command, cwd=REPO_ROOT, text=True, check=False)
    return {"returncode": completed.returncode, "command": command, "dry_run": False}


def main() -> int:
    args = parse_args()
    candidates = selected_candidates(args)

    if args.list_candidates:
        for candidate in candidates:
            print(f"{candidate.track}/{candidate.name}\t{candidate.cases}")
        return 0

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    mode_args = MODE_ARGS[args.mode]
    summary: dict[str, object] = {
        "run_id": run_id,
        "mode": args.mode,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "candidates": [],
    }

    for candidate in candidates:
        candidate_summary = {
            "candidate": asdict(candidate) | {"patch": str(candidate.patch)},
            "baseline": None,
            "patched": None,
        }
        baseline_output = args.output_root / f"{run_id}-{candidate.label}-baseline"
        patched_output = args.output_root / f"{run_id}-{candidate.label}-patched"

        baseline_command = [
            sys.executable,
            str(HARNESS),
            "--output",
            str(baseline_output),
            "--experiment-label",
            f"{args.mode}-{candidate.label}-baseline",
            "--cases",
            candidate.cases,
            *mode_args,
        ]
        patched_command = [
            sys.executable,
            str(HARNESS),
            "--output",
            str(patched_output),
            "--experiment-label",
            f"{args.mode}-{candidate.label}-patched",
            "--patch",
            str(candidate.patch),
            "--cases",
            candidate.cases,
            *mode_args,
        ]

        candidate_summary["baseline"] = run_command(baseline_command, dry_run=args.dry_run)
        candidate_summary["patched"] = run_command(patched_command, dry_run=args.dry_run)
        summary["candidates"].append(candidate_summary)

    args.output_root.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_root / f"{run_id}-matrix-summary.json"
    if not args.dry_run:
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    else:
        print(json.dumps(summary, indent=2))

    failures = [
        entry
        for entry in summary["candidates"]
        if entry["baseline"]["returncode"] != 0 or entry["patched"]["returncode"] != 0
    ]
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
