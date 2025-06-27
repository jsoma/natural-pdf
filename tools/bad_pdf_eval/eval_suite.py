import argparse
import re
from pathlib import Path
from typing import Dict, List

import pandas as pd
from rich.console import Console

from .analyser import BadPDFAnalyzer, extract_page_hints
from .reporter import save_json
from .utils import find_local_pdf, slugify

console = Console()


DEFAULT_PAGES = [1, 2, 3]


def build_pages_list(row: Dict[str, str]) -> List[int]:
    pages = DEFAULT_PAGES.copy()
    text_fields = [
        row.get("What are we trying to get out of the PDF?", ""),
        row.get("What do you think makes this PDF bad?", ""),
    ]
    for field in text_fields:
        # Guard against NaN/None or other non-string pandas dtypes
        if isinstance(field, str) and field:
            pages += extract_page_hints(field)
    # deduplicate and sort
    pages = sorted(set(pages))
    return pages


def main():
    parser = argparse.ArgumentParser(description="Run bad PDF evaluation suite")
    parser.add_argument(
        "--csv",
        default="bad-pdfs/Bad PDF Submission form_Submissions_2025-06-22.csv",
        help="Path to submissions CSV",
    )
    parser.add_argument(
        "--output-dir",
        default="eval_results",
        help="Directory to write results into (will be git-ignored)",
    )
    parser.add_argument(
        "--max-row", type=int, default=None, help="debug: process only first n CSV rows"
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="process at most N PDFs with local files"
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="re-run analysis even if summary.json exists"
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    df = pd.read_csv(csv_path)
    if args.max_row:
        df = df.head(args.max_row)

    output_root = Path(args.output_dir)
    output_root.mkdir(exist_ok=True)

    master_records = []
    processed = 0

    try:
        for idx, row in df.iterrows():
            submission_id = row["Submission ID"]
            pdf_url = row.get("Your bad PDF (one per submission!)", "")
            pdf_path = find_local_pdf(submission_id, pdf_url)
            if not pdf_path or not pdf_path.exists():
                console.print(f"[red]PDF not found for {submission_id}. Skipping.")
                continue

            # Ignore files that are not .pdf (e.g. ZIPs mistakenly included)
            if pdf_path.suffix.lower() != ".pdf":
                console.print(
                    f"[yellow]Not a PDF ({pdf_path.suffix}) for {submission_id}; skipping."
                )
                continue

            sub_output = output_root / submission_id
            summary_path = sub_output / "summary.json"

            # Ensure the original PDF is stored alongside the analysis artefacts
            try:
                from shutil import copy2

                sub_output.mkdir(parents=True, exist_ok=True)
                dest_pdf = sub_output / pdf_path.name
                if not dest_pdf.exists():
                    copy2(pdf_path, dest_pdf)
            except Exception as copy_err:
                console.print(f"[yellow]Could not copy PDF into results folder: {copy_err}")

            if summary_path.exists() and not args.overwrite:
                console.print(
                    f"[yellow]Summary exists for {submission_id}; skipping (use --overwrite to refresh)"
                )
                continue

            pages = build_pages_list(row)
            try:
                analyser = BadPDFAnalyzer(
                    pdf_path=pdf_path, output_dir=sub_output, submission_meta=row, pages=pages
                )
                summary = analyser.run()
                master_records.append(summary)
            except Exception as e:
                console.print(f"[red]Error processing {submission_id}: {e}. Skipping.")
                continue
            processed += 1
            if args.limit and processed >= args.limit:
                break
    except KeyboardInterrupt:
        console.print("[bold yellow]\nInterrupted by user – saving progress made so far…")
    finally:
        # Save master index even on interrupt
        if master_records:
            save_json(master_records, output_root / "master_index.json")
            console.print(f"[bold green]Progress saved to {output_root / 'master_index.json'}")
        console.print(f"[bold green]Finished. Results in {output_root}")


if __name__ == "__main__":
    main()
