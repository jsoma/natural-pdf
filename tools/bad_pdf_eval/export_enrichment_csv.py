from __future__ import annotations

"""Export enrichment data (id, thought_process, code_suggestion) to a CSV.

Usage
-----
python -m tools.bad_pdf_eval.export_enrichment_csv --out eval_results/enrichment.csv
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parent.parent.parent  # repo root
EVAL_DIR = ROOT / "eval_results"


def collect_records() -> List[Dict[str, str]]:
    records: List[Dict[str, str]] = []
    for summary_path in EVAL_DIR.glob("*/summary.json"):
        try:
            data = json.loads(summary_path.read_text())
        except Exception:
            continue
        tp = data.get("thought_process", "").strip()
        cs = data.get("code_suggestion", "").strip()
        if not tp and not cs:
            # Skip summaries without enrichment at doc level
            continue
        records.append(
            {
                "id": data.get("submission_id", summary_path.parent.name),
                "thought_process": tp.replace("\n", " ").strip(),
                "code_suggestion": cs.replace("\n", " ").strip(),
            }
        )
    return records


def main():
    ap = argparse.ArgumentParser(description="Export enriched summaries to CSV.")
    ap.add_argument(
        "--out", default=str(EVAL_DIR / "enrichment_export.csv"), help="Output CSV path"
    )
    args = ap.parse_args()

    records = collect_records()
    if not records:
        print("No enriched summaries found.")
        return

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "thought_process", "code_suggestion"])
        writer.writeheader()
        for rec in records:
            writer.writerow(rec)

    print(f"Wrote {len(records)} records to {out_path}")


if __name__ == "__main__":
    main()
