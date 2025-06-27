import json
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parent.parent.parent / "eval_results"


def collect() -> List[dict]:
    rows = []
    for subdir in ROOT.iterdir():
        if not subdir.is_dir():
            continue
        summary_path = subdir / "summary.json"
        if not summary_path.exists():
            continue
        try:
            data = json.loads(summary_path.read_text())
        except Exception as e:
            print(f"Failed to parse {summary_path}: {e}")
            continue
        submission_id = data.get("submission_id", subdir.name)
        description = data.get("description", "")
        language = data.get("language", "")
        issues = data.get("issues", "")

        # ---------------- document-level enrichment (added by llm_enrich.py) ----
        doc_tp = (data.get("thought_process") or "").strip()
        doc_cs = (data.get("code_suggestion") or "").strip()
        doc_diff = data.get("difficult_elements", [])
        doc_test = (data.get("test_case") or "").strip()

        page_snippets = []
        features = set()
        for p in data.get("pages", [])[:5]:  # first 5 pages enough for summary
            cs = (p.get("code_suggestion") or "").strip()
            tp = (p.get("thought_process") or "").strip()
            if not cs and not tp:
                continue
            page_snippets.append(
                {
                    "page": p.get("page_number"),
                    "code": cs,
                    "thought": tp,
                }
            )
            # --- lightweight feature tagging --------------------------------
            gt = (p.get("goal_tag") or "").lower()
            if "table" in gt:
                features.add("table")
            if "text" in gt:
                features.add("text")
            # look into region labels for structural hints
            for reg in p.get("layout_tatr_regions", []) + p.get("layout_yolo_regions", []):
                label = (reg.get("label") or reg.get("type") or "").lower()
                if label == "table":
                    features.add("table")
                if label in {"figure", "isolate_formula"}:
                    features.add("figure")
            # parse difficulties hints in thought_process
            difficulties = tp.lower()
            if "scanned_image" in difficulties:
                features.add("scanned_image")
            if "tiny_font" in difficulties or "small font" in difficulties:
                features.add("small_font")
        # language-based feature
        if language and language.lower() not in {"english", "en", "en-us"}:
            features.add("non_english")

        rows.append(
            {
                "id": submission_id,
                "language": language,
                "issues": issues,
                "description": description,
                "doc_thought": doc_tp,
                "doc_code": doc_cs,
                "doc_difficult": doc_diff,
                "doc_test": doc_test,
                "snippets": page_snippets,
                "features": sorted(features),
            }
        )
    return rows


def export_markdown(rows: List[dict]):
    lines = ["# Evaluation Summaries\n"]
    for r in sorted(rows, key=lambda x: x["id"]):
        lines.append(f"## {r['id']}")
        if r["description"]:
            lines.append(f"*Description*: {r['description']}")
        if r["issues"]:
            lines.append(f"*Issues*: {r['issues']}")
        if r["language"]:
            lines.append(f"*Language*: {r['language']}")
        if r.get("features"):
            lines.append(f"*Features*: {', '.join(r['features'])}")

        # ---- document-level enrichment -----------------------------------
        if r.get("doc_thought") or r.get("doc_code"):
            lines.append("\n### Document-level enrichment")
        if r.get("doc_thought"):
            lines.append("**Doc thought process:**")
            lines.append(f"```")
            lines.append(r["doc_thought"])
            lines.append(f"```")
        if r.get("doc_code"):
            lines.append("**Doc code suggestion:**")
            lines.append(f"```python")
            lines.append(r["doc_code"])
            lines.append(f"```")
        if r.get("doc_difficult"):
            lines.append("*Difficult elements*: " + ", ".join(r["doc_difficult"]))
        if r.get("doc_test"):
            lines.append("*Suggested test*: " + r["doc_test"])

        lines.append("")
        for s in r["snippets"]:
            lines.append(f"### Page {s['page']}")
            if s["thought"]:
                lines.append("**Thoughts**:")
                lines.append(f"```\n{s['thought']}\n```")
            if s["code"]:
                lines.append("**Code suggestion**:")
                lines.append(f"```python\n{s['code']}\n```")
            lines.append("")
        lines.append("\n---\n")
    Path("eval_results/collated_summary.md").write_text("\n".join(lines))


if __name__ == "__main__":
    rows = collect()
    export_markdown(rows)
    print(f"Wrote {len(rows)} summaries to eval_results/collated_summary.md")
