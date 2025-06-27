"""Enrich evaluation summaries with LLM-generated thought_process and code_suggestion.

Usage
-----
python -m tools.bad_pdf_eval.llm_enrich --submission ja6EqV1 --model o3

Environment
-----------
OPENAI_API_KEY must be set or passed via --api-key.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import textwrap
from pathlib import Path
from typing import Any, Dict, List

from openai import OpenAI
from PIL import Image
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent.parent  # repo root
EVAL_DIR = ROOT / "eval_results"
CHEATSHEET_PATH = ROOT / "tools" / "bad_pdf_eval" / "LLM_NaturalPDF_CheatSheet.md"
WORKFLOWS_PATH = ROOT / "tools" / "bad_pdf_eval" / "LLM_NaturalPDF_Workflows.md"
DECISION_TREE_PATH = ROOT / "tools" / "bad_pdf_eval" / "extraction_decision_tree.md"


def read_md(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def img_to_b64_jpeg(path: Path, max_px: int = 1024) -> str:
    """Return base64-encoded tiny JPEG thumbnail."""
    with Image.open(path) as im:
        im.thumbnail((max_px, max_px))
        buffered = BytesIO()
        im.convert("RGB").save(buffered, format="JPEG", quality=40, optimize=True)
        return base64.b64encode(buffered.getvalue()).decode()


from io import BytesIO


def build_prompt(page: Dict[str, Any]) -> List[Dict[str, str]]:
    """Return OpenAI chat prompt messages list."""
    cheatsheet = read_md(CHEATSHEET_PATH)
    workflows = read_md(WORKFLOWS_PATH)

    image_section = None
    if page.get("image") and Path(page["image"]).exists():
        try:
            b64 = img_to_b64_jpeg(Path(page["image"]))
            image_section = {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "low"},
            }
        except Exception:
            pass

    context_json = {
        "describe": page.get("describe", ""),
        "inspect": page.get("inspect", ""),
        "layout_yolo_regions": page.get("layout_yolo_regions", []),
        "layout_tatr_regions": page.get("layout_tatr_regions", []),
        "blob_sample": page.get("blobs_sample", []),
        "ocr_sample": page.get("ocr_sample", []),
    }

    sys_msg = textwrap.dedent(
        f"""
        You are an expert Natural-PDF engineer. Use the cheat-sheet and workflows to craft bespoke extraction code.
        Return JSON with two keys: thought_process (concise reasoning) and code_suggestion (Python code). Do not add extra keys.
        """
    )

    messages = [
        {"role": "system", "content": sys_msg},
        {"role": "system", "content": "CHEATSHEET:\n" + cheatsheet},
        {"role": "system", "content": "WORKFLOWS:\n" + workflows},
    ]

    user_parts = [
        f"Goal: {page.get('goal_tag') or 'generic_extraction'} — {page.get('goal', '') or 'Extract the most useful information (text, tables, key/value pairs) from the page.'}",
        f"Page number: {page['page_number']}",
        "Context JSON:" + json.dumps(context_json),
        "Provide your JSON response now.",
    ]
    if image_section:
        user_parts.insert(2, image_section)
    messages.append({"role": "user", "content": "\n\n".join(user_parts)})
    return messages


def build_pdf_prompt(summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    cheatsheet = read_md(CHEATSHEET_PATH)
    workflows = read_md(WORKFLOWS_PATH)
    decision_tree = read_md(DECISION_TREE_PATH)

    pdf_overview = [
        f"PDF: {Path(summary['pdf']).name}",
        f"Goal: {summary.get('goal') or 'Extract useful information from the document'}",
        f"Total pages analysed: {len(summary['pages'])}",
    ]

    per_page_sections = []
    for page in summary["pages"]:
        image_section = None
        if page.get("image") and Path(page["image"]).exists():
            try:
                b64 = img_to_b64_jpeg(Path(page["image"]))
                image_section = {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "low"},
                }
            except Exception:
                pass
        context_json = {
            "describe": page.get("describe", ""),
            "inspect": page.get("inspect", ""),
            "layout_yolo_regions": page.get("layout_yolo_regions", []),
            "layout_tatr_regions": page.get("layout_tatr_regions", []),
            "blob_sample": page.get("blobs_sample", []),
            "ocr_sample": page.get("ocr_sample", []),
        }
        per_page_sections.append(
            {
                "page_number": page["page_number"],
                "goal_tag": page.get("goal_tag") or "generic_extraction",
                "image": image_section,
                "context": context_json,
            }
        )

    sys_msg = textwrap.dedent(
        """
        You are a Natural-PDF engineer with full access to the provided evidence
        (describe/inspect text, YOLO & TATR regions, blob samples, OCR snippets, images).

        Rely on these artefacts—not on generic heuristics. Avoid phrases like "try" or "usually this works".
        If the evidence is genuinely insufficient, state exactly what is missing.

        Extraction strategy:
          1. Start with the text layer: `page.extract_text()`, `page.extract_table()`, or region selectors.
          2. For tables, strongly prefer the Guides API over TATR:
             • Use `Guides.from_content()` with actual column headers as markers
             • Apply `.snap_to_whitespace()` to auto-align to natural gaps
             • Only fall back to TATR for genuinely complex multi-table pages
          3. Use **anchor-based region selection**: locate a stable header/label/line/rect and select the area
             between anchors via `.find()`, `.below()`, `.above()`, `.until()`, `.expand()`, etc.
             Example: `page.find('text:contains("Violations")').below(until='text:bold')`.
          4. Strongly prefer until= to find a specific ending point as opposed to a pixel-based approach,
             as this allows your code to work on potentially other similar pages of the document.
          5. Direct region extraction often works: `region.extract_table()` without any layout model.

        Recent improvements to leverage:
          • Tiny text (<7pt) is now extracted reliably - no need to flag as difficult
          • RTL languages (Arabic, Hebrew) work automatically with proper BiDi
          • Use `.extract_table()` (singular) which returns TableResult with .df property
          • Guides API can detect lines from pixels directly - no vector lines needed
          • Can discard corrupted text layers with `PDF(..., text_layer=False)` or `page.remove_text_layer()`

        Handle tables, key-value forms, and free-form paragraphs with the same anchor-driven approach. Key-value
        forms might be easily extracted with .ask(...) or .extract(), feel free to mention as an option
        but try to not rely on it.

        Use Natural PDF Flows to access multi-page or columnar content, falling back on loops when necessary.

        If it seems like the approach is not ideal, or that additional features would be useful in
        this use case, outline the specifics of the issues and what additional information/approaches/code/etc
        would allow you to more easily extract the information.

        When working with pages or elements, try to use .apply and .filter. Natural PDF stresses
        a fluent API, and for loops are discouraged.

        Return ONE JSON object with exactly these keys:
          • thought_process – concise reasoning about your approach, noting if Guides would work better than TATR
          • code_suggestion – executable Python snippet using natural_pdf
          • difficult_elements – bullet list of page features that are *hard* for any extraction engine **and that you can _prove_ from the supplied evidence** (exclude tiny fonts unless <5pt, exclude RTL languages). If no difficult element is evident, return an empty list. Do *not* speculate.
          • test_case – short description of how this PDF/page could be turned into an automated regression test

        Code-style expectations:
          • Use **real sample text** from the page as anchors — never placeholders such as
            "AnchorText", "Texts", or "Also".  Look in the inspect/describe data for actual text.
          • When a page is flagged as *scanned_image* (or no text layer exists) your code
            MUST call `page.apply_ocr()` *before* any `.find()` or `.extract_text()` calls.
          • If text appears as "(cid:xxx)" in the evidence, use `page.remove_text_layer()` or
            `PDF(..., text_layer=False)` before OCR to avoid corrupted text interference.
          • For table extraction, show Guides-based approach first, TATR only as fallback
          • Prefer `header_el.parent('table')` (up-tree navigation) over a global
            `page.find('table')[i]` positional index — this is more robust to layout changes.
          • Use `.below()` or `.above()` to select regions. Add `until=` only when you need to
            stop before reaching the page edge (e.g., before another section). Going to page edge
            is fine without `until`.
          • Keep page-level suggestions consistent with document-level patterns (same extraction approach)
        """
    )

    messages = [
        {"role": "system", "content": sys_msg},
        {"role": "system", "content": "DECISION TREE:\n" + decision_tree},
        {"role": "system", "content": "CHEATSHEET:\n" + cheatsheet},
        {"role": "system", "content": "WORKFLOWS:\n" + workflows},
    ]

    user_content: List[Dict[str, Any]] = [{"type": "text", "text": "\n".join(pdf_overview)}]
    for sec in per_page_sections:
        txt = json.dumps({k: sec[k] for k in ("page_number", "goal_tag", "context")})
        user_content.append({"type": "text", "text": txt})
        if sec["image"]:
            user_content.append(sec["image"])

    messages.append({"role": "user", "content": user_content})
    return messages


# -------------------------------------------------
# Structured output via Pydantic model + function call
# -------------------------------------------------


class DocOutput(BaseModel):
    """LLM enrichment for a whole PDF (single object)."""

    thought_process: str = Field(
        ...,
        description="Overall reasoning about the PDF and extraction plan, noting whether Guides API would be better than TATR for tables",
    )
    code_suggestion: str = Field(
        ...,
        description="Python snippet using natural_pdf, preferring Guides API over TATR for table extraction",
    )
    difficult_elements: List[str] = Field(
        ...,
        description="Bullet list of page features that are genuinely hard (not tiny fonts >5pt or RTL languages)",
    )
    test_case: str = Field(
        ..., description="Specific assertion that could verify the extraction worked correctly"
    )


def enrich_summary(summary_path: Path, api_key: str, model: str = "o3"):
    summary = json.loads(summary_path.read_text())

    # Decide whether to re-enrich
    if not FORCE and summary.get("thought_process") and summary.get("code_suggestion"):
        print(f"[skip] {summary_path.parent.name}: already enriched (use --force to overwrite)")
        return

    print(f"[send] {summary_path.parent.name}: requesting enrichment for entire document")

    client = OpenAI(api_key=api_key)
    msgs = build_pdf_prompt(summary)

    completion = client.beta.chat.completions.parse(
        model=model, messages=msgs, response_format=DocOutput
    )

    # Expect exactly one function call in the first choice
    doc_out = completion.choices[0].message.parsed

    summary["thought_process"] = doc_out.thought_process
    summary["code_suggestion"] = doc_out.code_suggestion
    summary["difficult_elements"] = doc_out.difficult_elements
    summary["test_case"] = doc_out.test_case

    print("** Code suggestion:\n", doc_out.code_suggestion)
    print("** Thought process:\n", doc_out.thought_process)

    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[update] Wrote enriched data to {summary_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", help="Submission ID to enrich (folder name)")
    ap.add_argument("--model", default="o3")
    ap.add_argument(
        "--api-key", default=os.getenv("OPENAI_API_KEY"), help="OpenAI key if not in env"
    )
    ap.add_argument("--force", action="store_true", help="overwrite existing enrichment")
    args = ap.parse_args()

    if not args.api_key:
        raise SystemExit("OPENAI_API_KEY not provided")

    global FORCE
    FORCE = args.force

    if args.submission:
        summary_path = EVAL_DIR / args.submission / "summary.json"
        if not summary_path.exists():
            raise SystemExit("summary.json not found for submission")
        enrich_summary(summary_path, args.api_key, args.model)
    else:
        for summary_path in EVAL_DIR.glob("*/summary.json"):
            enrich_summary(summary_path, args.api_key, args.model)


if __name__ == "__main__":
    main()
