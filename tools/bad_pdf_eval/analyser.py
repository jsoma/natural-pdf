from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image
from rich.console import Console
from rich.table import Table

import natural_pdf as npdf

from .reporter import log_section, save_json
from .utils import slugify

console = Console()


class BadPDFAnalyzer:
    """Run a battery of Natural-PDF probes on a PDF and dump artefacts."""

    def __init__(
        self,
        pdf_path: Path,
        output_dir: Path,
        submission_meta: Dict[str, Any],
        pages: List[int],
        resolution: int = 216,
    ):
        self.pdf_path = pdf_path
        self.output_dir = output_dir
        self.meta = submission_meta
        self.pages_to_analyze = pages
        self.resolution = resolution

    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------
    def _save_page_image(self, page, page_num: int) -> Path:
        """Render and save page image as high-quality JPG."""
        img: Image.Image = page.render(resolution=self.resolution)
        if img.mode != "RGB":
            img = img.convert("RGB")
        img_path = self.output_dir / f"page_{page_num:04d}.jpg"
        img.save(img_path, "JPEG", quality=90, optimize=True, progressive=True)
        return img_path

    # ------------------------------------------------------------------
    def run(self) -> Dict[str, Any]:
        """Return master JSON summary (also persisted inside output_dir)."""
        console.print(f"[green]Analyzing[/] {self.pdf_path.name}  ↦  pages {self.pages_to_analyze}")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        pdf = npdf.PDF(str(self.pdf_path))
        summary: Dict[str, Any] = {
            "submission_id": self.meta["Submission ID"],
            "pdf": str(self.pdf_path),
            "total_pages": len(pdf.pages),
            "pages": [],
            "goal": self.meta.get("What are we trying to get out of the PDF?", ""),
            "language": self.meta.get("What language(s) or script is the content in?", ""),
            "issues": self.meta.get("What do you think makes this PDF bad?", ""),
            "description": self.meta.get("What is the PDF, and/or where did it come from?", ""),
        }

        for page_idx_1based in self.pages_to_analyze:
            if page_idx_1based < 1 or page_idx_1based > len(pdf.pages):
                console.print(f"[yellow]- skipping page {page_idx_1based} (out of range)")
                continue
            page = pdf.pages[page_idx_1based - 1]
            page_result: Dict[str, Any] = {"page_number": page_idx_1based}
            difficulties: List[str] = []

            # ---------------- image
            img_path = self._save_page_image(page, page_idx_1based)
            page_result["image"] = str(img_path)

            # ---------------- describe / inspect
            try:
                descr = page.describe()
                page_result["describe"] = str(descr)
            except Exception as e:
                page_result["describe_error"] = str(e)

            try:
                page_result["inspect"] = str(page.inspect(limit=30))
            except Exception as e:
                page_result["inspect_error"] = str(e)

            # ---------------- extract text
            text = ""
            try:
                text = page.extract_text()
                page_result["text_len"] = len(text or "")
                if text:
                    page_result["text_preview"] = text[:300]
            except Exception as e:
                page_result["extract_text_error"] = str(e)

            # ---------------- tiny font detection
            try:
                words_sample = page.words[:5000]  # not expensive
                if words_sample:
                    small_fonts = sum(1 for w in words_sample if getattr(w, "size", 10) < 4)
                    ratio = small_fonts / len(words_sample)
                    page_result["tiny_font_ratio"] = round(ratio, 3)
                    if ratio >= 0.2:
                        difficulties.append("tiny_font")
            except Exception:
                pass

            # ---------------- extract table simple
            try:
                table_data = page.extract_table()
                if table_data and table_data[0]:
                    page_result["table_found"] = True
                    page_result["table_dims"] = [len(table_data), len(table_data[0])]
                else:
                    page_result["table_found"] = False
            except Exception as e:
                page_result["table_error"] = str(e)

            # ---------------- layout YOLO
            try:
                yolo_layout = page.analyze_layout()
                page_result["layout_yolo_count"] = len(yolo_layout)
                page_result["layout_yolo_regions"] = [
                    {
                        "type": getattr(r, "type", "unknown"),
                        "bbox": [r.x0, r.top, r.x1, r.bottom],
                        "confidence": getattr(r, "confidence", None),
                    }
                    for r in yolo_layout
                ]
            except Exception as e:
                page_result["layout_yolo_error"] = str(e)

            # ---------------- layout TATR for tables
            try:
                tatr_layout = page.analyze_layout("tatr", existing="append")
                page_result["layout_tatr_count"] = len(tatr_layout)
                page_result["layout_tatr_regions"] = [
                    {
                        "type": getattr(r, "type", "unknown"),
                        "bbox": [r.x0, r.top, r.x1, r.bottom],
                        "confidence": getattr(r, "confidence", None),
                    }
                    for r in tatr_layout
                ]
            except Exception as e:
                page_result["layout_tatr_error"] = str(e)

            # ---------------- color blob detection (rect fills / graphical anchors)
            try:
                blobs = page.detect_blobs()
                page_result["blob_count"] = len(blobs)
                page_result["blobs_sample"] = [
                    {
                        "color": getattr(b, "color", None),
                        "bbox": [b.x0, b.top, b.x1, b.bottom],
                    }
                    for b in blobs[:20]
                ]
            except Exception as e:
                page_result["blobs_error"] = str(e)

            # ---------------- OCR pass (only if little native text)
            ocr_elements = []
            if page_result.get("text_len", 0) < 100:
                start = time.time()
                try:
                    ocr_elements = page.extract_ocr_elements(engine="easyocr")
                    page_result["ocr_text_elements"] = len(ocr_elements)
                    page_result["ocr_runtime_sec"] = round(time.time() - start, 2)
                    # Embed small OCR preview instead of separate file
                    ocr_json = [
                        {
                            "text": el.text,
                            "bbox": [el.x0, el.top, el.x1, el.bottom],
                            "size": getattr(el, "size", None),
                        }
                        for el in ocr_elements[:500]
                    ]
                    page_result["ocr_sample"] = ocr_json[:30]
                except Exception as e:
                    page_result["ocr_error"] = str(e)
            else:
                page_result["ocr_text_elements"] = 0

            # ---------------- tags – handle non-string entries (NaN etc.)
            goal_raw = summary.get("goal", "")
            # Convert to string to avoid attribute errors if the CSV cell is NaN/float
            goal_str = str(goal_raw) if goal_raw is not None else ""
            goal = goal_str.lower()

            if "table" in goal:
                page_result["goal_tag"] = "table_extraction"
            elif any(word in goal for word in ["text", "content", "information"]):
                page_result["goal_tag"] = "text_extraction"
            else:
                page_result["goal_tag"] = "unknown"

            # Difficulties determination
            if (
                page_result.get("text_len", 0) < 100
                and page_result.get("ocr_text_elements", 0) > 20
            ):
                difficulties.append("scanned_image")

            page_result["difficulties"] = difficulties

            # Suggested approach heuristic
            approach = []
            if "table" in goal:
                if page_result.get("layout_tatr_count", 0) > 0:
                    approach.append("Crop TATR regions → extract_table('tatr')")
                else:
                    approach.append("Anchor header text, .below(), extract_table(custom settings)")
            if "text" in goal and "scanned_image" in difficulties:
                approach.append("Apply OCR (paddle for non-Latin)")
            if "tiny_font" in difficulties:
                approach.append("Re-render at higher scale or adjust char_margin")
            page_result["suggested_approach"] = "; ".join(approach)

            # ---------------- code snippet suggestion
            def _first_anchor_from_goal(g: str) -> str:
                """Pick a plausible anchor token (capitalised word) from the free-form goal text."""
                for tok in g.split():
                    t = tok.strip().strip(".;:,()[]{}")
                    if len(t) > 3 and t[0].isupper() and t.isalpha():
                        return t
                return "AnchorText"

            import_lines = [
                "from natural_pdf import PDF",
            ]
            if page_result["goal_tag"] == "table_extraction":
                import_lines.append("import pandas as pd")

            code_lines: List[str] = import_lines + [
                f'pdf = PDF("{self.pdf_path}")',
                f"page = pdf.pages[{page_idx_1based - 1}]  # page {page_idx_1based}",
            ]

            thought_lines: List[str] = []
            # build reasoning
            thought_lines.append(
                f"Goal tag: {page_result['goal_tag']}. Detected difficulties: {', '.join(difficulties) or 'none'}."
            )

            if page_result["goal_tag"] == "table_extraction":
                thought_lines.append(
                    "Plan: rely on layout models to locate tables, then extract with Natural-PDF helper."
                )
                if page_result.get("layout_tatr_count", 0) > 0:
                    code_lines.append("page.analyze_layout('tatr')  # adds 'table' regions")
                else:
                    code_lines.append("page.analyze_layout()  # YOLO fallback")

                if page_result.get("layout_tatr_count", 0) > 1:
                    thought_lines.append(
                        "Multiple tables detected, choose second as goal mentions 'second table'."
                    )
                    code_lines.append("tables = page.find_all('table')")
                    code_lines.append("tbl = tables[1]")
                else:
                    code_lines.append("tbl = page.find('table')  # first table")

                code_lines.extend(
                    [
                        "data = tbl.extract_table()",
                        "columns, rows = data[0], data[1:]",
                        "df = pd.DataFrame(rows, columns=columns)",
                    ]
                )
            elif page_result["goal_tag"] == "text_extraction":
                anchor = _first_anchor_from_goal(goal_str)
                if "scanned_image" in difficulties:
                    thought_lines.append("No native text detected; need OCR before querying.")
                    code_lines.append("page.apply_ocr(engine='paddle')")
                thought_lines.append(f"Anchor on text '{anchor}' then read below region.")
                code_lines.append(f'section = page.find("text:contains({anchor})").below(0, 50)')
                code_lines.append("text = section.extract_text()")
            else:
                thought_lines.append("Goal unclear; placeholder snippet provided.")
                code_lines.append("# TODO: clarify extraction goal")

            page_result["code_suggestion"] = "\n".join(code_lines)
            page_result["thought_process"] = " ".join(thought_lines)

            summary["pages"].append(page_result)

            # Provide quick heuristic comment
            if page_result.get("text_len", 0) == 0 and page_result.get("ocr_text_elements", 0) > 20:
                page_result["auto_comment"] = "Likely scanned/needs OCR; no native text."
            elif (
                page_result.get("text_len", 0) > 1000
                and page_result.get("layout_yolo_count", 0) == 0
            ):
                page_result["auto_comment"] = (
                    "Native dense text; YOLO found no regions – may be fine, fonts just small."
                )
            else:
                page_result.setdefault("auto_comment", "")

        # Save master summary
        save_json(summary, self.output_dir / "summary.json")
        return summary


# -------------------------------------------------------------------------
# Helper to parse specific pages mentioned in free text
# -------------------------------------------------------------------------
PAGE_REGEX = re.compile(r"page\s*(\d{1,4})", re.IGNORECASE)


def extract_page_hints(text: str) -> List[int]:
    return [int(m.group(1)) for m in PAGE_REGEX.finditer(text)]
