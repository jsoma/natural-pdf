from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SNAPSHOT_PATH = (
    Path(__file__).with_suffix("").parent / "snapshots" / "element_manager_snapshots.json"
)


@dataclass(frozen=True)
class RealPDFCase:
    name: str
    path: Path
    page_index: int = 0

    @property
    def absolute_path(self) -> Path:
        return (REPO_ROOT / self.path).resolve()


REAL_PDF_CASES: List[RealPDFCase] = [
    RealPDFCase("practice_page_0", Path("pdfs/01-practice.pdf")),
    RealPDFCase("shapes_page_0", Path("pdfs/shapes-and-text.pdf")),
    RealPDFCase("arabic_page_0", Path("pdfs/arabic.pdf")),
    RealPDFCase("needs_ocr_page_0", Path("pdfs/needs-ocr.pdf")),
]


def _round_number(value: Any, places: int = 4) -> float:
    try:
        return round(float(value), places)
    except (TypeError, ValueError):
        return float("nan")


def _element_signature(element: Any) -> Dict[str, Any]:
    signature: Dict[str, Any] = {}
    bbox = getattr(element, "bbox", None)
    if bbox:
        signature["bbox"] = [_round_number(coord, 4) for coord in bbox]
    size = getattr(element, "size", None)
    if size is not None:
        signature["size"] = _round_number(size, 4)
    for attr in ("object_type", "source"):
        value = getattr(element, attr, None)
        if value is not None:
            signature[attr] = value
    text = getattr(element, "text", None)
    if text is not None:
        if len(text) > 80:
            text = text[:77] + "..."
        signature["text"] = text
    return signature


def _elements_digest(elements: Iterable[Any], limit: int = 40) -> str:
    subset = [_element_signature(element) for element in list(elements)[:limit]]
    payload = json.dumps(subset, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _summarize_elements(elements: List[Any]) -> Dict[str, Any]:
    return {
        "count": len(elements),
        "digest": _elements_digest(elements),
        "samples": [_element_signature(element) for element in elements[:5]],
    }


def capture_case(case: RealPDFCase) -> Dict[str, Any]:
    from natural_pdf import PDF

    pdf = PDF(str(case.absolute_path))
    try:
        page = pdf.pages[case.page_index]
        manager = page._element_mgr  # noqa: SLF001 - intentional internal access for testing
        manager.load_elements()
        elements = {
            "chars": manager.chars,
            "words": manager.words,
            "rects": manager.rects,
            "lines": manager.lines,
            "images": manager.images,
        }
        summary = {
            "pdf": str(case.path),
            "page_index": case.page_index,
            "page_bbox": [
                _round_number(0.0),
                _round_number(0.0),
                _round_number(page.width),
                _round_number(page.height),
            ],
            "elements": {name: _summarize_elements(items) for name, items in elements.items()},
        }
        return summary
    finally:
        pdf.close()


def load_snapshots(path: Path = DEFAULT_SNAPSHOT_PATH) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Snapshot file {path} is missing. Run `python -m tests.realpdf.snapshot_utils --update` first."
        )
    return json.loads(path.read_text())


def write_snapshots(path: Path = DEFAULT_SNAPSHOT_PATH) -> Dict[str, Any]:
    data = {case.name: capture_case(case) for case in REAL_PDF_CASES}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True))
    return data


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Update ElementManager real-PDF snapshots.")
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_SNAPSHOT_PATH, help="Snapshot file path."
    )
    args = parser.parse_args()
    updated = write_snapshots(args.output)
    print(f"Wrote {len(updated)} snapshots to {args.output}")
