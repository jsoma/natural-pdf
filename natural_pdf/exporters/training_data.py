"""
Model-agnostic training data exporter for OCR fine-tuning.

Produces a HuggingFace ImageFolder-compatible directory with cropped text images
and metadata in JSONL (ShareGPT conversation format) or CSV.

Output is directly usable with:
- ``datasets.load_dataset("imagefolder", data_dir=...)``
- Unsloth / TRL VLM fine-tuning (via the ``conversations`` field)
- Any OCR framework that accepts image + ground-truth text pairs
"""

from __future__ import annotations

import csv
import json
import logging
import os
import random
import shutil
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Union

from tqdm.auto import tqdm

from natural_pdf.utils.identifiers import generate_short_path_hash

if TYPE_CHECKING:
    from natural_pdf.core.pdf import PDF
    from natural_pdf.core.pdf_collection import PDFCollection

logger = logging.getLogger(__name__)


def _resolve_source_pdfs(
    source: Union["PDF", "PDFCollection", List["PDF"]],
) -> List["PDF"]:
    """Normalize *source* to a flat list of PDF objects."""
    from natural_pdf.core.pdf import PDF
    from natural_pdf.core.pdf_collection import PDFCollection

    if isinstance(source, PDF):
        return [source]
    if isinstance(source, PDFCollection):
        return list(source.pdfs)
    if isinstance(source, list) and all(isinstance(p, PDF) for p in source):
        return list(source)
    raise TypeError(
        f"Unsupported source type: {type(source)}. Must be PDF, PDFCollection, or List[PDF]."
    )


def export_training_data(
    source: Union["PDF", "PDFCollection", List["PDF"]],
    output_dir: str,
    *,
    selector: Optional[str] = "text",
    prompt: str = "OCR this image. Return only the exact text.",
    resolution: int = 150,
    padding: int = 2,
    output_format: Literal["jsonl", "csv"] = "jsonl",
    overwrite: bool = False,
    split: Optional[float] = None,
    random_seed: int = 42,
    include_metadata: bool = True,
) -> dict:
    """Export cropped text-element images and labels for OCR model training.

    Args:
        source: One or more PDFs to export from.
        output_dir: Destination directory (created if needed).
        selector: CSS-like selector for which elements to crop (default ``"text"``).
        prompt: Instruction string used in the ``conversations`` field.
        resolution: Render DPI for crop images.
        padding: Points of padding around each element bbox.
        output_format: ``"jsonl"`` (ShareGPT + HF ImageFolder) or ``"csv"``.
        overwrite: If *False* and *output_dir* already exists, raise ``FileExistsError``.
        split: Train/validation split ratio (e.g. ``0.9`` for 90 % train).
            *None* means no split.
        random_seed: Seed for reproducible train/val shuffling.
        include_metadata: Include source PDF path, page number, and bbox in output.

    Returns:
        Summary dict: ``{"images": N, "skipped": M, "output_dir": path}``.
    """
    # ── validate ────────────────────────────────────────────────────────
    if os.path.exists(output_dir):
        if overwrite:
            shutil.rmtree(output_dir)
        else:
            raise FileExistsError(
                f"Output directory already exists: {output_dir}. "
                "Pass overwrite=True to replace it."
            )

    pdfs = _resolve_source_pdfs(source)
    if not pdfs:
        logger.warning("No PDFs provided — nothing to export.")
        return {"images": 0, "skipped": 0, "output_dir": output_dir}

    if split is not None and not (0.0 < split < 1.0):
        raise ValueError(f"split must be between 0 and 1 (exclusive), got {split}")

    # ── collect records ─────────────────────────────────────────────────
    records: List[Dict[str, Any]] = []
    skipped = 0

    # We'll write images to a temporary flat list first, then move to
    # the correct split directories at the end.
    os.makedirs(output_dir, exist_ok=True)
    tmp_images_dir = os.path.join(output_dir, "_tmp_images")
    os.makedirs(tmp_images_dir, exist_ok=True)

    for pdf in tqdm(pdfs, desc="Processing PDFs", disable=len(pdfs) == 1):
        if not hasattr(pdf, "path") or not isinstance(pdf.path, str):
            logger.warning(f"Skipping PDF without a valid path: {pdf}")
            continue

        pdf_hash = generate_short_path_hash(pdf.path)
        elements = pdf.find_all(selector or "text", apply_exclusions=False)

        if not elements:
            logger.debug(f"No elements matching '{selector}' in {pdf.path}")
            continue

        for i, element in enumerate(
            tqdm(elements, desc=f"Exporting '{os.path.basename(pdf.path)}'", leave=False)
        ):
            # ── skip invalid elements ───────────────────────────────
            text = getattr(element, "text", None)
            if not text or not isinstance(text, str) or "\n" in text:
                skipped += 1
                continue

            page_index = getattr(element.page, "index", 0)
            image_filename = f"{pdf_hash}_p{page_index}_e{i}.png"
            image_path = os.path.join(tmp_images_dir, image_filename)

            try:
                region = element.expand(padding)
                img = region.render(resolution=resolution, crop=True)
                img.save(image_path, "PNG")
            except Exception as exc:
                logger.warning(
                    f"Failed to render element {i} on page {page_index} of {pdf.path}: {exc}"
                )
                skipped += 1
                continue

            record: Dict[str, Any] = {
                "file_name": f"images/{image_filename}",
                "text": text,
                "_abs_image_path": image_path,  # internal, stripped before writing
            }
            if include_metadata:
                bbox = [
                    float(element.x0),
                    float(element.top),
                    float(element.x1),
                    float(element.bottom),
                ]
                record["metadata"] = {
                    "source_pdf": os.path.basename(pdf.path),
                    "page": page_index,
                    "bbox": bbox,
                }
            records.append(record)

    if not records:
        # Clean up temp dir
        _rmdir_safe(tmp_images_dir)
        logger.warning("No elements were exported.")
        return {"images": 0, "skipped": skipped, "output_dir": output_dir}

    # ── split ───────────────────────────────────────────────────────────
    rng = random.Random(random_seed)
    rng.shuffle(records)

    if split is not None:
        split_idx = int(len(records) * split)
        splits: Dict[str, List[Dict[str, Any]]] = {
            "train": records[:split_idx],
            "validation": records[split_idx:],
        }
    else:
        splits = {"": records}  # empty key → root level

    # ── write output ────────────────────────────────────────────────────
    for split_name, split_records in splits.items():
        if not split_records:
            continue

        if split_name:
            split_dir = os.path.join(output_dir, split_name)
        else:
            split_dir = output_dir

        images_dir = os.path.join(split_dir, "images")
        os.makedirs(images_dir, exist_ok=True)

        # Move image files into this split's images/ directory
        for rec in split_records:
            src = rec.pop("_abs_image_path")
            dst = os.path.join(images_dir, os.path.basename(src))
            shutil.move(src, dst)

        # Write metadata file
        if output_format == "jsonl":
            _write_jsonl(split_dir, split_records, prompt, include_metadata)
        else:
            _write_csv(split_dir, split_records, include_metadata)

    # Clean up temp images dir
    _rmdir_safe(tmp_images_dir)

    total_images = len(records)
    logger.info(
        f"Exported {total_images} training images to '{output_dir}' "
        f"(skipped {skipped}, format={output_format}"
        f"{f', split={split}' if split else ''})."
    )
    return {"images": total_images, "skipped": skipped, "output_dir": output_dir}


# ── helpers ─────────────────────────────────────────────────────────────


def _write_jsonl(
    directory: str,
    records: List[Dict[str, Any]],
    prompt: str,
    include_metadata: bool,
) -> None:
    path = os.path.join(directory, "metadata.jsonl")
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            entry: Dict[str, Any] = {
                "file_name": rec["file_name"],
                "text": rec["text"],
                "conversations": [
                    {"role": "user", "content": f"<image>\n{prompt}"},
                    {"role": "assistant", "content": rec["text"]},
                ],
            }
            if include_metadata and "metadata" in rec:
                entry["metadata"] = rec["metadata"]
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _write_csv(
    directory: str,
    records: List[Dict[str, Any]],
    include_metadata: bool,
) -> None:
    path = os.path.join(directory, "metadata.csv")
    base_fields = ["file_name", "text"]
    meta_fields = ["source_pdf", "page", "x0", "y0", "x1", "y1"] if include_metadata else []
    fieldnames = base_fields + meta_fields

    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            row: Dict[str, Any] = {
                "file_name": rec["file_name"],
                "text": rec["text"],
            }
            if include_metadata and "metadata" in rec:
                meta = rec["metadata"]
                row["source_pdf"] = meta.get("source_pdf", "")
                row["page"] = meta.get("page", "")
                bbox = meta.get("bbox", [0, 0, 0, 0])
                row["x0"], row["y0"], row["x1"], row["y1"] = bbox
            writer.writerow(row)


def _rmdir_safe(path: str) -> None:
    """Remove a directory if it exists, ignoring errors."""
    try:
        if os.path.isdir(path):
            shutil.rmtree(path)
    except OSError:
        pass
