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
import secrets
import shutil
import stat
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Union

from tqdm.auto import tqdm

from natural_pdf.utils.filesystem import (
    PathIdentity,
    path_has_identity,
    path_identity,
    publish_directory_noreplace,
    remove_private_directory,
)
from natural_pdf.utils.identifiers import generate_short_path_hash

if TYPE_CHECKING:
    from natural_pdf.core.pdf import PDF
    from natural_pdf.core.pdf_collection import PDFCollection

logger = logging.getLogger(__name__)

# Marker file identifying a directory as produced by this exporter, so that
# overwrite=True never deletes a directory it didn't create.
_EXPORT_MARKER = ".natural-pdf-export"


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
    output_dir: Union[str, os.PathLike],
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
    # ── validate (before touching any existing export) ──────────────────
    # Accept str or pathlib.Path destinations; everything below builds
    # sibling paths via string operations, so normalize up front.
    output_dir = os.fspath(output_dir)

    if output_format not in ("jsonl", "csv"):
        raise ValueError(f"output_format must be 'jsonl' or 'csv', got {output_format!r}")

    if split is not None and not (0.0 < split < 1.0):
        raise ValueError(f"split must be between 0 and 1 (exclusive), got {split}")

    destination_identity: Optional[PathIdentity] = None
    if os.path.lexists(output_dir):
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists: {output_dir}. "
                "Pass overwrite=True to replace it."
            )
        destination_identity = _validate_existing_destination(output_dir)

    pdfs = _resolve_source_pdfs(source)
    if not pdfs:
        logger.warning("No PDFs provided — nothing to export.")
        return {"images": 0, "skipped": 0, "output_dir": output_dir}

    # ── build into a staging directory ──────────────────────────────────
    # The whole export is assembled in a sibling staging directory and only
    # swapped into place after complete success, so a failure partway through
    # never destroys a previous export at output_dir.
    output_path = Path(output_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_directory_mode = _normal_directory_mode(output_path.parent)
    staging_dir = _create_owned_directory(
        output_path.parent,
        prefix=f"{output_path.name}.staging-",
        mode=0o700,
    )

    try:
        result = _build_export(
            pdfs,
            staging_dir,
            selector=selector,
            prompt=prompt,
            resolution=resolution,
            padding=padding,
            output_format=output_format,
            split=split,
            random_seed=random_seed,
            include_metadata=include_metadata,
        )
    except BaseException:
        _cleanup_private_directory(staging_dir)
        raise

    # ── revalidate the destination before promotion (TOCTOU guard) ───────
    # The destination was checked before the build, but the build takes time:
    # another process may have created data at output_dir in the meantime.
    # Promotion must never move aside or delete anything that fails the same
    # rules used up front; on violation the fully-built staging directory is
    # left in place and the foreign destination is not touched.
    _ensure_destination_still_safe(
        output_dir,
        overwrite=overwrite,
        staging_dir=staging_dir,
        expected_identity=destination_identity,
    )

    if result["images"] == 0:
        # Nothing exported: never replace an existing export with an empty
        # one. For a fresh destination, keep the (marker-only) directory so
        # behavior matches a normal empty export.
        if destination_identity is not None:
            _cleanup_private_directory(staging_dir)
        else:
            _publish_new_destination(
                staging_dir,
                output_dir,
                final_mode=final_directory_mode,
            )
        return {**result, "output_dir": output_dir}

    # ── swap staging into place ──────────────────────────────────────────
    # Move any existing export aside (rather than deleting it) before
    # promoting staging, so a failed promote can restore the original
    # instead of losing both directories.
    aside_container: Optional[str] = None
    old_aside: Optional[str] = None
    if destination_identity is not None:
        aside_container = tempfile.mkdtemp(
            prefix=f"{output_path.name}.old-",
            suffix=f"-{secrets.token_hex(16)}",
            dir=output_path.parent,
        )
        old_aside = os.path.join(aside_container, "previous-export")
        try:
            if not path_has_identity(output_dir, destination_identity):
                raise FileExistsError(
                    f"Refusing to overwrite {output_dir}: it was replaced during promotion. "
                    f"The completed export was left at {staging_dir}."
                )
            os.rename(output_dir, old_aside)
            _validate_claimed_destination(
                old_aside,
                output_dir,
                destination_identity,
                staging_dir,
            )
        except BaseException:
            if aside_container is not None and not os.path.lexists(old_aside):
                _cleanup_private_directory(aside_container, recursive=False)
            raise
    try:
        _publish_new_destination(
            staging_dir,
            output_dir,
            final_mode=final_directory_mode,
        )
    except BaseException:
        if old_aside is not None:
            try:
                publish_directory_noreplace(old_aside, output_dir)
            except OSError as restore_exc:  # pragma: no cover - filesystem race
                logger.error(
                    f"Failed to restore previous export from {old_aside} to "
                    f"{output_dir}: {restore_exc}. The original data is still "
                    f"at {old_aside}."
                )
            else:
                if aside_container is not None:
                    _cleanup_private_directory(aside_container, recursive=False)
        # Publication did not consume staging. Preserve the completed export
        # for recovery instead of deleting good data after a filesystem race.
        raise
    if old_aside is not None:
        if not _cleanup_private_directory(old_aside):
            logger.warning("Previous export was preserved at %s", old_aside)
        if aside_container is not None:
            _cleanup_private_directory(aside_container, recursive=False)

    logger.info(
        f"Exported {result['images']} training images to '{output_dir}' "
        f"(skipped {result['skipped']}, format={output_format}"
        f"{f', split={split}' if split else ''})."
    )
    return {**result, "output_dir": output_dir}


def _build_export(
    pdfs: List["PDF"],
    output_dir: str,
    *,
    selector: Optional[str],
    prompt: str,
    resolution: int,
    padding: int,
    output_format: str,
    split: Optional[float],
    random_seed: int,
    include_metadata: bool,
) -> dict:
    """Render crops and write metadata into *output_dir* (a staging directory).

    Returns the summary dict with ``output_dir`` pointing at the build
    directory; the caller is responsible for swapping it into its final
    location.
    """
    # ── collect records ─────────────────────────────────────────────────
    records: List[Dict[str, Any]] = []
    skipped = 0

    # We'll write images to a temporary flat list first, then move to
    # the correct split directories at the end.
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, _EXPORT_MARKER), "w", encoding="utf-8") as marker_fh:
        marker_fh.write("Created by natural_pdf.exporters.training_data.export_training_data\n")
    tmp_images_dir = os.path.join(output_dir, "_tmp_images")
    os.mkdir(tmp_images_dir, mode=0o700)

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
        _cleanup_private_directory(tmp_images_dir)
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

        # Copy image files into this split's images/ directory. Copy (not move)
        # so a mid-write failure leaves _tmp_images/ intact and recoverable;
        # the temp directory is removed only after all splits are written.
        for rec in split_records:
            src = rec["_abs_image_path"]
            dst = os.path.join(images_dir, os.path.basename(src))
            shutil.copy2(src, dst)

        # Write metadata file
        if output_format == "jsonl":
            _write_jsonl(split_dir, split_records, prompt, include_metadata)
        else:
            _write_csv(split_dir, split_records, include_metadata)

    # Clean up temp images dir
    _cleanup_private_directory(tmp_images_dir)

    return {"images": len(records), "skipped": skipped, "output_dir": output_dir}


# ── helpers ─────────────────────────────────────────────────────────────


def _ensure_destination_still_safe(
    output_dir: str,
    *,
    overwrite: bool,
    staging_dir: str,
    expected_identity: Optional[PathIdentity],
) -> None:
    """Re-apply the pre-build destination rules immediately before promotion.

    Raises ``FileExistsError`` (leaving both the destination and the staging
    directory untouched) when the destination no longer satisfies the rules
    that were checked before the build started.
    """
    if expected_identity is None:
        if not os.path.lexists(output_dir):
            return
        raise FileExistsError(
            f"Output destination appeared at {output_dir} while the export was "
            "being built. It has not been touched. The "
            f"completed export was left at {staging_dir}; move it into place "
            "manually or retry after resolving the destination."
        )
    if not overwrite or not path_has_identity(output_dir, expected_identity):
        raise FileExistsError(
            f"Refusing to overwrite {output_dir}: it changed while the export was "
            f"being built. It has not been touched. The completed export was left "
            f"at {staging_dir}."
        )


def _validate_existing_destination(output_dir: str) -> PathIdentity:
    """Validate overwrite safety against one stable directory identity."""

    try:
        initial_stat = os.lstat(output_dir)
    except OSError as exc:
        raise FileExistsError(
            f"Refusing to overwrite {output_dir}: it changed during validation."
        ) from exc
    if stat.S_ISLNK(initial_stat.st_mode) or not stat.S_ISDIR(initial_stat.st_mode):
        raise FileExistsError(
            f"Refusing to overwrite {output_dir}: the destination is not a directory."
        )

    identity = PathIdentity(initial_stat.st_dev, initial_stat.st_ino)
    entries = os.listdir(output_dir)

    if not path_has_identity(output_dir, identity):
        raise FileExistsError(f"Refusing to overwrite {output_dir}: it changed during validation.")

    # Refuse to delete a directory this exporter didn't create: overwrite is
    # meant to replace a previous export, not arbitrary user data.
    if entries and _EXPORT_MARKER not in entries:
        raise FileExistsError(
            f"Refusing to overwrite {output_dir}: it is not empty and does not "
            f"look like a previous export (missing {_EXPORT_MARKER}). "
            "Delete it manually if you really want to replace it."
        )
    return identity


def _validate_claimed_destination(
    old_aside: str,
    output_dir: str,
    expected_identity: PathIdentity,
    staging_dir: str,
) -> None:
    """Revalidate the claimed old root, restoring it on every failure."""

    try:
        claimed_identity = _validate_existing_destination(old_aside)
        if claimed_identity != expected_identity:
            raise FileExistsError("A different destination object was claimed")
    except BaseException as validation_error:
        if _restore_displaced_destination(old_aside, output_dir):
            raise FileExistsError(
                f"Refusing to overwrite {output_dir}: the claimed destination no longer "
                f"satisfies the overwrite rules. The completed export was left at "
                f"{staging_dir}."
            ) from validation_error
        raise RuntimeError(
            f"Could not restore the destination after validation failed. Its data is "
            f"preserved at {old_aside}; the completed export is at {staging_dir}."
        ) from validation_error


def _publish_new_destination(
    staging_dir: str,
    output_dir: str,
    *,
    final_mode: int,
) -> None:
    """Publish *staging_dir* without ever replacing an existing destination."""

    try:
        # Staging remains private while data is being assembled. Expose normal
        # os.makedirs/umask-derived permissions only once it is complete and
        # immediately before publication.
        os.chmod(staging_dir, final_mode)
        publish_directory_noreplace(staging_dir, output_dir)
    except FileExistsError as exc:
        raise FileExistsError(
            f"Output destination appeared at {output_dir} during promotion and was not "
            f"touched. The completed export was left at {staging_dir}."
        ) from exc


def _restore_displaced_destination(old_aside: str, output_dir: str) -> bool:
    """Best-effort restoration after detecting that the wrong object was moved."""

    try:
        publish_directory_noreplace(old_aside, output_dir)
    except OSError as restore_exc:
        logger.error(
            f"Could not restore concurrently replaced destination from {old_aside} "
            f"to {output_dir}: {restore_exc}. No data was deleted."
        )
        return False
    return True


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
    # Column names follow the library's pdfplumber-style coordinate vocabulary:
    # the vertical values are top/bottom (distance from page top), not y0/y1.
    meta_fields = ["source_pdf", "page", "x0", "top", "x1", "bottom"] if include_metadata else []
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
                row["x0"], row["top"], row["x1"], row["bottom"] = bbox
            writer.writerow(row)


def _create_owned_directory(parent: Path, *, prefix: str, mode: int) -> str:
    """Create an exclusive full-entropy directory with *mode*."""

    for _attempt in range(100):
        candidate = parent / f"{prefix}{secrets.token_hex(16)}"
        try:
            os.mkdir(candidate, mode=mode)
        except FileExistsError:
            continue
        return os.fspath(candidate)
    raise FileExistsError(f"Could not reserve a unique staging directory under {parent}")


def _normal_directory_mode(parent: Path) -> int:
    """Measure the mode ``os.makedirs`` would apply under the current umask."""

    probe = _create_owned_directory(
        parent,
        prefix=".natural-pdf-mode-probe-",
        mode=0o777,
    )
    probe_mode = os.lstat(probe).st_mode & 0o777
    _cleanup_private_directory(probe, recursive=False)
    return probe_mode


def _cleanup_private_directory(
    path: str,
    *,
    recursive: bool = True,
) -> bool:
    """Clean a uniquely named private directory without masking export results."""

    removed = remove_private_directory(path, recursive=recursive)
    if not removed:
        logger.warning("Could not clean private export directory: %s", path)
    return removed
