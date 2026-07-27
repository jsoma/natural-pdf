"""Tests for the model-agnostic training data exporter."""

import csv
import json
import os
import tempfile
from pathlib import Path

import pytest

from natural_pdf.core.pdf import PDF
from natural_pdf.exporters.training_data import export_training_data

TEST_PDF = "pdfs/01-practice.pdf"


@pytest.fixture
def pdf():
    p = PDF(TEST_PDF)
    yield p
    p.close()


@pytest.fixture
def out_dir():
    with tempfile.TemporaryDirectory() as d:
        yield os.path.join(d, "export")


# ── basic JSONL export ──────────────────────────────────────────────────


def test_default_jsonl_export(pdf, out_dir):
    """Default export produces images/ and metadata.jsonl with correct structure."""
    result = export_training_data(pdf, out_dir)

    assert result["images"] > 0
    assert result["output_dir"] == out_dir

    images_dir = Path(out_dir) / "images"
    assert images_dir.is_dir()
    png_files = list(images_dir.glob("*.png"))
    assert len(png_files) == result["images"]

    jsonl_path = Path(out_dir) / "metadata.jsonl"
    assert jsonl_path.exists()

    with open(jsonl_path, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f if line.strip()]

    assert len(lines) == result["images"]

    # Check first record structure
    rec = lines[0]
    assert "file_name" in rec
    assert "text" in rec
    assert "conversations" in rec
    assert rec["file_name"].startswith("images/")
    assert rec["file_name"].endswith(".png")

    convos = rec["conversations"]
    assert len(convos) == 2
    assert convos[0]["role"] == "user"
    assert "<image>" in convos[0]["content"]
    assert convos[1]["role"] == "assistant"
    assert convos[1]["content"] == rec["text"]

    # Metadata present by default
    assert "metadata" in rec
    meta = rec["metadata"]
    assert "source_pdf" in meta
    assert "page" in meta
    assert "bbox" in meta
    assert len(meta["bbox"]) == 4


# ── CSV export ──────────────────────────────────────────────────────────


def test_csv_export(pdf, out_dir):
    """CSV format produces metadata.csv with correct columns."""
    result = export_training_data(pdf, out_dir, output_format="csv")

    csv_path = Path(out_dir) / "metadata.csv"
    assert csv_path.exists()

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) == result["images"]
    assert "file_name" in rows[0]
    assert "text" in rows[0]
    # Metadata columns present by default
    assert "source_pdf" in rows[0]
    assert "page" in rows[0]
    assert "x0" in rows[0]


def test_csv_no_metadata(pdf, out_dir):
    """CSV without metadata has only file_name and text columns."""
    export_training_data(pdf, out_dir, output_format="csv", include_metadata=False)

    csv_path = Path(out_dir) / "metadata.csv"
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert set(rows[0].keys()) == {"file_name", "text"}


def test_jsonl_no_metadata(pdf, out_dir):
    """JSONL without metadata should not have the 'metadata' key."""
    export_training_data(pdf, out_dir, output_format="jsonl", include_metadata=False)

    jsonl_path = Path(out_dir) / "metadata.jsonl"
    with open(jsonl_path, "r", encoding="utf-8") as f:
        rec = json.loads(f.readline())

    assert "metadata" not in rec
    # Core fields still present
    assert "file_name" in rec
    assert "text" in rec
    assert "conversations" in rec


# ── train/val split ────────────────────────────────────────────────────


def test_train_val_split(pdf, out_dir):
    """With split=0.9, produces train/ and validation/ subdirectories."""
    result = export_training_data(pdf, out_dir, split=0.9)

    train_dir = Path(out_dir) / "train"
    val_dir = Path(out_dir) / "validation"

    assert train_dir.is_dir()
    assert val_dir.is_dir()

    train_images = list((train_dir / "images").glob("*.png"))
    val_images = list((val_dir / "images").glob("*.png"))

    total = len(train_images) + len(val_images)
    assert total == result["images"]
    assert len(train_images) > len(val_images)  # 90/10 split

    # Both splits have metadata files
    assert (train_dir / "metadata.jsonl").exists()
    assert (val_dir / "metadata.jsonl").exists()


# ── custom options ──────────────────────────────────────────────────────


def test_custom_prompt(pdf, out_dir):
    """Custom prompt string appears in conversation."""
    custom = "Read the text in this crop."
    export_training_data(pdf, out_dir, prompt=custom)

    jsonl_path = Path(out_dir) / "metadata.jsonl"
    with open(jsonl_path, "r", encoding="utf-8") as f:
        rec = json.loads(f.readline())

    assert custom in rec["conversations"][0]["content"]


def test_custom_selector(pdf, out_dir):
    """Selector filters which elements are exported."""
    # Export only bold text — may be fewer elements
    result_all = export_training_data(pdf, out_dir, selector="text")
    out_dir_bold = out_dir + "_bold"
    result_bold = export_training_data(pdf, out_dir_bold, selector="text:bold")

    # Bold subset should be smaller or equal
    assert result_bold["images"] <= result_all["images"]


# ── error handling ──────────────────────────────────────────────────────


def test_overwrite_false_raises(pdf, out_dir):
    """Raises FileExistsError if output dir exists and overwrite=False."""
    os.makedirs(out_dir)
    with pytest.raises(FileExistsError):
        export_training_data(pdf, out_dir, overwrite=False)


def test_overwrite_true_succeeds(pdf, out_dir):
    """With overwrite=True, existing directory is reused."""
    os.makedirs(out_dir)
    result = export_training_data(pdf, out_dir, overwrite=True)
    assert result["images"] > 0


def test_invalid_split_raises(pdf, out_dir):
    """Split outside (0, 1) raises ValueError."""
    with pytest.raises(ValueError):
        export_training_data(pdf, out_dir, split=1.5)


# ── overwrite must not destroy the previous export on failure ──────────


def _staging_dirs(out_dir):
    parent = Path(out_dir).parent
    return [p for p in parent.iterdir() if p.name.startswith(Path(out_dir).name + ".staging-")]


def test_overwrite_with_empty_source_preserves_old_export(pdf, out_dir):
    """overwrite=True with an empty source must leave the old export intact."""
    from natural_pdf.core.pdf_collection import PDFCollection

    old = export_training_data(pdf, out_dir)
    assert old["images"] > 0

    result = export_training_data(PDFCollection([]), out_dir, overwrite=True)
    assert result["images"] == 0

    # Previous export untouched
    assert (Path(out_dir) / "metadata.jsonl").exists()
    png_files = list((Path(out_dir) / "images").glob("*.png"))
    assert len(png_files) == old["images"]
    assert _staging_dirs(out_dir) == []


def test_overwrite_preserves_old_export_when_build_fails(pdf, out_dir, monkeypatch):
    """A failure while building the new export must leave the old one intact
    and clean up the staging directory."""
    import natural_pdf.exporters.training_data as td

    old = export_training_data(pdf, out_dir)
    assert old["images"] > 0

    def boom(*args, **kwargs):
        raise RuntimeError("disk exploded")

    monkeypatch.setattr(td, "_write_jsonl", boom)

    with pytest.raises(RuntimeError, match="disk exploded"):
        export_training_data(pdf, out_dir, overwrite=True)

    # Previous export untouched, no staging leftovers
    assert (Path(out_dir) / "metadata.jsonl").exists()
    png_files = list((Path(out_dir) / "images").glob("*.png"))
    assert len(png_files) == old["images"]
    assert _staging_dirs(out_dir) == []


def test_overwrite_with_no_matching_elements_preserves_old_export(pdf, out_dir):
    """A selector that matches nothing must not replace a previous export."""
    old = export_training_data(pdf, out_dir)
    assert old["images"] > 0

    result = export_training_data(
        pdf, out_dir, overwrite=True, selector='text:contains("ZZZNONEXISTENT")'
    )
    assert result["images"] == 0

    assert (Path(out_dir) / "metadata.jsonl").exists()
    assert _staging_dirs(out_dir) == []


def test_validation_errors_do_not_delete_old_export(pdf, out_dir):
    """Bad arguments with overwrite=True must fail before touching the old export."""
    export_training_data(pdf, out_dir)

    with pytest.raises(ValueError):
        export_training_data(pdf, out_dir, overwrite=True, split=1.5)
    with pytest.raises(ValueError):
        export_training_data(pdf, out_dir, overwrite=True, output_format="parquet")
    with pytest.raises(TypeError):
        export_training_data(object(), out_dir, overwrite=True)

    assert (Path(out_dir) / "metadata.jsonl").exists()
    assert _staging_dirs(out_dir) == []


def test_pathlib_path_output_dir(pdf, out_dir):
    """A pathlib.Path destination must work exactly like a str destination."""
    result = export_training_data(pdf, Path(out_dir))
    assert result["images"] > 0
    assert (Path(out_dir) / "metadata.jsonl").exists()

    # overwrite with a Path also exercises the staging/promote path
    result = export_training_data(pdf, Path(out_dir), overwrite=True)
    assert result["images"] > 0
    assert _staging_dirs(out_dir) == []
    assert _old_aside_dirs(out_dir) == []


def test_promote_failure_preserves_old_export(pdf, out_dir, monkeypatch):
    """If renaming staging into place fails, the ORIGINAL export must be
    restored intact (not deleted before the promote)."""
    old = export_training_data(pdf, out_dir)
    assert old["images"] > 0
    old_files = sorted(p.name for p in (Path(out_dir) / "images").glob("*.png"))

    real_rename = os.rename

    def failing_rename(src, dst):
        if ".staging-" in os.fspath(src):
            raise OSError("simulated promote failure")
        return real_rename(src, dst)

    monkeypatch.setattr(os, "rename", failing_rename)

    with pytest.raises(OSError, match="simulated promote failure"):
        export_training_data(pdf, out_dir, overwrite=True)

    monkeypatch.undo()

    # Original export restored at its original location, byte-for-byte set
    assert (Path(out_dir) / "metadata.jsonl").exists()
    assert sorted(p.name for p in (Path(out_dir) / "images").glob("*.png")) == old_files
    # No staging or aside leftovers
    assert _staging_dirs(out_dir) == []
    assert _old_aside_dirs(out_dir) == []


def _old_aside_dirs(out_dir):
    parent = Path(out_dir).parent
    return [p for p in parent.iterdir() if p.name.startswith(Path(out_dir).name + ".old-")]


def test_overwrite_refuses_unmarked_directory(pdf, out_dir):
    """overwrite=True still refuses a non-empty directory without the marker."""
    os.makedirs(out_dir)
    (Path(out_dir) / "precious.txt").write_text("user data")

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        export_training_data(pdf, out_dir, overwrite=True)

    assert (Path(out_dir) / "precious.txt").read_text() == "user data"
    assert _staging_dirs(out_dir) == []


# ── destination appearing between build and promote (TOCTOU) ───────────


def _intrude_after_build(monkeypatch, intrude):
    """Monkeypatch _build_export to run *intrude* between build and promote."""
    import natural_pdf.exporters.training_data as td

    real_build = td._build_export

    def build_then_intrude(*args, **kwargs):
        result = real_build(*args, **kwargs)
        intrude()
        return result

    monkeypatch.setattr(td, "_build_export", build_then_intrude)


def test_destination_dir_appearing_during_build_raises_and_is_untouched(pdf, out_dir, monkeypatch):
    """overwrite=False, destination absent at start: a directory created by
    another process during the build must survive, the export must raise, and
    the completed staging directory must be preserved (path in the error)."""

    def intrude():
        os.makedirs(out_dir)
        (Path(out_dir) / "valuable.txt").write_text("irreplaceable")

    _intrude_after_build(monkeypatch, intrude)

    with pytest.raises(FileExistsError, match="appeared") as excinfo:
        export_training_data(pdf, out_dir, overwrite=False)

    # The interloper is intact.
    assert (Path(out_dir) / "valuable.txt").read_text() == "irreplaceable"
    # The finished export is preserved in staging and named in the error.
    staging = _staging_dirs(out_dir)
    assert len(staging) == 1
    assert str(staging[0]) in str(excinfo.value)
    assert (staging[0] / "metadata.jsonl").exists()


def test_destination_file_appearing_during_build_raises_with_overwrite(pdf, out_dir, monkeypatch):
    """Even with overwrite=True, a plain file that appeared at the destination
    is not an export and must not be deleted."""

    def intrude():
        Path(out_dir).write_text("irreplaceable")

    _intrude_after_build(monkeypatch, intrude)

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        export_training_data(pdf, out_dir, overwrite=True)

    assert Path(out_dir).read_text() == "irreplaceable"
    assert len(_staging_dirs(out_dir)) == 1


def test_destination_replaced_with_foreign_dir_during_build_raises(pdf, out_dir, monkeypatch):
    """overwrite=True on a valid old export: if the destination is swapped for
    a marker-less non-empty directory during the build, promotion must refuse
    to touch it and keep the staging directory."""
    import shutil

    old = export_training_data(pdf, out_dir)
    assert old["images"] > 0

    def intrude():
        shutil.rmtree(out_dir)
        os.makedirs(out_dir)
        (Path(out_dir) / "valuable.txt").write_text("irreplaceable")

    _intrude_after_build(monkeypatch, intrude)

    with pytest.raises(FileExistsError, match="Refusing to overwrite") as excinfo:
        export_training_data(pdf, out_dir, overwrite=True)

    assert (Path(out_dir) / "valuable.txt").read_text() == "irreplaceable"
    staging = _staging_dirs(out_dir)
    assert len(staging) == 1
    assert str(staging[0]) in str(excinfo.value)


# ── empty elements are skipped ──────────────────────────────────────────


def test_skipped_count(pdf, out_dir):
    """Elements with empty text are counted as skipped."""
    result = export_training_data(pdf, out_dir)
    # skipped should be a non-negative int
    assert isinstance(result["skipped"], int)
    assert result["skipped"] >= 0


# ── convenience methods ─────────────────────────────────────────────────


def test_pdf_convenience_method(pdf, out_dir):
    """PDF.export_training_data() delegates correctly."""
    result = pdf.export_training_data(out_dir)
    assert result["images"] > 0
    assert (Path(out_dir) / "metadata.jsonl").exists()


def test_pdf_collection_convenience_method(out_dir):
    """PDFCollection.export_training_data() delegates correctly."""
    from natural_pdf.core.pdf_collection import PDFCollection

    collection = PDFCollection([TEST_PDF])
    result = collection.export_training_data(out_dir)
    assert result["images"] > 0
