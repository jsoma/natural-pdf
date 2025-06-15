import os
import shutil
import tempfile
from pathlib import Path

import pytest

from natural_pdf.core.pdf import PDF
from natural_pdf.exporters import PaddleOCRRecognitionExporter

# Use the new test file with known content
TEST_PDF_PATH = Path("pdfs/word-counter.pdf")


@pytest.fixture
def temp_output_dir():
    """Creates a temporary directory for test output."""
    temp_dir = tempfile.mkdtemp()
    print(f"Created temp dir: {temp_dir}")  # For debugging
    yield temp_dir
    print(f"Removing temp dir: {temp_dir}")  # For debugging
    shutil.rmtree(temp_dir)


def test_paddleocr_export_word_counter_min_char_freq(temp_output_dir):
    """Test export with word-counter.pdf and min_char_freq=2 (should export only 'hello' and 'world')."""
    if not TEST_PDF_PATH.exists():
        pytest.fail(f"Test PDF not found at: {TEST_PDF_PATH.resolve()}")

    pdf = PDF(str(TEST_PDF_PATH))
    exporter = PaddleOCRRecognitionExporter(
        split_ratio=0.8,
        include_guide=False,
        random_seed=42,
        min_char_freq=2,
    )
    exporter.export(pdf, temp_output_dir)

    output_dir = Path(temp_output_dir)
    images_dir = output_dir / "images"
    dict_file = output_dir / "dict.txt"
    train_file = output_dir / "train.txt"
    val_file = output_dir / "val.txt"

    # Files/directories should exist
    assert output_dir.exists()
    assert images_dir.exists()
    assert images_dir.is_dir()
    assert dict_file.exists()
    assert dict_file.is_file()
    assert train_file.exists()
    assert train_file.is_file()
    assert val_file.exists()
    assert val_file.is_file()

    # Should be 4 exported images (2 'hello', 2 'world')
    exported_images = list(images_dir.glob("*.png"))
    assert len(exported_images) == 4, f"Expected 4 exported images, got {len(exported_images)}"

    with open(train_file, "r", encoding="utf-8") as f:
        train_lines = f.readlines()
    with open(val_file, "r", encoding="utf-8") as f:
        val_lines = f.readlines()
    total_label_lines = len(train_lines) + len(val_lines)
    assert total_label_lines == 4, f"Expected 4 label lines, got {total_label_lines}"

    # Check that all label lines are either 'hello' or 'world'
    all_labels = [line.strip().split("\t", 1)[1] for line in train_lines + val_lines]
    for label in all_labels:
        assert label in ("hello", "world"), f"Unexpected label: {label}"

    # Dictionary should contain only the characters from 'hello' and 'world'
    expected_chars = set("helloworld")
    with open(dict_file, "r", encoding="utf-8") as f:
        dict_lines = {line.strip() for line in f.readlines() if line.strip()}
    assert dict_lines == expected_chars, f"Expected dictionary {expected_chars}, got {dict_lines}"

    pdf.close()
