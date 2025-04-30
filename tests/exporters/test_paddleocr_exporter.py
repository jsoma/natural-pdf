import os
import shutil
import tempfile
from pathlib import Path

import pytest

from natural_pdf.core.pdf import PDF
from natural_pdf.exporters import PaddleOCRRecognitionExporter

# Define the path to the test PDF relative to the project root
# Adjust if your tests run from a different context
TEST_PDF_PATH = Path("pdfs/01-practice.pdf")


@pytest.fixture
def temp_output_dir():
    """Creates a temporary directory for test output."""
    temp_dir = tempfile.mkdtemp()
    print(f"Created temp dir: {temp_dir}")  # For debugging
    yield temp_dir
    print(f"Removing temp dir: {temp_dir}")  # For debugging
    shutil.rmtree(temp_dir)


def test_paddleocr_export_basic(temp_output_dir):
    """Test basic export functionality with default settings."""
    # Ensure the test PDF exists
    if not TEST_PDF_PATH.exists():
        pytest.fail(f"Test PDF not found at: {TEST_PDF_PATH.resolve()}")

    try:
        pdf = PDF(str(TEST_PDF_PATH))
    except Exception as e:
        pytest.fail(f"Failed to load test PDF: {e}")

    exporter = PaddleOCRRecognitionExporter(
        split_ratio=0.8,  # Use a specific split for predictability
        include_guide=False,  # Don't need the guide for this test
        random_seed=42,  # Ensure reproducible split
    )

    try:
        exporter.export(pdf, temp_output_dir)
    except Exception as e:
        pytest.fail(f"Exporter failed during export: {e}")

    # --- Assertions ---
    output_dir = Path(temp_output_dir)
    images_dir = output_dir / "images"
    dict_file = output_dir / "dict.txt"
    train_file = output_dir / "train.txt"
    val_file = output_dir / "val.txt"

    # 1. Check if files and directories exist
    assert output_dir.exists()
    assert images_dir.exists()
    assert images_dir.is_dir()
    assert dict_file.exists()
    assert dict_file.is_file()
    assert train_file.exists()
    assert train_file.is_file()
    assert val_file.exists()
    assert val_file.is_file()

    # 2. Check number of images
    # Get expected number of text elements from the PDF that the exporter *would* process
    all_elements = pdf.find_all("text")
    expected_elements = []
    for el in all_elements:
        # Apply the same checks as the exporter
        if not (hasattr(el, "page") and hasattr(el, "text") and hasattr(el, "expand")):
            continue
        if not el.text or not isinstance(el.text, str):
            continue
        expected_elements.append(el)

    num_expected_elements = len(expected_elements)
    assert num_expected_elements > 0, "Test PDF should contain processable text elements"

    exported_images = list(images_dir.glob("*.png"))
    num_exported_images = len(exported_images)
    assert (
        num_exported_images == num_expected_elements
    ), f"Number of exported images ({num_exported_images}) should match processable text elements ({num_expected_elements})"

    # 3. Check label file entries count
    with open(train_file, "r", encoding="utf-8") as f:
        train_lines = f.readlines()
    with open(val_file, "r", encoding="utf-8") as f:
        val_lines = f.readlines()

    total_label_lines = len(train_lines) + len(val_lines)
    assert (
        total_label_lines == num_exported_images
    ), f"Total lines in label files ({total_label_lines}) should match exported images ({num_exported_images})"

    # Check split ratio (approximate due to integer split)
    expected_train_count = int(num_exported_images * 0.8)
    expected_val_count = num_exported_images - expected_train_count
    # Allow for slight variation due to rounding
    assert abs(len(train_lines) - expected_train_count) <= 1
    assert abs(len(val_lines) - expected_val_count) <= 1

    # 4. Check label file format (first line of train)
    if train_lines:
        first_line = train_lines[0].strip()
        assert "\t" in first_line, "Label line should be tab-separated"
        img_part, text_part = first_line.split("\t", 1)
        assert img_part.startswith("images/") and img_part.endswith(
            ".png"
        ), "Image path format incorrect"
        assert len(text_part) > 0, "Text part should not be empty"

    # 5. Check dictionary content
    # Build expected chars the same way the exporter does
    expected_char_set = set()
    for el in expected_elements:
        expected_char_set.update(el.text)
    expected_chars = sorted(list(expected_char_set), reverse=True)

    # Read dict.txt more carefully
    dict_chars_raw = []
    dict_chars_processed = []
    try:
        with open(dict_file, "r", encoding="utf-8") as f:
            dict_chars_raw = f.readlines()  # Read raw lines including \n
            # Use rstrip to remove only the trailing newline, preserving the character itself
            dict_chars_processed = [
                line.rstrip("\n") for line in dict_chars_raw if line.rstrip("\n")
            ]

    except FileNotFoundError:
        # Let the assertion below fail clearly
        pass

    assert dict_chars_processed == expected_chars, "Dictionary content mismatch"

    # Cleanup handled by the fixture
    pdf.close()
