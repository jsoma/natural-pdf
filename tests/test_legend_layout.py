import pytest
from PIL import Image

from natural_pdf.exporters.annotated_pdf import (
    _prepare_sidebar_columns,
    _wrap_pdf_lines,
    create_annotated_pdf,
)
from natural_pdf.utils.visualization import (
    create_legend,
    legend_width_for_image,
    merge_images_with_legend,
)


def test_create_legend_wraps_long_labels_with_fixed_width():
    short = create_legend({"short": (255, 0, 0, 100)}, width=220)
    long = create_legend(
        {
            "very long field name with a value that should wrap instead of overflowing": (
                255,
                0,
                0,
                100,
            )
        },
        width=220,
    )

    assert long.width == 220
    assert long.height > short.height


def test_create_legend_adds_columns_when_height_is_capped():
    labels = {f"field {idx}": (255, 0, 0, 100) for idx in range(30)}

    legend = create_legend(labels, width=240, max_height=90)

    assert legend.width > 240
    assert legend.height <= 90


def test_side_legend_width_is_capped():
    assert legend_width_for_image(300, "right") == 180
    assert legend_width_for_image(2000, "left") == 340
    assert legend_width_for_image(320, "bottom") == 320


def test_merge_images_with_legend_centers_short_side_legend():
    image = Image.new("RGBA", (300, 200), (255, 255, 255, 255))
    legend = Image.new("RGBA", (100, 50), (240, 240, 240, 255))

    merged = merge_images_with_legend(image, legend, position="right")

    assert merged.size == (400, 200)


def test_pdf_sidebar_wrapping_and_column_helpers():
    lines = _wrap_pdf_lines(
        "field:\n  a very long cited value that should wrap inside the sidebar",
        max_width=80,
    )
    assert len(lines) > 2

    items = [{"label": f"field {idx}: value", "rgba": (255, 0, 0, 255)} for idx in range(20)]
    columns = _prepare_sidebar_columns(items, text_width=90, available_height=65)

    assert len(columns) > 1
    assert all(not item["lines"][0].startswith("+ ") for column in columns for item in column)


def test_create_annotated_pdf_rejects_invalid_legend_scope(tmp_path):
    with pytest.raises(ValueError, match="legend_scope"):
        create_annotated_pdf({}, str(tmp_path / "out.pdf"), legend_scope="all")
