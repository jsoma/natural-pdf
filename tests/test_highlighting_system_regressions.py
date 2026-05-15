from pathlib import Path

import pytest
from PIL import ImageChops

from natural_pdf import PDF
from natural_pdf.core.render_spec import RenderSpec, add_explicit_highlights_to_spec
from natural_pdf.utils.visualization import ColorManager

PDF_PATH = Path("pdfs/multicolumn.pdf")


@pytest.fixture
def sample_pdf():
    if not PDF_PATH.exists():
        pytest.skip("Test requires pdfs/multicolumn.pdf fixture")

    pdf = PDF(str(PDF_PATH))
    try:
        yield pdf
    finally:
        pdf.close()


def _images_differ(left, right) -> bool:
    if left.mode != right.mode:
        right = right.convert(left.mode)
    return ImageChops.difference(left, right).getbbox() is not None


def test_page_show_highlights_false_suppresses_persistent_highlights(sample_pdf):
    page = sample_pdf.pages[0]
    page.add_highlight(bbox=(50, 50, 150, 100), label="persistent")

    specs = page._get_render_specs(mode="show", highlights=False)

    assert len(specs) == 1
    assert specs[0].highlights == []


def test_collection_show_highlights_false_suppresses_collection_highlights(sample_pdf):
    page = sample_pdf.pages[0]
    elements = page.find_all("text")[:3]

    specs = elements._get_render_specs(mode="show", highlights=False)

    assert specs
    assert all(spec.highlights == [] for spec in specs)


def test_render_with_direct_highlight_changes_pixels(sample_pdf):
    page = sample_pdf.pages[0]
    clean = page.render(width=300)
    highlighted = page.render(
        width=300,
        highlights=[{"bbox": (50, 50, 150, 100), "color": "red"}],
    )

    assert clean is not None
    assert highlighted is not None
    assert _images_differ(clean, highlighted)


def test_explicit_highlight_style_options_are_preserved():
    spec = RenderSpec(page=object())

    add_explicit_highlights_to_spec(
        spec,
        [
            {
                "bbox": (1, 2, 3, 4),
                "color": "blue",
                "fill": False,
                "line_width": 3,
                "vertices": False,
            }
        ],
    )

    assert spec.highlights == [
        {
            "bbox": (1, 2, 3, 4),
            "color": "blue",
            "fill": False,
            "line_width": 3,
            "vertices": False,
        }
    ]


def test_highlight_context_render_uses_accumulated_highlights(sample_pdf):
    page = sample_pdf.pages[0]
    clean = page.render(width=300)
    region = page.region(left=50, right=150, top=50, bottom=100)

    with page.highlights() as highlights:
        highlights.add(region, label="region", color="red")
        highlighted = highlights.render(width=300)

    assert clean is not None
    assert highlighted is not None
    assert _images_differ(clean, highlighted)


def test_label_format_is_written_back_for_legend_generation(sample_pdf):
    page = sample_pdf.pages[0]
    spec = RenderSpec(page=page)
    spec.add_highlight(bbox=(50, 50, 150, 100), color="red")

    image = sample_pdf.highlighter.unified_render(
        specs=[spec],
        width=300,
        labels=True,
        label_format="Item {index}",
    )

    assert image is not None
    assert spec.highlights[0]["label"] == "Item 0"


def test_unit_rgb_color_tuples_are_normalized(sample_pdf):
    highlighter = sample_pdf.highlighter

    assert highlighter._process_color_input((1, 0, 0)) == (255, 0, 0, 100)
    assert highlighter._process_color_input((1.0, 0, 0)) == (255, 0, 0, 100)
    assert highlighter._process_color_input((1, 0, 0, 0.5)) == (255, 0, 0, 127)
    assert highlighter._process_color_input((255, 0, 0)) == (255, 0, 0, 100)


def test_color_manager_uses_deterministic_order():
    first = ColorManager()
    second = ColorManager()

    assert first.get_color(label="one") == (255, 0, 0, 100)
    assert second.get_color(label="one") == (255, 0, 0, 100)

    first.get_color(label="two")
    first.reset()

    assert first.get_color(label="one") == (255, 0, 0, 100)


def test_label_color_assignments_remain_persistent(sample_pdf):
    page = sample_pdf.pages[0]
    spec = RenderSpec(page=page)
    spec.add_highlight(bbox=(50, 50, 150, 100), label="persistent label")

    image = sample_pdf.highlighter.unified_render(specs=[spec], width=300, labels=False)

    assert image is not None
    assert sample_pdf.highlighter.get_labels_and_colors()["persistent label"] == (
        255,
        0,
        0,
        100,
    )
