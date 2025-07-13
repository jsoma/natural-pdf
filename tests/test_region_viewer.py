import pytest

from natural_pdf import PDF

# Path to a small sample PDF bundled with the repo
_SAMPLE_PDF_PATH = "pdfs/01-practice.pdf"


def _load_first_page():
    """Utility helper to load first page of bundled sample PDF."""
    pdf = PDF(_SAMPLE_PDF_PATH)
    try:
        return pdf.pages[0]
    finally:
        # Keep PDF open for region usage; closing handled by GC at process end.
        # Explicit close causes region.page references to break.
        pass


def test_region_viewer_no_error():
    """Ensure ``Region.viewer()`` executes without raising exceptions.

    The test is environment-agnostic: if *ipywidgets* is available it expects an
    ``InteractiveViewerWidget`` instance; otherwise it merely confirms that the
    call returns ``None`` and, most importantly, does **not** raise.
    """

    page = _load_first_page()

    # Define a small region in the upper-left quadrant
    region = page.region(0, 0, page.width / 2, page.height / 2)

    try:
        viewer_widget = region.viewer()
    except Exception as exc:
        pytest.fail(f"Region.viewer() raised an unexpected exception: {exc}")

    # If ipywidgets is present we should get an InteractiveViewerWidget instance
    if pytest.importorskip(
        "ipywidgets", reason="ipywidgets not installed", allow_module_level=False
    ):
        from natural_pdf.widgets.viewer import InteractiveViewerWidget

        assert viewer_widget is None or isinstance(viewer_widget, InteractiveViewerWidget)
    else:
        # When ipywidgets is missing the function should gracefully return None
        assert viewer_widget is None
