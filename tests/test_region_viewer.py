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

    # The library expects *ipywidgets* to be available in our test environment.
    # Attempt to import it directly; if this fails we skip the test entirely.
    try:
        import ipywidgets as _  # noqa: F401
    except ModuleNotFoundError:
        pytest.skip("ipywidgets not installed – skipping interactive viewer test")

    # Validate that the viewer returns the expected widget type.
    from natural_pdf.widgets.viewer import InteractiveViewerWidget

    assert (
        viewer_widget is not None
    ), "Region.viewer() should return a widget when ipywidgets is installed"
    assert isinstance(viewer_widget, InteractiveViewerWidget)
