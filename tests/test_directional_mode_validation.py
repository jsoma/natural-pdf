"""The cross-direction parameter is a mode string ('full'/'element').

A number there used to be silently treated like 'element' — e.g.
.below(width=200) produced a plausible-looking but wrong region. It must
raise instead.
"""

import pytest

from natural_pdf import PDF


@pytest.fixture(scope="module")
def page():
    pdf = PDF("pdfs/01-practice.pdf")
    yield pdf.pages[0]
    pdf.close()


@pytest.fixture(scope="module")
def anchor(page):
    el = page.find("text")
    assert el is not None
    return el


class TestNumericCrossSizeRaises:
    def test_below_numeric_width(self, anchor):
        with pytest.raises(TypeError, match="width.*'full' or 'element'"):
            anchor.below(width=200)

    def test_above_numeric_width(self, anchor):
        with pytest.raises(TypeError, match="width.*'full' or 'element'"):
            anchor.above(width=200)

    def test_left_numeric_height(self, anchor):
        with pytest.raises(TypeError, match="height.*'full' or 'element'"):
            anchor.left(height=50)

    def test_right_numeric_height(self, anchor):
        with pytest.raises(TypeError, match="height.*'full' or 'element'"):
            anchor.right(height=50)

    def test_region_path_raises_too(self, page):
        region = page.create_region(0, 0, page.width, page.height / 2)
        with pytest.raises(TypeError, match="width.*'full' or 'element'"):
            region.below(width=100)

    def test_message_points_to_extent_param(self, anchor):
        with pytest.raises(TypeError, match="height=<number>"):
            anchor.below(width=200)


class TestValidModesStillWork:
    def test_below_modes(self, anchor):
        assert anchor.below(width="full") is not None
        assert anchor.below(width="element") is not None

    def test_right_modes(self, anchor):
        assert anchor.right(height="element") is not None
        assert anchor.right(height="full") is not None

    def test_numeric_extent_params_fine(self, anchor):
        region = anchor.below(height=100)
        assert region is not None
        region = anchor.right(width=100)
        assert region is not None
