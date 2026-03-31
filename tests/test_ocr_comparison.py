"""Tests for OCR comparison tool."""

from unittest.mock import MagicMock

import pytest

from natural_pdf.ocr.alignment import (
    _split_into_columns,
    _TaggedBox,
    _vertical_overlap_ratio,
    align_by_rows,
    align_by_tiles,
    align_ocr_outputs,
    check_alignment_health,
)
from natural_pdf.ocr.comparison import (
    ComparisonRegion,
    OcrComparison,
    _edit_distance_ratio,
    classify_region,
    compute_consensus,
    find_outlier,
    normalize_text,
    render_char_diff_html,
)

# ---------------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------------


class TestNormalizeText:
    def test_collapse_whitespace(self):
        assert normalize_text("  hello   world  ") == "hello world"

    def test_collapse_tabs_newlines(self):
        assert normalize_text("hello\t\n  world") == "hello world"

    def test_strict_preserves_all(self):
        assert normalize_text("  hello   world  ", mode="strict") == "  hello   world  "

    def test_ignore_strips_all(self):
        assert normalize_text("THIS TEXT", mode="ignore") == "THISTEXT"

    def test_nfkc_normalization(self):
        # fi ligature → fi
        assert normalize_text("\ufb01le") == "file"

    def test_empty_string(self):
        assert normalize_text("") == ""


# ---------------------------------------------------------------------------
# Edit distance
# ---------------------------------------------------------------------------


class TestEditDistance:
    def test_identical(self):
        assert _edit_distance_ratio("hello", "hello") == 0.0

    def test_completely_different(self):
        ratio = _edit_distance_ratio("abc", "xyz")
        assert ratio > 0.9

    def test_near_miss(self):
        ratio = _edit_distance_ratio("Hello", "He11o")
        assert 0.1 < ratio < 0.5

    def test_empty_both(self):
        assert _edit_distance_ratio("", "") == 0.0

    def test_one_empty(self):
        assert _edit_distance_ratio("hello", "") == 1.0


# ---------------------------------------------------------------------------
# Consensus
# ---------------------------------------------------------------------------


class TestConsensus:
    def test_unanimous(self):
        texts = {"a": "hello", "b": "hello", "c": "hello"}
        assert compute_consensus(texts) == "hello"

    def test_majority(self):
        texts = {"a": "hello", "b": "hello", "c": "hallo"}
        # Medoid should be "hello" (closer to both others)
        assert compute_consensus(texts) == "hello"

    def test_single_engine(self):
        assert compute_consensus({"a": "test"}) == "test"

    def test_empty(self):
        assert compute_consensus({}) is None


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


class TestClassification:
    def test_agreement(self):
        dists = {"a": 0.0, "b": 0.02}
        assert classify_region(dists, 2, 2) == "agreement"

    def test_near_miss(self):
        dists = {"a": 0.0, "b": 0.15}
        assert classify_region(dists, 2, 2) == "near_miss"

    def test_catastrophic(self):
        dists = {"a": 0.0, "b": 0.6}
        assert classify_region(dists, 2, 2) == "catastrophic"

    def test_missing_engine(self):
        dists = {"a": 0.0}
        assert classify_region(dists, 1, 2) == "catastrophic"

    def test_short_string_lower_threshold(self):
        # For short strings, 0.22 should be catastrophic (threshold 0.20)
        dists = {"a": 0.0, "b": 0.22}
        assert classify_region(dists, 2, 2, text_length=5) == "catastrophic"

    def test_long_string_same_value_near_miss(self):
        # For longer strings, 0.22 is near-miss (threshold 0.25)
        dists = {"a": 0.0, "b": 0.22}
        assert classify_region(dists, 2, 2, text_length=50) == "near_miss"


# ---------------------------------------------------------------------------
# Outlier detection
# ---------------------------------------------------------------------------


class TestOutlier:
    def test_clear_outlier(self):
        dists = {"a": 0.01, "b": 0.02, "c": 0.50}
        assert find_outlier(dists) == "c"

    def test_no_outlier(self):
        dists = {"a": 0.05, "b": 0.06, "c": 0.07}
        assert find_outlier(dists) is None

    def test_two_engines(self):
        # Not meaningful with 2 engines
        dists = {"a": 0.0, "b": 0.5}
        assert find_outlier(dists) is None

    def test_all_agree(self):
        dists = {"a": 0.0, "b": 0.0, "c": 0.0}
        assert find_outlier(dists) is None

    def test_one_disagrees_from_zero(self):
        dists = {"a": 0.0, "b": 0.0, "c": 0.10}
        assert find_outlier(dists) == "c"


# ---------------------------------------------------------------------------
# HTML diff rendering
# ---------------------------------------------------------------------------


class TestHtmlDiff:
    def test_identical(self):
        html = render_char_diff_html("hello", "hello")
        assert "hello" in html

    def test_substitution(self):
        html = render_char_diff_html("Hello", "He11o")
        assert "fff3cd" in html  # yellow highlight for disagreement
        assert "11" in html

    def test_missing(self):
        html = render_char_diff_html("hello", "")
        assert "[missing]" in html

    def test_disagreement_highlighted(self):
        html = render_char_diff_html("helo", "hello")
        assert "fff3cd" in html  # yellow highlight


# ---------------------------------------------------------------------------
# Vertical overlap
# ---------------------------------------------------------------------------


class TestVerticalOverlap:
    def test_full_overlap(self):
        assert _vertical_overlap_ratio(0, 10, 0, 10) == 1.0

    def test_no_overlap(self):
        assert _vertical_overlap_ratio(0, 10, 20, 30) == 0.0

    def test_partial_overlap(self):
        ratio = _vertical_overlap_ratio(0, 10, 5, 15)
        assert 0.4 < ratio < 0.6

    def test_contained(self):
        # Small box inside big box
        ratio = _vertical_overlap_ratio(0, 100, 40, 60)
        assert ratio == 1.0


# ---------------------------------------------------------------------------
# Column splitting
# ---------------------------------------------------------------------------


class TestColumnSplitting:
    def _make_box(self, x0, top, x1, bottom, text="word"):
        elem = MagicMock()
        elem.x0 = x0
        elem.top = top
        elem.x1 = x1
        elem.bottom = bottom
        elem.text = text
        elem.confidence = 0.9
        return _TaggedBox("engine_a", elem)

    def test_single_column(self):
        boxes = [
            self._make_box(10, 0, 50, 10),
            self._make_box(55, 0, 100, 10),
        ]
        segments = _split_into_columns(boxes, median_width=40)
        assert len(segments) == 1  # no big gap

    def test_two_columns(self):
        boxes = [
            self._make_box(10, 0, 50, 10),
            self._make_box(300, 0, 350, 10),  # large gap
        ]
        segments = _split_into_columns(boxes, median_width=40)
        assert len(segments) == 2


# ---------------------------------------------------------------------------
# Row-based alignment with mock elements
# ---------------------------------------------------------------------------


class TestAlignByRows:
    def _make_elem(self, x0, top, x1, bottom, text="word"):
        elem = MagicMock()
        elem.x0 = x0
        elem.top = top
        elem.x1 = x1
        elem.bottom = bottom
        elem.text = text
        elem.confidence = 0.9
        return elem

    def test_two_engines_same_text(self):
        engine_a = [self._make_elem(10, 10, 100, 25, "Hello World")]
        engine_b = [self._make_elem(10, 10, 100, 25, "Hello World")]
        regions = align_by_rows({"a": engine_a, "b": engine_b})
        assert len(regions) >= 1
        assert regions[0].classification == "agreement"

    def test_two_engines_different_text(self):
        engine_a = [self._make_elem(10, 10, 100, 25, "Hello")]
        engine_b = [self._make_elem(10, 10, 100, 25, "Goodbye")]
        regions = align_by_rows({"a": engine_a, "b": engine_b})
        assert len(regions) >= 1
        assert regions[0].classification == "catastrophic"

    def test_near_miss(self):
        engine_a = [self._make_elem(10, 10, 100, 25, "Hello")]
        engine_b = [self._make_elem(10, 10, 100, 25, "He11o")]
        regions = align_by_rows({"a": engine_a, "b": engine_b})
        assert len(regions) >= 1
        assert regions[0].classification in ("near_miss", "catastrophic")

    def test_multiple_lines(self):
        engine_a = [
            self._make_elem(10, 10, 100, 25, "Line one"),
            self._make_elem(10, 30, 100, 45, "Line two"),
        ]
        engine_b = [
            self._make_elem(10, 10, 100, 25, "Line one"),
            self._make_elem(10, 30, 100, 45, "Line two"),
        ]
        regions = align_by_rows({"a": engine_a, "b": engine_b})
        assert len(regions) == 2
        assert all(r.classification == "agreement" for r in regions)

    def test_missing_engine(self):
        engine_a = [self._make_elem(10, 10, 100, 25, "Hello")]
        engine_b = []  # no results
        regions = align_by_rows({"a": engine_a, "b": engine_b})
        assert len(regions) >= 1
        assert regions[0].classification == "catastrophic"

    def test_tall_box_handling(self):
        """Block-level element (tall) should be set aside and mapped back."""
        # Engine A has word-level
        engine_a = [
            self._make_elem(10, 10, 100, 25, "Line one"),
            self._make_elem(10, 30, 100, 45, "Line two"),
        ]
        # Engine B has a single tall block
        engine_b = [self._make_elem(10, 10, 100, 45, "Line one Line two")]
        regions = align_by_rows({"a": engine_a, "b": engine_b})
        # Engine B's tall box should be mapped to at least one region
        has_b = any("b" in r.texts for r in regions)
        assert has_b


# ---------------------------------------------------------------------------
# Tile-based alignment
# ---------------------------------------------------------------------------


class TestAlignByTiles:
    def _make_elem(self, x0, top, x1, bottom, text="word"):
        elem = MagicMock()
        elem.x0 = x0
        elem.top = top
        elem.x1 = x1
        elem.bottom = bottom
        elem.text = text
        elem.confidence = 0.9
        return elem

    def test_basic(self):
        engine_a = [self._make_elem(10, 10, 50, 25, "Hello")]
        engine_b = [self._make_elem(10, 10, 50, 25, "Hello")]
        regions = align_by_tiles(
            {"a": engine_a, "b": engine_b},
            page_bbox=(0, 0, 612, 792),
        )
        assert len(regions) >= 1


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


class TestHealthCheck:
    def test_healthy(self):
        regions = [
            ComparisonRegion(
                bbox=(0, 0, 100, 20),
                texts={"a": "hi", "b": "hi"},
                normalized_texts={"a": "hi", "b": "hi"},
                confidences={"a": 0.9, "b": 0.9},
                consensus="hi",
                classification="agreement",
                edit_distances={"a": 0.0, "b": 0.0},
                outlier_engine=None,
                elements={
                    "a": [MagicMock(x0=0, top=0, x1=100, bottom=20)],
                    "b": [MagicMock(x0=0, top=0, x1=100, bottom=20)],
                },
            )
        ]
        healthy, diag = check_alignment_health(regions, 2)
        assert healthy

    def test_unhealthy_orphans(self):
        regions = [
            ComparisonRegion(
                bbox=(0, 0, 100, 20),
                texts={"a": "hi"},
                normalized_texts={"a": "hi"},
                confidences={"a": 0.9},
                consensus="hi",
                classification="catastrophic",
                edit_distances={"a": 0.0},
                outlier_engine=None,
                elements={"a": [MagicMock(x0=0, top=0, x1=100, bottom=20)]},
            )
        ] * 5  # all orphans
        healthy, diag = check_alignment_health(regions, 2)
        assert not healthy


# ---------------------------------------------------------------------------
# Auto alignment
# ---------------------------------------------------------------------------


class TestAutoAlignment:
    def _make_elem(self, x0, top, x1, bottom, text="word"):
        elem = MagicMock()
        elem.x0 = x0
        elem.top = top
        elem.x1 = x1
        elem.bottom = bottom
        elem.text = text
        elem.confidence = 0.9
        return elem

    def test_auto_uses_rows(self):
        engine_a = [self._make_elem(10, 10, 100, 25, "Hello")]
        engine_b = [self._make_elem(10, 10, 100, 25, "Hello")]
        regions, strategy, diag = align_ocr_outputs(
            {"a": engine_a, "b": engine_b},
            page_bbox=(0, 0, 612, 792),
        )
        assert strategy == "rows"

    def test_forced_tiles(self):
        engine_a = [self._make_elem(10, 10, 100, 25, "Hello")]
        engine_b = [self._make_elem(10, 10, 100, 25, "Hello")]
        regions, strategy, diag = align_ocr_outputs(
            {"a": engine_a, "b": engine_b},
            page_bbox=(0, 0, 612, 792),
            strategy="tiles",
        )
        assert strategy == "tiles"


# ---------------------------------------------------------------------------
# OcrComparison result object
# ---------------------------------------------------------------------------


class TestOcrComparison:
    def _make_comparison(self):
        page = MagicMock()
        page.width = 612
        page.height = 792

        regions = [
            ComparisonRegion(
                bbox=(10, 10, 100, 25),
                texts={"easyocr": "Hello World", "surya": "Hello World"},
                normalized_texts={"easyocr": "Hello World", "surya": "Hello World"},
                confidences={"easyocr": 0.9, "surya": 0.95},
                consensus="Hello World",
                classification="agreement",
                edit_distances={"easyocr": 0.0, "surya": 0.0},
                outlier_engine=None,
            ),
            ComparisonRegion(
                bbox=(10, 30, 100, 45),
                texts={"easyocr": "He11o", "surya": "Hello"},
                normalized_texts={"easyocr": "He11o", "surya": "Hello"},
                confidences={"easyocr": 0.7, "surya": 0.9},
                consensus="Hello",
                classification="near_miss",
                edit_distances={"easyocr": 0.2, "surya": 0.0},
                outlier_engine="easyocr",
            ),
        ]

        return OcrComparison(
            page=page,
            engines=["easyocr", "surya"],
            failed_engines={},
            regions=regions,
            engine_elements={"easyocr": [], "surya": []},
            strategy_used="rows",
            diagnostics={},
            runtimes={"easyocr": 1.5, "surya": 2.3},
            resolution=150,
            normalize_mode="collapse",
        )

    def test_summary(self):
        comp = self._make_comparison()
        df = comp.summary()
        assert len(df) == 2
        assert "engine" in df.columns
        assert "agreement" in df.columns

    def test_diff_default_hides_agreement(self):
        comp = self._make_comparison()
        result = comp.diff()
        html = result._repr_html_()
        # Should contain near_miss row but not agreement row
        assert "near_miss" in html
        # For 2 engines, should show engine names, not consensus
        assert "easyocr:" in html
        assert "surya:" in html

    def test_diff_show_all(self):
        comp = self._make_comparison()
        result = comp.diff(only="all")
        html = result._repr_html_()
        assert "agreement" in html
        assert "near_miss" in html

    def test_diff_pairwise_no_consensus(self):
        """With 2 engines, consensus column should not appear."""
        comp = self._make_comparison()
        result = comp.diff(only="all")
        html = result._repr_html_()
        assert "consensus:" not in html

    def test_repr_html(self):
        comp = self._make_comparison()
        html = comp._repr_html_()
        assert "OCR Comparison" in html
        assert "easyocr" in html
        assert "surya" in html

    def test_engines_property(self):
        comp = self._make_comparison()
        assert comp.engines == ["easyocr", "surya"]

    def test_strategy_used(self):
        comp = self._make_comparison()
        assert comp.strategy_used == "rows"
