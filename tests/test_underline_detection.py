from natural_pdf import PDF


def test_underline_detection_types_of_type():
    pdf = PDF("pdfs/types-of-type.pdf")
    page = pdf.pages[0]

    underlined_words = [w for w in page.words if getattr(w, "underline", False)]
    assert underlined_words, "Expected at least one underlined word"

    texts = " ".join(w.text for w in underlined_words).lower()
    assert "underlined" in texts, "Word 'Underlined' should be flagged underline"
    # ensure 'but' not underlined
    assert not any(
        w.text.lower().startswith("but") and w.underline for w in page.words
    ), "Word 'but' should not be underlined"


# --- Regression: decoration drawn as multiple overlapping segments ---------
# A single underline/strike is often emitted as several abutting line segments.
# Per-segment coverage testing left glyphs straddling a segment join below the
# threshold, which fragmented words at those glyphs (e.g. "Transactions" ->
# "Tra", "n", "sactio", "n", "s").  _union_coverage merges the segments first.

from natural_pdf.core.decoration_detector import _union_coverage


def test_union_coverage_merges_overlapping_segments():
    # Char spans x[10,16]; underline drawn as two pieces that overlap mid-glyph.
    # Neither single piece covers the char, but together they cover it fully.
    segments = [(10, 13), (12, 16)]
    assert _union_coverage(10, 16, segments) == 1.0


def test_union_coverage_ignores_chars_with_no_segment():
    # "helloWORLD" with strike over WORLD only: hello chars have no segment
    # above them and must report zero coverage (so the word splits correctly).
    strike = [(30, 46), (45, 60)]  # covers x30..60 (WORLD), fragmented
    # 'o' (last of hello) sits at x[24,30] -> no overlap
    assert _union_coverage(24, 30, strike) == 0.0
    # 'W' (first of WORLD) sits at x[30,36] -> fully covered
    assert _union_coverage(30, 36, strike) == 1.0


def test_union_coverage_partial_overlap_below_threshold():
    # A char only half over a decoration stays below the 0.7/0.8 thresholds.
    assert _union_coverage(0, 10, [(0, 4)]) == 0.4
