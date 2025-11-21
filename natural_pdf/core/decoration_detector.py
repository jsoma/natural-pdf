from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)

STRIKE_DEFAULTS = {
    "thickness_tol": 1.5,
    "horiz_tol": 1.0,
    "coverage_ratio": 0.7,
    "band_top_frac": 0.35,
    "band_bottom_frac": 0.65,
}

UNDERLINE_DEFAULTS = {
    "thickness_tol": 1.5,
    "horiz_tol": 1.0,
    "coverage_ratio": 0.8,
    "band_frac": 0.25,
    "below_pad": 0.7,
}

HIGHLIGHT_DEFAULTS = {
    "height_min_ratio": 0.6,
    "height_max_ratio": 2.0,
    "coverage_ratio": 0.6,
    "color_saturation_min": 0.4,
    "color_value_min": 0.4,
}


class DecorationDetector:
    """Detects visual decorations (strike, underline, highlight) and propagates them."""

    def __init__(self, page) -> None:
        self._page = page

    def annotate_chars(self, char_dicts: List[Dict[str, Any]]) -> None:
        if not char_dicts:
            return
        self._mark_strikethrough_chars(char_dicts)
        self._mark_underline_chars(char_dicts)
        self._mark_highlight_chars(char_dicts)

    def propagate_to_words(
        self, word_elements: List[Any], prepared_char_dicts: List[Dict[str, Any]]
    ) -> None:
        if not prepared_char_dicts:
            return

        for w in word_elements:
            strike_chars = 0
            ul_chars = 0
            hl_chars = 0
            total_chars = 0

            if getattr(w, "_char_indices", None):
                indices = [idx for idx in w._char_indices if 0 <= idx < len(prepared_char_dicts)]
                total_chars = len(indices)
                for idx in indices:
                    ch = prepared_char_dicts[idx]
                    if ch.get("strike"):
                        strike_chars += 1
                    if ch.get("underline"):
                        ul_chars += 1
                    if ch.get("highlight"):
                        hl_chars += 1
            elif getattr(w, "_char_dicts", None):
                total_chars = len(w._char_dicts)
                for ch in w._char_dicts:
                    if ch.get("strike"):
                        strike_chars += 1
                    if ch.get("underline"):
                        ul_chars += 1
                    if ch.get("highlight"):
                        hl_chars += 1

            w._obj["strike"] = bool(total_chars) and (strike_chars / total_chars) >= 0.6
            w._obj["underline"] = bool(total_chars) and (ul_chars / total_chars) >= 0.6
            w._obj["highlight"] = bool(total_chars) and (hl_chars / total_chars) >= 0.6

            if w._obj.get("highlight"):
                color_counts: Dict[Any, int] = {}
                if getattr(w, "_char_indices", None):
                    source_iter = (
                        prepared_char_dicts[idx]
                        for idx in w._char_indices
                        if 0 <= idx < len(prepared_char_dicts)
                    )
                else:
                    source_iter = getattr(w, "_char_dicts", []) or []

                for chd in source_iter:
                    if chd.get("highlight") and chd.get("highlight_color") is not None:
                        col = chd["highlight_color"]
                        color_counts[col] = color_counts.get(col, 0) + 1

                if color_counts:
                    dominant_color = max(color_counts.items(), key=lambda t: t[1])[0]
                    try:
                        w._obj["highlight_color"] = (
                            tuple(dominant_color)
                            if isinstance(dominant_color, (list, tuple))
                            else dominant_color
                        )
                    except Exception:
                        w._obj["highlight_color"] = dominant_color

    # --- Internal helpers -------------------------------------------------
    def _mark_strikethrough_chars(
        self,
        char_dicts: List[Dict[str, Any]],
    ) -> None:
        raw_lines = list(getattr(self._page._page, "lines", []))
        raw_rects = list(getattr(self._page._page, "rects", []))
        candidates: List[Tuple[float, float, float, float]] = []

        horiz_tol = STRIKE_DEFAULTS["horiz_tol"]
        for ln in raw_lines:
            y0 = min(ln.get("y0", 0), ln.get("y1", 0))
            y1 = max(ln.get("y0", 0), ln.get("y1", 0))
            if abs(y1 - y0) <= horiz_tol:
                candidates.append((ln.get("x0", 0), y0, ln.get("x1", 0), y1))

        pg_height = self._page.height
        for rc in raw_rects:
            rb0 = rc.get("y0", 0)
            rb1 = rc.get("y1", 0)
            y0_raw = min(rb0, rb1)
            y1_raw = max(rb0, rb1)
            if (y1_raw - y0_raw) <= STRIKE_DEFAULTS["thickness_tol"]:
                y0 = pg_height - y1_raw
                y1 = pg_height - y0_raw
                candidates.append((rc.get("x0", 0), y0, rc.get("x1", 0), y1))

        if not candidates:
            for ch in char_dicts:
                ch.setdefault("strike", False)
            return

        for ch in char_dicts:
            ch.setdefault("strike", False)
            try:
                x0, top, x1, bottom = ch["x0"], ch["top"], ch["x1"], ch["bottom"]
            except KeyError:
                continue

            width = x1 - x0
            height = bottom - top
            if width <= 0 or height <= 0:
                continue

            mid_y0 = top + STRIKE_DEFAULTS["band_top_frac"] * height
            mid_y1 = top + STRIKE_DEFAULTS["band_bottom_frac"] * height

            for lx0, ly0, lx1, ly1 in candidates:
                if (ly0 >= (mid_y0 - 1.0)) and (ly1 <= (mid_y1 + 1.0)):
                    overlap = min(x1, lx1) - max(x0, lx0)
                    if overlap > 0 and (overlap / width) >= STRIKE_DEFAULTS["coverage_ratio"]:
                        ch["strike"] = True
                        break

    def _mark_underline_chars(
        self,
        char_dicts: List[Dict[str, Any]],
    ) -> None:
        raw_lines = list(getattr(self._page._page, "lines", []))
        raw_rects = list(getattr(self._page._page, "rects", []))
        candidates: List[Tuple[float, float, float, float]] = []

        horiz_tol = UNDERLINE_DEFAULTS["horiz_tol"]
        for ln in raw_lines:
            y0 = min(ln.get("y0", 0), ln.get("y1", 0))
            y1 = max(ln.get("y0", 0), ln.get("y1", 0))
            if abs(y1 - y0) <= horiz_tol:
                candidates.append((ln.get("x0", 0), y0, ln.get("x1", 0), y1))

        pg_height = self._page.height
        for rc in raw_rects:
            rb0 = rc.get("y0", 0)
            rb1 = rc.get("y1", 0)
            y0_raw = min(rb0, rb1)
            y1_raw = max(rb0, rb1)
            if (y1_raw - y0_raw) <= UNDERLINE_DEFAULTS["thickness_tol"]:
                y0 = pg_height - y1_raw
                y1 = pg_height - y0_raw
                candidates.append((rc.get("x0", 0), y0, rc.get("x1", 0), y1))

        if not candidates:
            for ch in char_dicts:
                ch.setdefault("underline", False)
            return

        for ch in char_dicts:
            ch.setdefault("underline", False)
            try:
                x0, top, x1, bottom = ch["x0"], ch["top"], ch["x1"], ch["bottom"]
            except KeyError:
                continue

            width = x1 - x0
            height = bottom - top
            if width <= 0 or height <= 0:
                continue

            band_top = bottom - UNDERLINE_DEFAULTS["band_frac"] * height
            band_bottom = bottom + UNDERLINE_DEFAULTS["below_pad"]

            for lx0, ly0, lx1, ly1 in candidates:
                line_mid = (ly0 + ly1) / 2.0
                if band_top <= line_mid <= band_bottom:
                    overlap = min(x1, lx1) - max(x0, lx0)
                    if overlap > 0 and (overlap / width) >= UNDERLINE_DEFAULTS["coverage_ratio"]:
                        ch["underline"] = True
                        break

    def _mark_highlight_chars(self, char_dicts: List[Dict[str, Any]]) -> None:
        cfg = self._page._parent._config.get("highlight_detection", {})
        height_min_ratio = cfg.get("height_min_ratio", HIGHLIGHT_DEFAULTS["height_min_ratio"])
        height_max_ratio = cfg.get("height_max_ratio", HIGHLIGHT_DEFAULTS["height_max_ratio"])
        coverage_ratio = cfg.get("coverage_ratio", HIGHLIGHT_DEFAULTS["coverage_ratio"])

        raw_rects = list(getattr(self._page._page, "rects", []))
        highlight_rects = []
        for rc in raw_rects:
            if rc.get("stroke", False) or not rc.get("fill", False):
                continue
            fill_col = rc.get("non_stroking_color")
            if fill_col is None:
                continue

            y0_rect = min(rc.get("y0", 0), rc.get("y1", 0))
            y1_rect = max(rc.get("y0", 0), rc.get("y1", 0))
            rheight = y1_rect - y0_rect
            highlight_rects.append(
                (rc.get("x0", 0), y0_rect, rc.get("x1", 0), y1_rect, rheight, fill_col)
            )

        if not highlight_rects:
            for ch in char_dicts:
                ch.setdefault("highlight", False)
            return

        for ch in char_dicts:
            ch.setdefault("highlight", False)
            try:
                x0_raw, y0_raw, x1_raw, y1_raw = ch["x0"], ch["y0"], ch["x1"], ch["y1"]
            except KeyError:
                continue

            width = x1_raw - x0_raw
            height = y1_raw - y0_raw
            if width <= 0 or height <= 0:
                continue

            for rx0, ry0, rx1, ry1, rheight, rcolor in highlight_rects:
                ratio = rheight / height if height else 0
                if ratio < height_min_ratio or ratio > height_max_ratio:
                    continue

                if not (y0_raw + 1 >= ry0 and y1_raw - 1 <= ry1):
                    continue

                overlap = min(x1_raw, rx1) - max(x0_raw, rx0)
                if overlap > 0 and (overlap / width) >= coverage_ratio:
                    ch["highlight"] = True
                    try:
                        ch["highlight_color"] = (
                            tuple(rcolor) if isinstance(rcolor, (list, tuple)) else rcolor
                        )
                    except Exception:
                        ch["highlight_color"] = rcolor
                    break
