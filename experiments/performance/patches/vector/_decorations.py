"""Shared vector decoration prototype implementation."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from experiments.performance import vector_metrics as metrics
from experiments.performance.vector_helpers import element_array, prepared_char_array
from natural_pdf.core.decoration_detector import (
    HIGHLIGHT_DEFAULTS,
    STRIKE_DEFAULTS,
    UNDERLINE_DEFAULTS,
)


def annotate_chars_vectorized(
    detector: Any, char_dicts: list[dict[str, Any]], *, reuse: bool
) -> None:
    if not char_dicts:
        return

    label = "vector.decorations_page_store.chars" if reuse else "vector.decorations_ephemeral.chars"
    chars = (
        prepared_char_array(detector._page, char_dicts, label=label)
        if reuse
        else element_array(char_dicts, label=label)
    )
    n = len(char_dicts)
    strike = np.zeros(n, dtype=bool)
    underline = np.zeros(n, dtype=bool)
    highlight = np.zeros(n, dtype=bool)
    highlight_color: list[Any] = [None] * n

    valid = chars.valid & (chars.width > 0) & (chars.height > 0)
    if not np.any(valid):
        _write_char_flags(char_dicts, strike, underline, highlight, highlight_color)
        return

    _mark_strike(detector, chars, valid, strike)
    _mark_underline(detector, chars, valid, underline)
    _mark_highlight(detector, chars, valid, highlight, highlight_color)
    chars.strike[:] = strike
    chars.underline[:] = underline
    chars.highlight[:] = highlight
    _write_char_flags(char_dicts, strike, underline, highlight, highlight_color)


def propagate_to_words_vectorized(
    detector: Any,
    word_elements: list[Any],
    prepared_char_dicts: list[dict[str, Any]],
    *,
    reuse: bool,
) -> None:
    if not prepared_char_dicts:
        return

    label = (
        "vector.decorations_page_store.propagate"
        if reuse
        else "vector.decorations_ephemeral.propagate"
    )
    chars = (
        prepared_char_array(detector._page, prepared_char_dicts, label=label)
        if reuse
        else element_array(prepared_char_dicts, label=label)
    )

    for word in word_elements:
        indices = getattr(word, "_char_indices", None)
        if indices:
            index_array = np.asarray(
                [idx for idx in indices if 0 <= idx < len(prepared_char_dicts)],
                dtype=int,
            )
        else:
            char_dicts = getattr(word, "_char_dicts", None) or []
            index_array = np.asarray(
                [
                    idx
                    for idx, char_dict in enumerate(prepared_char_dicts)
                    if char_dict in char_dicts
                ],
                dtype=int,
            )

        total = len(index_array)
        if total == 0:
            word._obj["strike"] = False
            word._obj["underline"] = False
            word._obj["highlight"] = False
            continue

        word._obj["strike"] = bool(np.count_nonzero(chars.strike[index_array]) / total >= 0.6)
        word._obj["underline"] = bool(np.count_nonzero(chars.underline[index_array]) / total >= 0.6)
        word._obj["highlight"] = bool(np.count_nonzero(chars.highlight[index_array]) / total >= 0.6)
        if word._obj["highlight"]:
            colors: dict[Any, int] = {}
            for idx in index_array:
                char_dict = prepared_char_dicts[int(idx)]
                if char_dict.get("highlight") and char_dict.get("highlight_color") is not None:
                    color = char_dict["highlight_color"]
                    try:
                        color = tuple(color) if isinstance(color, (list, tuple)) else color
                    except Exception:
                        pass
                    colors[color] = colors.get(color, 0) + 1
            if colors:
                word._obj["highlight_color"] = max(colors.items(), key=lambda item: item[1])[0]

    metrics.count(f"{label}.word_count", len(word_elements))


def _write_char_flags(
    char_dicts: Sequence[dict[str, Any]],
    strike: np.ndarray,
    underline: np.ndarray,
    highlight: np.ndarray,
    highlight_color: Sequence[Any],
) -> None:
    for idx, char_dict in enumerate(char_dicts):
        char_dict["strike"] = bool(strike[idx])
        char_dict["underline"] = bool(underline[idx])
        char_dict["highlight"] = bool(highlight[idx])
        if highlight[idx] and highlight_color[idx] is not None:
            char_dict["highlight_color"] = highlight_color[idx]


def _horizontal_line_candidates(
    detector: Any, defaults: dict[str, float]
) -> list[tuple[float, float, float, float]]:
    candidates: list[tuple[float, float, float, float]] = []
    horiz_tol = defaults["horiz_tol"]
    raw_lines = list(getattr(detector._page._page, "lines", []))
    raw_rects = list(getattr(detector._page._page, "rects", []))
    for line in raw_lines:
        y0 = min(line.get("y0", 0), line.get("y1", 0))
        y1 = max(line.get("y0", 0), line.get("y1", 0))
        if abs(y1 - y0) <= horiz_tol:
            candidates.append((line.get("x0", 0), y0, line.get("x1", 0), y1))

    page_height = detector._page.height
    for rect in raw_rects:
        y0_raw = min(rect.get("y0", 0), rect.get("y1", 0))
        y1_raw = max(rect.get("y0", 0), rect.get("y1", 0))
        if (y1_raw - y0_raw) <= defaults["thickness_tol"]:
            y0 = page_height - y1_raw
            y1 = page_height - y0_raw
            candidates.append((rect.get("x0", 0), y0, rect.get("x1", 0), y1))
    return candidates


def _mark_strike(detector: Any, chars: Any, valid: np.ndarray, strike: np.ndarray) -> None:
    candidates = _horizontal_line_candidates(detector, STRIKE_DEFAULTS)
    metrics.count("vector.decorations.strike_candidates", len(candidates))
    if not candidates:
        return
    mid_y0 = chars.top + STRIKE_DEFAULTS["band_top_frac"] * chars.height
    mid_y1 = chars.top + STRIKE_DEFAULTS["band_bottom_frac"] * chars.height
    for lx0, ly0, lx1, ly1 in candidates:
        overlap = np.minimum(chars.x1, lx1) - np.maximum(chars.x0, lx0)
        mask = valid & (ly0 >= (mid_y0 - 1.0)) & (ly1 <= (mid_y1 + 1.0))
        mask &= overlap > 0
        mask &= (overlap / chars.width) >= STRIKE_DEFAULTS["coverage_ratio"]
        strike |= mask
        metrics.byte_count("vector.decorations.mask_bytes", mask.nbytes)


def _mark_underline(detector: Any, chars: Any, valid: np.ndarray, underline: np.ndarray) -> None:
    candidates = _horizontal_line_candidates(detector, UNDERLINE_DEFAULTS)
    metrics.count("vector.decorations.underline_candidates", len(candidates))
    if not candidates:
        return
    band_top = chars.bottom - UNDERLINE_DEFAULTS["band_frac"] * chars.height
    band_bottom = chars.bottom + UNDERLINE_DEFAULTS["below_pad"]
    for lx0, ly0, lx1, ly1 in candidates:
        line_mid = (ly0 + ly1) / 2.0
        overlap = np.minimum(chars.x1, lx1) - np.maximum(chars.x0, lx0)
        mask = valid & (band_top <= line_mid) & (line_mid <= band_bottom)
        mask &= overlap > 0
        mask &= (overlap / chars.width) >= UNDERLINE_DEFAULTS["coverage_ratio"]
        underline |= mask
        metrics.byte_count("vector.decorations.mask_bytes", mask.nbytes)


def _mark_highlight(
    detector: Any,
    chars: Any,
    valid: np.ndarray,
    highlight: np.ndarray,
    highlight_color: list[Any],
) -> None:
    cfg = detector._page._parent._config.get("highlight_detection", {})
    height_min_ratio = cfg.get("height_min_ratio", HIGHLIGHT_DEFAULTS["height_min_ratio"])
    height_max_ratio = cfg.get("height_max_ratio", HIGHLIGHT_DEFAULTS["height_max_ratio"])
    coverage_ratio = cfg.get("coverage_ratio", HIGHLIGHT_DEFAULTS["coverage_ratio"])

    highlight_rects = []
    for rect in list(getattr(detector._page._page, "rects", [])):
        if rect.get("stroke", False) or not rect.get("fill", False):
            continue
        color = rect.get("non_stroking_color")
        if color is None:
            continue
        y0 = min(rect.get("y0", 0), rect.get("y1", 0))
        y1 = max(rect.get("y0", 0), rect.get("y1", 0))
        highlight_rects.append((rect.get("x0", 0), y0, rect.get("x1", 0), y1, y1 - y0, color))

    metrics.count("vector.decorations.highlight_candidates", len(highlight_rects))
    if not highlight_rects:
        return

    raw_height = chars.y1 - chars.y0
    raw_valid = valid & np.isfinite(chars.y0) & np.isfinite(chars.y1) & (raw_height > 0)
    for rx0, ry0, rx1, ry1, rect_height, color in highlight_rects:
        ratio = rect_height / raw_height
        overlap = np.minimum(chars.x1, rx1) - np.maximum(chars.x0, rx0)
        mask = raw_valid & (ratio >= height_min_ratio) & (ratio <= height_max_ratio)
        mask &= (chars.y0 + 1 >= ry0) & (chars.y1 - 1 <= ry1)
        mask &= overlap > 0
        mask &= (overlap / chars.width) >= coverage_ratio
        if np.any(mask):
            for idx in np.flatnonzero(mask):
                highlight_color[int(idx)] = color
        highlight |= mask
        metrics.byte_count("vector.decorations.mask_bytes", mask.nbytes)
