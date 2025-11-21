"""Font styling utilities shared between text elements and character prep."""

from __future__ import annotations

from typing import Any, Mapping, Optional


def _normalize_fontname(data: Mapping[str, Any]) -> str:
    """Return the best available font name for a character or element."""
    real_name = data.get("real_fontname")
    if real_name:
        return str(real_name)
    fallback = data.get("fontname") or data.get("font") or ""
    return str(fallback)


def detect_bold_style(data: Mapping[str, Any]) -> bool:
    """Heuristically determine whether the given character/element is bold."""
    fontname = _normalize_fontname(data).lower()
    if "bold" in fontname or "black" in fontname or fontname.endswith("-b"):
        return True

    flags = data.get("flags")
    if isinstance(flags, int) and (flags & 4) != 0:
        return True

    stemv = _numeric_value(data, "stemv", "StemV")
    if stemv is not None and stemv > 120:
        return True

    weight = _numeric_value(data, "weight", "FontWeight")
    if weight is not None and weight >= 700:
        return True

    render_mode = data.get("render_mode")
    if isinstance(render_mode, int) and render_mode == 2:
        return True

    stroke_width = _numeric_value(data, "stroke_width", "lineWidth")
    if stroke_width is not None and stroke_width > 0:
        return True

    return False


def detect_italic_style(data: Mapping[str, Any]) -> bool:
    """Heuristically determine whether the given character/element is italic."""
    fontname = _normalize_fontname(data).lower()
    if "italic" in fontname or "oblique" in fontname or fontname.endswith("-i"):
        return True

    flags = data.get("flags")
    if isinstance(flags, int) and (flags & 64) != 0:
        return True

    italic_angle = _numeric_value(data, "italic_angle", "ItalicAngle")
    if italic_angle is not None and italic_angle != 0:
        return True

    return False


def resolve_fontname(data: Mapping[str, Any]) -> str:
    """Public helper for resolving font names consistently."""
    return _normalize_fontname(data)


def _numeric_value(data: Mapping[str, Any], *keys: str) -> Optional[float]:
    """Fetch the first present numeric value for the provided keys."""
    for key in keys:
        value = data.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


__all__ = ["detect_bold_style", "detect_italic_style", "resolve_fontname"]
