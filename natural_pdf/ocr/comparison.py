"""OCR comparison utilities: text normalization, consensus, classification, and result objects."""

from __future__ import annotations

import difflib
import logging
import re
import unicodedata
from dataclasses import dataclass, field
from statistics import median
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


def _load_font(size: int):
    """Load a TrueType font at the given size, with robust fallback chain."""
    from PIL import ImageFont

    for name in (
        "Arial.ttf",
        "DejaVuSans.ttf",
        "Helvetica.ttc",
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(name, size)
        except (OSError, IOError):
            continue
    # Pillow 10+ supports size param on load_default
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


# ---------------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------------


def normalize_text(text: str, mode: str = "collapse") -> str:
    """Normalize text for comparison.

    Args:
        text: Input text.
        mode: "collapse" (NFKC + collapse whitespace, default),
              "strict" (no changes),
              "ignore" (strip all whitespace).
    """
    if mode == "strict":
        return text
    # Always apply NFKC normalization for non-strict modes
    text = unicodedata.normalize("NFKC", text)
    if mode == "ignore":
        return re.sub(r"\s+", "", text)
    # "collapse" — collapse runs of whitespace to single space, strip ends
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------------------
# Consensus computation
# ---------------------------------------------------------------------------


def _edit_distance_ratio(a: str, b: str) -> float:
    """Return normalized edit distance: 1.0 - SequenceMatcher.ratio().

    0.0 means identical, 1.0 means completely different.
    """
    if not a and not b:
        return 0.0
    return 1.0 - difflib.SequenceMatcher(None, a, b).ratio()


def compute_consensus(texts: Dict[str, str]) -> Optional[str]:
    """Compute consensus text (medoid) from multiple engine outputs.

    Returns the text with the lowest average edit distance to all others,
    or None if no texts are provided.
    """
    if not texts:
        return None
    values = list(texts.values())
    if len(values) == 1:
        return values[0]

    # Find the medoid — text with minimum total distance to others
    best_text = values[0]
    best_total = float("inf")
    for candidate in values:
        total = sum(_edit_distance_ratio(candidate, other) for other in values)
        if total < best_total:
            best_total = total
            best_text = candidate
    return best_text


# ---------------------------------------------------------------------------
# Region classification
# ---------------------------------------------------------------------------


def classify_region(
    edit_distances: Dict[str, float],
    present_engines: int,
    total_engines: int,
    text_length: int = 0,
) -> str:
    """Classify a comparison region based on edit distances to consensus.

    Args:
        edit_distances: engine_name → normalized edit distance to consensus.
        present_engines: Number of engines that produced text in this region.
        total_engines: Total number of engines in the comparison.
        text_length: Length of the consensus text (for length-aware thresholds).

    Returns:
        "agreement", "near_miss", or "catastrophic".
    """
    if present_engines == 0:
        return "catastrophic"

    # Any engine missing entirely is catastrophic
    if present_engines < total_engines:
        return "catastrophic"

    if not edit_distances:
        return "agreement"

    worst = max(edit_distances.values())

    # Length-aware thresholds: short strings are more sensitive
    catastrophic_threshold = 0.20 if text_length < 10 else 0.25

    if worst <= 0.05:
        return "agreement"
    elif worst <= catastrophic_threshold:
        return "near_miss"
    else:
        return "catastrophic"


# ---------------------------------------------------------------------------
# Outlier detection
# ---------------------------------------------------------------------------


def find_outlier(edit_distances: Dict[str, float]) -> Optional[str]:
    """Identify the engine that's the odd one out, if any.

    Returns the engine name if one engine's distance is ≥2× the median
    of the others, and the median of others is > 0. Returns None otherwise.
    """
    if len(edit_distances) < 3:
        # With only 2 engines, outlier detection isn't meaningful
        return None

    items = list(edit_distances.items())
    for engine, dist in items:
        others = [d for e, d in items if e != engine]
        med = median(others)
        if med > 0 and dist >= 2.0 * med:
            return engine
        # Also flag if others all agree (med ≈ 0) but this engine disagrees
        if med <= 0.01 and dist > 0.05:
            return engine
    return None


# ---------------------------------------------------------------------------
# Character-level diff rendering (HTML)
# ---------------------------------------------------------------------------


def render_char_diff_html(reference: str, engine_text: str) -> str:
    """Render character-level diff as HTML — highlight disagreements only.

    Characters that differ from the reference get a yellow highlight.
    Deletions (chars in reference but not in engine) are silently omitted.
    """
    if reference == engine_text:
        return f'<span style="color:#222">{_html_escape(engine_text)}</span>'
    if not engine_text:
        return '<span style="color:#999">[missing]</span>'

    sm = difflib.SequenceMatcher(None, reference, engine_text)
    parts: list[str] = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            parts.append(f'<span style="color:#222">{_html_escape(engine_text[j1:j2])}</span>')
        else:
            # replace or insert — highlight the engine's actual chars
            chunk = engine_text[j1:j2]
            if chunk:
                parts.append(
                    f'<span style="background:#fff3cd;color:#222;font-weight:bold">'
                    f"{_html_escape(chunk)}</span>"
                )
            # delete — skip entirely (don't show placeholders)
    return "".join(parts)


def _html_escape(text: str) -> str:
    """Minimal HTML escaping."""
    return (
        text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
    )


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ComparisonRegion:
    """One aligned region across all engines."""

    bbox: Tuple[float, float, float, float]
    texts: Dict[str, str]  # engine_name → raw text
    normalized_texts: Dict[str, str]  # engine_name → normalized text
    confidences: Dict[str, Optional[float]]  # engine_name → avg confidence
    consensus: Optional[str]
    classification: str  # "agreement" | "near_miss" | "catastrophic"
    edit_distances: Dict[str, float]  # engine_name → distance to consensus
    outlier_engine: Optional[str]
    elements: Dict[str, list] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# OcrComparison result object
# ---------------------------------------------------------------------------


class OcrComparison:
    """Result of comparing multiple OCR engines on a single page.

    Immutable after construction. Provides visualization methods
    for exploring differences between engines.
    """

    def __init__(
        self,
        *,
        page: Any,
        engines: List[str],
        failed_engines: Dict[str, str],
        regions: List[ComparisonRegion],
        engine_elements: Dict[str, list],
        strategy_used: str,
        diagnostics: Dict[str, Any],
        runtimes: Dict[str, float],
        resolution: int,
        normalize_mode: str,
    ):
        self._page = page
        self._engines = list(engines)
        self._failed_engines = dict(failed_engines)
        self._regions = list(regions)
        self._engine_elements = dict(engine_elements)
        self._strategy_used = strategy_used
        self._diagnostics = dict(diagnostics)
        self._runtimes = dict(runtimes)
        self._resolution = resolution
        self._normalize_mode = normalize_mode

    @property
    def page(self) -> Any:
        return self._page

    @property
    def engines(self) -> List[str]:
        return list(self._engines)

    @property
    def failed_engines(self) -> Dict[str, str]:
        return dict(self._failed_engines)

    @property
    def regions(self) -> List[ComparisonRegion]:
        return list(self._regions)

    @property
    def strategy_used(self) -> str:
        return self._strategy_used

    @property
    def diagnostics(self) -> Dict[str, Any]:
        return dict(self._diagnostics)

    @property
    def runtimes(self) -> Dict[str, float]:
        return dict(self._runtimes)

    # -- Visualization methods (Phase 2) --

    def summary(self):
        """Return a pandas DataFrame with per-engine statistics."""
        import pandas as pd

        rows = []
        for engine in self._engines:
            counts = {"agreement": 0, "near_miss": 0, "catastrophic": 0}
            missing = 0
            confs = []
            for r in self._regions:
                if engine not in r.texts:
                    missing += 1
                    counts["catastrophic"] += 1
                else:
                    counts[r.classification] = counts.get(r.classification, 0) + 1
                    if r.confidences.get(engine) is not None:
                        confs.append(r.confidences[engine])
            rows.append(
                {
                    "engine": engine,
                    "regions_found": len(self._regions) - missing,
                    "agreement": counts["agreement"],
                    "near_miss": counts["near_miss"],
                    "catastrophic": counts["catastrophic"],
                    "missing": missing,
                    "avg_confidence": round(sum(confs) / len(confs), 3) if confs else None,
                    "runtime_s": round(self._runtimes.get(engine, 0), 2),
                }
            )
        return pd.DataFrame(rows)

    def show(
        self,
        *,
        render_text: bool = True,
        resolution: int | None = None,
        columns: int | None = None,
        mode: str = "grid",
    ) -> Any:
        """Render per-engine OCR overlays on the page image.

        Args:
            render_text: Whether to draw OCR text inside boxes.
            resolution: Render DPI (default: comparison resolution).
            columns: Number of grid columns (default: auto).
            mode: "grid" (default) for side-by-side panels, "toggle" for
                CSS hover-swap between engines (2 engines only).

        Returns:
            PIL Image for "grid" mode, HTML widget for "toggle" mode.
        """
        if mode == "toggle" and len(self._engines) == 2:
            return self._show_toggle(render_text=render_text, resolution=resolution)
        if mode == "toggle" and len(self._engines) != 2:
            import logging

            logging.getLogger(__name__).info(
                "Toggle mode requires 2 engines, falling back to grid."
            )

        from PIL import Image, ImageDraw, ImageFont

        from natural_pdf.utils.visualization import render_plain_page

        res = resolution or self._resolution
        base = render_plain_page(self._page, res)
        scale = res / 72.0

        # Assign colors to engines — outline + very subtle fill
        colors = [
            (66, 133, 244),  # blue
            (234, 67, 53),  # red
            (52, 168, 83),  # green
            (251, 188, 4),  # yellow
            (171, 71, 188),  # purple
            (0, 172, 193),  # teal
            (255, 112, 67),  # orange
            (141, 110, 99),  # brown
        ]

        panels = []
        for idx, engine in enumerate(self._engines):
            panel = base.copy().convert("RGBA")
            overlay = Image.new("RGBA", panel.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)
            rgb = colors[idx % len(colors)]
            fill_color = rgb + (230,)  # nearly opaque so text is readable
            border_color = rgb + (255,)  # solid outline

            elements = self._engine_elements.get(engine, [])
            for elem in elements:
                x0 = elem.x0 * scale
                top = elem.top * scale
                x1 = elem.x1 * scale
                bottom = elem.bottom * scale
                draw.rectangle([x0, top, x1, bottom], fill=fill_color, outline=border_color)

                if render_text and hasattr(elem, "text") and elem.text:
                    box_h = bottom - top
                    font_size = max(8, int(box_h * 0.7))
                    font = _load_font(font_size)
                    draw.text((x0 + 1, top), elem.text, fill=(255, 255, 255, 255), font=font)

            panel = Image.alpha_composite(panel, overlay)

            # Add engine label at top
            label_overlay = Image.new("RGBA", panel.size, (0, 0, 0, 0))
            label_draw = ImageDraw.Draw(label_overlay)
            label_size = max(18, int(res / 7))
            label_font = _load_font(label_size)
            banner_h = label_size + 8
            label_draw.rectangle([0, 0, panel.width, banner_h], fill=(255, 255, 255, 200))
            label_draw.text((8, 4), engine, fill=rgb + (255,), font=label_font)
            panel = Image.alpha_composite(panel, label_overlay)
            panels.append(panel.convert("RGB"))

        # Arrange in grid
        return self._grid_images(panels, columns=columns)

    def _show_toggle(self, *, render_text: bool = True, resolution: int | None = None) -> Any:
        """CSS hover-toggle: single image that swaps between 2 engines on hover."""
        import base64
        import io

        from PIL import Image, ImageDraw, ImageFont

        from natural_pdf.utils.visualization import render_plain_page

        res = resolution or self._resolution
        base = render_plain_page(self._page, res)
        scale = res / 72.0

        colors = [(66, 133, 244), (234, 67, 53)]
        panels_b64 = []

        for idx, engine in enumerate(self._engines):
            panel = base.copy().convert("RGBA")
            overlay = Image.new("RGBA", panel.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)
            rgb = colors[idx]
            fill = rgb + (230,)
            border = rgb + (255,)

            for elem in self._engine_elements.get(engine, []):
                x0, top = elem.x0 * scale, elem.top * scale
                x1, bottom = elem.x1 * scale, elem.bottom * scale
                draw.rectangle([x0, top, x1, bottom], fill=fill, outline=border)
                if render_text and hasattr(elem, "text") and elem.text:
                    box_h = bottom - top
                    font_size = max(8, int(box_h * 0.7))
                    font = _load_font(font_size)
                    draw.text((x0 + 1, top), elem.text, fill=(255, 255, 255, 255), font=font)

            panel = Image.alpha_composite(panel, overlay).convert("RGB")
            buf = io.BytesIO()
            panel.save(buf, format="PNG")
            panels_b64.append(base64.b64encode(buf.getvalue()).decode("ascii"))

        e0, e1 = self._engines[0], self._engines[1]
        uid = f"ocr-toggle-{id(self)}"

        html = f"""
        <style>
            #{uid} {{
                position: relative;
                width: {base.width}px;
                height: {base.height}px;
                background-image: url(data:image/png;base64,{panels_b64[0]});
                background-size: cover;
                cursor: crosshair;
            }}
            #{uid}:hover {{
                background-image: url(data:image/png;base64,{panels_b64[1]}) !important;
            }}
            #{uid} .ocr-toggle-label {{
                position: absolute; top: 4px; left: 4px;
                background: rgba(255,255,255,0.85); padding: 2px 8px;
                font-family: sans-serif; font-size: 12px; color: #222;
                border-radius: 3px; pointer-events: none;
            }}
            #{uid} .ocr-toggle-label::after {{ content: "{e0}"; }}
            #{uid}:hover .ocr-toggle-label::after {{ content: "{e1}"; }}
        </style>
        <div id="{uid}">
            <span class="ocr-toggle-label"></span>
        </div>
        """
        return _HtmlDisplay(html)

    def heatmap(
        self, *, engine: str | None = None, resolution: int | None = None
    ) -> "PIL.Image.Image":
        """Render a disagreement heatmap overlay on the page image."""
        from PIL import Image, ImageDraw

        from natural_pdf.utils.visualization import render_plain_page

        res = resolution or self._resolution
        base = render_plain_page(self._page, res).convert("RGBA")
        scale = res / 72.0

        overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        classification_colors = {
            "agreement": (76, 175, 80, 40),  # faint green
            "near_miss": (255, 152, 0, 100),  # orange
            "catastrophic": (244, 67, 54, 150),  # red
        }

        for region in self._regions:
            if engine is not None:
                # Per-engine mode: show this engine's distance
                if engine not in region.texts:
                    color = (158, 158, 158, 120)  # grey = missing
                else:
                    dist = region.edit_distances.get(engine, 0)
                    if dist <= 0.05:
                        continue  # skip agreement regions
                    elif dist <= 0.25:
                        color = (255, 152, 0, int(100 * min(dist / 0.25, 1)))
                    else:
                        color = (244, 67, 54, 150)
            else:
                color = classification_colors.get(region.classification)
                if color is None or region.classification == "agreement":
                    continue  # skip agreement in default mode

            x0, top, x1, bottom = region.bbox
            draw.rectangle(
                [x0 * scale, top * scale, x1 * scale, bottom * scale],
                fill=color,
                outline=color[:3] + (200,),
            )

        result = Image.alpha_composite(base, overlay)

        # Add legend
        from PIL import ImageFont

        legend_font_size = max(18, int(res / 7))
        legend_h = legend_font_size + 8
        legend_w = result.width
        font = _load_font(legend_font_size)

        items = [
            ("Agreement", (76, 175, 80)),
            ("Near-miss", (255, 152, 0)),
            ("Catastrophic", (244, 67, 54)),
        ]
        if engine:
            items.append(("Missing", (158, 158, 158)))

        legend = Image.new("RGBA", (legend_w, legend_h * len(items) + 20), (255, 255, 255, 240))
        legend_draw = ImageDraw.Draw(legend)
        swatch = max(14, legend_font_size - 2)
        for i, (label, rgb) in enumerate(items):
            y = i * legend_h + 10
            legend_draw.rectangle([16, y, 16 + swatch, y + swatch], fill=rgb + (200,))
            legend_draw.text((16 + swatch + 10, y), label, fill=(0, 0, 0, 255), font=font)

        # Stack result + legend
        final = Image.new("RGB", (result.width, result.height + legend.height), (255, 255, 255))
        final.paste(legend.convert("RGB"), (0, 0))
        final.paste(result.convert("RGB"), (0, legend.height))
        return final

    def coverage(self, *, resolution: int | None = None) -> "PIL.Image.Image":
        """Render a detection coverage map — where each engine found text.

        Unlike heatmap() which shows disagreement, this shows presence/absence:
        which engine detected text in each region, regardless of what it read.
        """
        from PIL import Image, ImageDraw, ImageFont

        from natural_pdf.utils.visualization import render_plain_page

        res = resolution or self._resolution
        base = render_plain_page(self._page, res).convert("RGBA")
        scale = res / 72.0

        overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        # For N=2: engine-specific colors for "only" regions
        # For N≥3: shade by count of engines present
        is_pairwise = len(self._engines) == 2

        if is_pairwise:
            engine_a, engine_b = self._engines[0], self._engines[1]
            color_a = (66, 133, 244, 120)  # blue — only engine A
            color_b = (234, 67, 53, 120)  # red — only engine B
            color_both = (200, 200, 200, 30)  # faint grey — both present

            for region in self._regions:
                has_a = engine_a in region.texts and region.texts[engine_a]
                has_b = engine_b in region.texts and region.texts[engine_b]

                if has_a and has_b:
                    color = color_both
                elif has_a:
                    color = color_a
                elif has_b:
                    color = color_b
                else:
                    continue

                x0, top, x1, bottom = region.bbox
                draw.rectangle(
                    [x0 * scale, top * scale, x1 * scale, bottom * scale],
                    fill=color,
                    outline=color[:3] + (180,),
                )
        else:
            # N≥3: shade by engine count
            n = len(self._engines)
            for region in self._regions:
                present = sum(1 for e in self._engines if e in region.texts and region.texts[e])
                if present == 0:
                    continue
                if present == n:
                    color = (200, 200, 200, 30)  # all present — faint
                elif present == 1:
                    color = (244, 67, 54, 150)  # only 1 — red
                else:
                    # partial — orange intensity by coverage
                    alpha = int(40 + 110 * (1 - present / n))
                    color = (255, 152, 0, alpha)

                x0, top, x1, bottom = region.bbox
                draw.rectangle(
                    [x0 * scale, top * scale, x1 * scale, bottom * scale],
                    fill=color,
                    outline=color[:3] + (180,),
                )

        result = Image.alpha_composite(base, overlay)

        # Legend
        legend_font_size = max(18, int(res / 7))
        font = _load_font(legend_font_size)

        if is_pairwise:
            items = [
                (f"{engine_a} only", (66, 133, 244)),
                (f"{engine_b} only", (234, 67, 53)),
                ("Both detected", (200, 200, 200)),
            ]
        else:
            items = [
                ("All engines", (200, 200, 200)),
                ("Some engines", (255, 152, 0)),
                ("1 engine only", (244, 67, 54)),
            ]

        legend_h = legend_font_size + 8
        swatch = max(14, legend_font_size - 2)
        legend = Image.new("RGBA", (result.width, legend_h * len(items) + 20), (255, 255, 255, 240))
        legend_draw = ImageDraw.Draw(legend)
        for idx, (label, rgb) in enumerate(items):
            y = idx * legend_h + 4
            legend_draw.rectangle([8, y, 8 + swatch, y + swatch], fill=rgb + (200,))
            legend_draw.text((8 + swatch + 8, y), label, fill=(0, 0, 0, 255), font=font)

        final = Image.new("RGB", (result.width, result.height + legend.height), (255, 255, 255))
        final.paste(legend.convert("RGB"), (0, 0))
        final.paste(result.convert("RGB"), (0, legend.height))
        return final

    def _crop_region_base64(
        self, region: ComparisonRegion, base_image: "PIL.Image.Image", scale: float
    ) -> str:
        """Crop a region from the page image and return as base64 PNG."""
        import base64
        import io

        try:
            img_w = int(base_image.width)
            img_h = int(base_image.height)
        except (TypeError, AttributeError):
            return ""

        x0, top, x1, bottom = region.bbox
        px0 = max(0, int(x0 * scale) - 2)
        ptop = max(0, int(top * scale) - 2)
        px1 = min(img_w, int(x1 * scale) + 2)
        pbottom = min(img_h, int(bottom * scale) + 2)

        if px1 <= px0 or pbottom <= ptop:
            return ""

        try:
            crop = base_image.crop((px0, ptop, px1, pbottom))
            buf = io.BytesIO()
            crop.save(buf, format="PNG")
            return base64.b64encode(buf.getvalue()).decode("ascii")
        except Exception:
            return ""

    def diff(self, *, only: str | None = None, show_image: bool = True) -> Any:
        """Return an HTML diff table of per-region text comparisons.

        Uses a vertical card layout: image crop on the left, engine texts
        stacked vertically in the middle, metadata on the right.
        For 2-engine comparisons, drops the consensus column and diffs
        the second engine against the first.

        Args:
            only: Filter by classification — "near_miss", "catastrophic",
                  "disagreement" (near_miss + catastrophic), or "all".
                  Default: show only disagreements (hide agreements).
            show_image: Include a cropped image of each region (default True).
        """
        show_filter = only or "disagreement"
        is_pairwise = len(self._engines) == 2

        # Pre-render page image for region crops
        base_image = None
        scale = 1.0
        if show_image:
            from natural_pdf.utils.visualization import render_plain_page

            base_image = render_plain_page(self._page, self._resolution)
            scale = self._resolution / 72.0

        rows_html: list[str] = []
        for i, region in enumerate(self._regions):
            cls = region.classification
            if show_filter == "disagreement" and cls == "agreement":
                continue
            if show_filter not in ("all", "disagreement") and cls != show_filter:
                continue

            cls_colors = {
                "agreement": "#e8f5e9",
                "near_miss": "#fff3e0",
                "catastrophic": "#ffebee",
            }
            bg = cls_colors.get(cls, "#ffffff")

            # --- Image crop cell (left) ---
            img_cell = '<td style="background:#fff;vertical-align:top;padding:2px;width:1px"></td>'
            if show_image and base_image is not None:
                b64 = self._crop_region_base64(region, base_image, scale)
                if b64:
                    img_cell = (
                        f'<td style="background:#fff;vertical-align:top;padding:2px;width:1px">'
                        f'<img src="data:image/png;base64,{b64}" '
                        f'style="min-height:40px;max-height:120px;max-width:350px;display:block">'
                        f"</td>"
                    )

            # --- Metadata cell (narrow left) ---
            outlier_html = ""
            if region.outlier_engine:
                outlier_html = (
                    f'<br><span style="color:#e65100;font-size:10px">{region.outlier_engine}</span>'
                )
            meta_cell = (
                f'<td style="vertical-align:top;width:80px;background:{bg};color:#222;padding:6px">'
                f'<b style="font-size:15px">#{i + 1}</b><br>'
                f'<span style="font-size:14px">{cls}</span>'
                f"{outlier_html}"
                f"</td>"
            )

            # --- Engine texts cell (stacked vertically) ---
            if is_pairwise:
                # 2 engines: both highlighted against each other
                eng_a = self._engines[0]
                eng_b = self._engines[1]
                text_a = region.texts.get(eng_a)
                text_b = region.texts.get(eng_b)

                lines: list[str] = []
                for eng, txt, other_txt in [(eng_a, text_a, text_b), (eng_b, text_b, text_a)]:
                    if txt is None:
                        lines.append(
                            f'<div style="margin-bottom:4px">'
                            f'<span style="color:#666;font-size:13px">{eng}:</span> '
                            f'<span style="color:#999">[missing]</span></div>'
                        )
                    elif other_txt is None:
                        # Other engine missing — show plain (nothing to diff against)
                        lines.append(
                            f'<div style="margin-bottom:4px">'
                            f'<span style="color:#666;font-size:13px">{eng}:</span> '
                            f'<span style="color:#222">{_html_escape(txt)}</span></div>'
                        )
                    else:
                        # Diff this engine's text against the other engine's text
                        diff_html = render_char_diff_html(other_txt, txt)
                        lines.append(
                            f'<div style="margin-bottom:4px">'
                            f'<span style="color:#666;font-size:13px">{eng}:</span> '
                            f"{diff_html}</div>"
                        )
                text_content = "".join(lines)
            else:
                # N≥3 engines: diff each against consensus
                consensus = region.consensus or ""
                lines = []
                for engine in self._engines:
                    raw = region.texts.get(engine)
                    if raw is None:
                        lines.append(
                            f'<div style="margin-bottom:2px">'
                            f'<span style="color:#666;font-size:13px">{engine}:</span> '
                            f'<span style="color:#999">[missing]</span></div>'
                        )
                    else:
                        diff_html = render_char_diff_html(consensus, raw)
                        lines.append(
                            f'<div style="margin-bottom:2px">'
                            f'<span style="color:#666;font-size:13px">{engine}:</span> '
                            f"{diff_html}</div>"
                        )
                lines.append(
                    f'<div style="margin-top:4px;border-top:1px solid #eee;padding-top:2px">'
                    f'<span style="color:#666;font-size:10px">consensus:</span> '
                    f'<b style="color:#222">{_html_escape(consensus)}</b></div>'
                )
                text_content = "".join(lines)

            text_cell = (
                f'<td style="background:#fff;vertical-align:top;padding:6px;'
                f'white-space:pre-wrap;word-break:break-word">'
                f"{text_content}</td>"
            )

            rows_html.append(f"<tr>{img_cell}{meta_cell}{text_cell}</tr>")

        html = f"""
        <style>
            .ocr-diff-table {{ font-family: monospace; font-size: 15px; border-collapse: collapse; color: #222; width: 100%; table-layout: auto; }}
            .ocr-diff-table td {{ border: 1px solid #ccc; text-align: left; color: #222; }}
        </style>
        <table class="ocr-diff-table">
            <tbody>{''.join(rows_html)}</tbody>
        </table>
        """

        return _HtmlDisplay(html)

    def apply(self, engine: str, *, replace: bool = True) -> Any:
        """Persist the chosen engine's OCR elements to the page.

        Args:
            engine: Engine name whose results to keep.
            replace: If True, remove existing OCR elements first.
        """
        if engine not in self._engine_elements:
            available = ", ".join(self._engines)
            raise ValueError(f"Engine '{engine}' not found. Available: {available}")

        elements = self._engine_elements[engine]
        if not elements:
            logger.warning("Engine '%s' produced no elements to apply.", engine)
            return self._page

        # Build OCR result dicts from the stored elements
        ocr_results = []
        for elem in elements:
            ocr_results.append(
                {
                    "bbox": (elem.x0, elem.top, elem.x1, elem.bottom),
                    "text": elem.text,
                    "confidence": elem.confidence if elem.confidence != 1.0 else None,
                }
            )

        if replace:
            self._page.services.ocr.remove_ocr_elements(self._page)

        self._page.services.ocr.create_text_elements_from_ocr(
            self._page,
            ocr_results,
            scale_x=1.0,  # already in PDF coords
            scale_y=1.0,
            engine_name=engine,
        )
        logger.info("Applied %d elements from '%s' to page.", len(ocr_results), engine)
        return self._page

    def loupe(self, *, resolution: int | None = None, zoom: int = 3) -> Any:
        """Interactive magnifier widget — hover to zoom, see per-engine text.

        Shows the page image with a cursor-following magnified patch.
        When the cursor enters a comparison region, the loupe also shows
        each engine's text and the classification for that region.
        Click to pin/unpin the loupe.

        Args:
            resolution: Render DPI (default: comparison resolution).
            zoom: Magnification factor (default: 3).

        Returns:
            HTML widget for Jupyter display.
        """
        import base64
        import io
        import json

        from natural_pdf.utils.visualization import render_plain_page

        res = resolution or self._resolution
        base = render_plain_page(self._page, res)
        scale = res / 72.0

        # Encode base image
        buf = io.BytesIO()
        base.save(buf, format="PNG")
        img_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

        # Engine colors for bbox highlights
        engine_colors = {}
        color_palette = [
            "rgba(66,133,244,0.5)",  # blue
            "rgba(234,67,53,0.5)",  # red
            "rgba(52,168,83,0.5)",  # green
            "rgba(251,188,4,0.5)",  # yellow
            "rgba(171,71,188,0.5)",  # purple
        ]
        for idx, eng in enumerate(self._engines):
            engine_colors[eng] = color_palette[idx % len(color_palette)]

        # Build region data for JS — include per-engine element bboxes
        region_data = []
        for i, region in enumerate(self._regions):
            x0, top, x1, bottom = region.bbox
            entry = {
                "id": i + 1,
                "x0": round(x0 * scale),
                "y0": round(top * scale),
                "x1": round(x1 * scale),
                "y1": round(bottom * scale),
                "cls": region.classification,
                "texts": {},
                "boxes": {},
            }
            for engine in self._engines:
                txt = region.texts.get(engine)
                entry["texts"][engine] = txt if txt else "[missing]"
                # Per-engine element bboxes for highlighting
                elems = region.elements.get(engine, [])
                entry["boxes"][engine] = [
                    {
                        "x0": round(getattr(e, "x0", 0) * scale),
                        "y0": round(getattr(e, "top", 0) * scale),
                        "x1": round(getattr(e, "x1", 0) * scale),
                        "y1": round(getattr(e, "bottom", 0) * scale),
                    }
                    for e in elems
                ]
            region_data.append(entry)

        regions_json = json.dumps(region_data)
        engine_colors_json = json.dumps(engine_colors)
        uid = f"ocr-loupe-{id(self)}"
        img_w, img_h = base.width, base.height
        loupe_w, loupe_h = 440, 200

        html = f"""
        <style>
            #{uid}-container {{
                position: relative;
                display: inline-block;
                cursor: crosshair;
            }}
            #{uid}-container img {{
                display: block;
                max-width: none;
            }}
            #{uid}-loupe {{
                display: none;
                position: absolute;
                pointer-events: none;
                width: {loupe_w}px;
                border: 2px solid #333;
                border-radius: 4px;
                background-color: #fff;
                box-shadow: 0 2px 8px rgba(0,0,0,0.3);
                z-index: 100;
                overflow: hidden;
            }}
            #{uid}-loupe.pinned {{
                pointer-events: auto;
                border-color: #1976d2;
            }}
            #{uid}-mag {{
                width: {loupe_w}px;
                height: {loupe_h}px;
                background-image: url(data:image/png;base64,{img_b64});
                background-repeat: no-repeat;
                background-size: {img_w * zoom}px {img_h * zoom}px;
            }}
            #{uid}-info {{
                padding: 6px 8px;
                font-family: monospace;
                font-size: 14px;
                color: #222;
                background: #f9f9f9;
                border-top: 1px solid #ddd;
                max-height: 120px;
                overflow-y: auto;
            }}
            #{uid}-info .engine-label {{
                color: #666;
                font-size: 12px;
            }}
            #{uid}-info .cls-badge {{
                display: inline-block;
                padding: 1px 6px;
                border-radius: 2px;
                font-size: 13px;
                margin-bottom: 3px;
            }}
        </style>
        <div id="{uid}-container">
            <img src="data:image/png;base64,{img_b64}" width="{img_w}" height="{img_h}" />
            <svg id="{uid}-svg" style="position:absolute;top:0;left:0;width:{img_w}px;height:{img_h}px;pointer-events:none"></svg>
            <div id="{uid}-loupe">
                <div id="{uid}-mag"></div>
                <div id="{uid}-info"></div>
            </div>
        </div>
        <script>
        (function() {{
            var container = document.getElementById("{uid}-container");
            var loupe = document.getElementById("{uid}-loupe");
            var mag = document.getElementById("{uid}-mag");
            var info = document.getElementById("{uid}-info");
            var svg = document.getElementById("{uid}-svg");
            var regions = {regions_json};
            var engineColors = {engine_colors_json};
            var zoom = {zoom};
            var imgW = {img_w}, imgH = {img_h};
            var loupeW = {loupe_w}, loupeH = {loupe_h};
            var pinned = false;
            var lastRegionId = -1;

            container.addEventListener("mousemove", function(e) {{
                if (pinned) return;
                var rect = container.getBoundingClientRect();
                var x = e.clientX - rect.left;
                var y = e.clientY - rect.top;
                updateLoupe(x, y);
            }});

            container.addEventListener("mouseenter", function() {{
                if (!pinned) loupe.style.display = "block";
            }});

            container.addEventListener("mouseleave", function() {{
                if (!pinned) {{
                    loupe.style.display = "none";
                    clearHighlights();
                }}
            }});

            container.addEventListener("click", function(e) {{
                if (pinned) {{
                    pinned = false;
                    loupe.classList.remove("pinned");
                    loupe.style.display = "none";
                    clearHighlights();
                }} else {{
                    pinned = true;
                    loupe.classList.add("pinned");
                }}
            }});

            function clearHighlights() {{
                svg.innerHTML = "";
                lastRegionId = -1;
            }}

            function drawHighlights(region) {{
                svg.innerHTML = "";
                for (var eng in region.boxes) {{
                    var color = engineColors[eng] || "rgba(128,128,128,0.4)";
                    var boxes = region.boxes[eng];
                    for (var j = 0; j < boxes.length; j++) {{
                        var b = boxes[j];
                        var rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
                        rect.setAttribute("x", b.x0);
                        rect.setAttribute("y", b.y0);
                        rect.setAttribute("width", b.x1 - b.x0);
                        rect.setAttribute("height", b.y1 - b.y0);
                        rect.setAttribute("fill", color);
                        rect.setAttribute("stroke", color.replace("0.5", "0.9"));
                        rect.setAttribute("stroke-width", "1.5");
                        svg.appendChild(rect);
                    }}
                }}
            }}

            function updateLoupe(x, y) {{
                // Position loupe offset from cursor
                var lx = x + 15;
                var ly = y + 15;
                if (lx + loupeW + 20 > container.offsetWidth) lx = x - loupeW - 15;
                if (ly + loupeH + 140 > container.offsetHeight) ly = y - loupeH - 15;
                loupe.style.left = lx + "px";
                loupe.style.top = ly + "px";
                loupe.style.display = "block";

                // Magnifier background position
                var bgX = -(x * zoom - loupeW / 2);
                var bgY = -(y * zoom - loupeH / 2);
                mag.style.backgroundPosition = bgX + "px " + bgY + "px";

                // Find region under cursor
                var html = "";
                var foundRegion = null;
                for (var i = 0; i < regions.length; i++) {{
                    var r = regions[i];
                    if (x >= r.x0 && x <= r.x1 && y >= r.y0 && y <= r.y1) {{
                        foundRegion = r;
                        var clsColors = {{"agreement":"#e8f5e9","near_miss":"#fff3e0","catastrophic":"#ffebee"}};
                        var bg = clsColors[r.cls] || "#eee";
                        html += '<span class="cls-badge" style="background:' + bg + '">#' + r.id + ' ' + r.cls + '</span><br>';
                        for (var eng in r.texts) {{
                            var ec = engineColors[eng] || "#666";
                            html += '<span class="engine-label" style="color:' + ec.replace("0.5","1") + '">' + eng + ':</span> ' + r.texts[eng] + '<br>';
                        }}
                        break;
                    }}
                }}
                info.innerHTML = html || '<span style="color:#999">No region here</span>';

                // Update bbox highlights
                if (foundRegion && foundRegion.id !== lastRegionId) {{
                    lastRegionId = foundRegion.id;
                    drawHighlights(foundRegion);
                }} else if (!foundRegion && lastRegionId !== -1) {{
                    clearHighlights();
                }}
            }}
        }})();
        </script>
        """
        return _HtmlDisplay(html)

    def _repr_html_(self) -> str:
        """Trade-off summary for Jupyter display."""
        n_regions = len(self._regions)
        counts = {"agreement": 0, "near_miss": 0, "catastrophic": 0}
        missing_per_engine: Dict[str, int] = {e: 0 for e in self._engines}
        for r in self._regions:
            counts[r.classification] = counts.get(r.classification, 0) + 1
            for e in self._engines:
                if e not in r.texts or not r.texts[e]:
                    missing_per_engine[e] += 1

        agree_pct = round(100 * counts["agreement"] / n_regions) if n_regions else 0

        engines_str = " vs ".join(self._engines)
        failed_str = ""
        if self._failed_engines:
            failed_str = (
                f'<div style="color:#c62828;margin-top:4px">'
                f'Failed: {", ".join(self._failed_engines.keys())}</div>'
            )

        # Runtime comparison
        runtime_parts = []
        for e in self._engines:
            t = self._runtimes.get(e, 0)
            runtime_parts.append(f"{e} {t:.1f}s")
        runtime_str = ", ".join(runtime_parts)
        if len(self._engines) == 2:
            t0 = self._runtimes.get(self._engines[0], 0)
            t1 = self._runtimes.get(self._engines[1], 0)
            delta = abs(t1 - t0)
            faster = self._engines[0] if t0 < t1 else self._engines[1]
            runtime_str += f" ({faster} is {delta:.1f}s faster)"

        # Missing text lines
        missing_lines = []
        for e, count in missing_per_engine.items():
            if count > 0:
                missing_lines.append(
                    f"{e} missed text in {count} region{'s' if count != 1 else ''}"
                )

        missing_html = ""
        if missing_lines:
            missing_html = "<br>".join(missing_lines)
            missing_html = f'<div style="margin-top:4px;color:#c62828">{missing_html}</div>'

        return f"""
        <div style="font-family: sans-serif; font-size: 16px; padding: 14px; border: 1px solid #ddd; border-radius: 4px; color: #222; background: #fff;">
            <b style="font-size:18px">OCR Comparison: {engines_str}</b>{failed_str}
            <div style="margin-top:8px">
                Total regions: {n_regions}<br>
                <span style="color:#4caf50">■</span> Agreement: {counts['agreement']} ({agree_pct}%)
                <span style="color:#ff9800">■</span> Near-miss: {counts['near_miss']}
                <span style="color:#f44336">■</span> Catastrophic: {counts['catastrophic']}
            </div>
            {missing_html}
            <div style="margin-top:6px;color:#555">Runtime: {runtime_str}</div>
            <div style="margin-top:8px;color:#888;font-size:13px">
                .diff() .show() .heatmap() .coverage() .loupe() .summary() .apply(engine=)
            </div>
        </div>
        """

    @staticmethod
    def _grid_images(images: list, columns: int | None = None, gap: int = 5) -> "PIL.Image.Image":
        """Arrange PIL images in a grid."""
        from PIL import Image

        if not images:
            return Image.new("RGB", (100, 100), (255, 255, 255))

        n = len(images)
        cols = columns or max(1, int(n**0.5 + 0.5))
        rows = (n + cols - 1) // cols

        max_w = max(img.width for img in images)
        max_h = max(img.height for img in images)

        grid_w = cols * max_w + (cols - 1) * gap
        grid_h = rows * max_h + (rows - 1) * gap

        grid = Image.new("RGB", (grid_w, grid_h), (255, 255, 255))
        for idx, img in enumerate(images):
            r, c = divmod(idx, cols)
            x = c * (max_w + gap)
            y = r * (max_h + gap)
            grid.paste(img, (x, y))

        return grid


class _HtmlDisplay:
    """Wrapper that renders as HTML in Jupyter."""

    def __init__(self, html: str):
        self._html = html

    def _repr_html_(self) -> str:
        return self._html

    def __repr__(self) -> str:
        return "<OcrComparison diff — view in Jupyter for HTML rendering>"
