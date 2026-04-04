"""
Builder functions for .to_llm() — assembles sections into final output.

Each level (page, region, collection, element, pdf) has its own builder
that composes section renderers from to_llm_sections.py.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from natural_pdf.core.page import Page
    from natural_pdf.core.page_collection import PageCollection
    from natural_pdf.core.pdf import PDF
    from natural_pdf.elements.base import Element
    from natural_pdf.elements.element_collection import ElementCollection
    from natural_pdf.elements.region import Region


def page_to_llm(
    page: "Page",
    detail: str = "standard",
    include_text: bool = True,
    include_hints: str = "none",
    show_boundaries: bool = True,
) -> str:
    """Build LLM representation for a single page.

    detail: "brief" | "standard" | "full"
    include_text: whether to show text samples in style tiers
    include_hints: "none" | "descriptive" | "api"
    show_boundaries: show element boundary separators (┃) in layout preview
    """
    from natural_pdf.describe.to_llm_sections import (
        render_alignment,
        render_hints,
        render_layout_preview,
        render_lines,
        render_rectangles,
        render_styles,
        render_text_layer,
    )

    is_brief = detail == "brief"
    is_full = detail == "full"

    # Caps vary by detail level
    max_tiers = 3 if is_brief else 8
    max_samples = 0 if is_brief else (10 if is_full else 5)
    max_clusters = 2 if is_brief else 3
    preview_lines = 0 if is_brief else (30 if is_full else 15)

    parts = []
    parts.append(
        f"=== Page {page.number} ({page.width:.0f} x {page.height:.0f} pts, origin top-left) ==="
    )
    parts.append("")

    # TEXT LAYER — always included
    parts.append(render_text_layer(page))

    # LAYOUT PREVIEW — standard and full only
    if preview_lines > 0:
        parts.append("")
        parts.append(
            render_layout_preview(page, max_lines=preview_lines, show_boundaries=show_boundaries)
        )

    # STYLES & CONTENT — always included, detail controls caps
    parts.append("")
    show_text = include_text and max_samples > 0
    parts.append(
        render_styles(
            page,
            include_text=show_text,
            max_tiers=max_tiers,
            max_samples=max_samples,
            max_clusters=max_clusters,
        )
    )

    # ALIGNMENT — always included
    parts.append("")
    parts.append(render_alignment(page))

    # LINES — always included (counts only at brief, full positions at standard+)
    parts.append("")
    parts.append(render_lines(page, counts_only=is_brief))

    # RECTANGLES — always included
    parts.append("")
    parts.append(render_rectangles(page, detail=detail))

    # HINTS — only if requested
    if include_hints in ("descriptive", "api"):
        parts.append("")
        parts.append(render_hints(page, style=include_hints))

    return "\n".join(parts)


def region_to_llm(
    region: "Region",
    detail: str = "standard",
    include_text: bool = True,
    include_hints: str = "none",
) -> str:
    """Build LLM representation for a region — microscope view."""
    from natural_pdf.describe.to_llm_sections import (
        _cluster_values,
        render_lines,
        render_rectangles,
    )

    page = region.page
    parts = []
    parts.append(
        f"=== Region ({region.x0:.0f}, {region.top:.0f})-({region.x1:.0f}, {region.bottom:.0f}) "
        f"on page {page.number} ({region.width:.0f}x{region.height:.0f} pts) ==="
    )
    parts.append("")

    # Layout text within region
    text = region.extract_text(layout=True) if include_text else None
    if text and text.strip():
        text_lines = text.split("\n")
        # Strip empty leading/trailing
        while text_lines and not text_lines[-1].strip():
            text_lines.pop()
        while text_lines and not text_lines[0].strip():
            text_lines.pop(0)
        parts.append("TEXT CONTENT")
        for line in text_lines[:30]:
            parts.append(f"  {line}")
        if len(text_lines) > 30:
            parts.append("  ...")
    else:
        parts.append("TEXT CONTENT\n  (no text in region)")

    # Local styles
    text_elements = region.find_all("text")
    if text_elements:
        from collections import defaultdict

        tiers: dict = defaultdict(list)
        for el in text_elements:
            size = round(el.size * 2) / 2
            bold = getattr(el, "bold", False)
            key = (size, bold)
            tiers[key].append(el)

        parts.append("")
        parts.append("STYLES")
        for (size, bold), elements in sorted(tiers.items(), key=lambda x: (-x[0][0], -len(x[1]))):
            style_str = f"{size:g}pt" + (" bold" if bold else "")
            samples = [f'"{el.text}"' for el in elements[:5]]
            parts.append(f"  {style_str} — {len(elements)} elements: {', '.join(samples)}")

    return "\n".join(parts)


def collection_to_llm(
    collection: "ElementCollection",
    detail: str = "standard",
    include_text: bool = True,
    include_hints: str = "none",
) -> str:
    """Build LLM representation for an element collection."""
    from natural_pdf.describe.to_llm_sections import _cluster_values

    parts = []
    count = len(collection)
    parts.append(f"=== ElementCollection: {count} elements ===")

    if count == 0:
        return "\n".join(parts)

    parts.append("")

    # Extent
    x0 = min(el.x0 for el in collection)
    top = min(el.top for el in collection)
    x1 = max(el.x1 for el in collection)
    bottom = max(el.bottom for el in collection)
    parts.append("EXTENT")
    parts.append(f"  x={x0:.0f}-{x1:.0f}, y={top:.0f}-{bottom:.0f}")

    # Style distribution (for text elements)
    text_els = [el for el in collection if hasattr(el, "size")]
    if text_els:
        sizes = set(round(el.size * 2) / 2 for el in text_els)
        bold_count = sum(1 for el in text_els if getattr(el, "bold", False))
        parts.append("")
        parts.append("STYLES")
        if len(sizes) == 1:
            size = next(iter(sizes))
            style = f"all {size:g}pt"
            if bold_count == len(text_els):
                style += " bold"
            elif bold_count > 0:
                style += f" ({bold_count} bold)"
            parts.append(f"  {style}")
        else:
            parts.append(f"  sizes: {', '.join(f'{s:g}pt' for s in sorted(sizes))}")
            if bold_count:
                parts.append(f"  {bold_count} bold")

    # Text samples
    if include_text and text_els:
        parts.append("")
        parts.append("TEXT SAMPLES")
        samples = [f'"{el.text}"' for el in text_els[:10] if hasattr(el, "text")]
        extra = len(text_els) - 10
        if extra > 0:
            samples.append(f"+{extra} more")
        parts.append(f"  {', '.join(samples)}")

    # Alignment
    if text_els:
        x_positions = [el.x0 for el in text_els]
        clusters = _cluster_values(x_positions, tolerance=3.0)
        significant = [(c, idx) for c, idx in clusters if len(idx) >= 2]
        if significant:
            parts.append("")
            parts.append("ALIGNMENT")
            for center, indices in significant:
                parts.append(f"  x≈{center:.0f} ({len(indices)} elements)")

    return "\n".join(parts)


def element_to_llm(
    element: "Element",
    detail: str = "standard",
    include_text: bool = True,
    include_hints: str = "none",
    vertical_radius: float = 30,
) -> str:
    """Build LLM representation for a single element — card view with neighbors.

    Neighbor search: full page width on the same horizontal band (±element height),
    then elements directly above/below within vertical_radius pts.
    """
    text = getattr(element, "text", "")
    size = getattr(element, "size", None)
    bold = getattr(element, "bold", False)
    italic = getattr(element, "italic", False)
    source = getattr(element, "source", "native")

    style_parts = []
    if size is not None:
        style_parts.append(f"{size:.0f}pt")
    if bold:
        style_parts.append("bold")
    if italic:
        style_parts.append("italic")
    style_str = " ".join(style_parts)

    parts = []
    parts.append(f'=== "{text}" — {style_str} at ({element.x0:.0f}, {element.top:.0f}) ===')
    parts.append(f"  size: {element.width:.0f}x{element.height:.0f} pts, source: {source}")

    if detail == "brief":
        return "\n".join(parts)

    page = getattr(element, "page", None) or getattr(element, "_page", None)
    if page is not None:
        all_text = page.find_all("text")
        el_height = element.bottom - element.top
        y_tolerance = max(el_height, 5)  # same-line band

        same_line = []
        above = []
        below = []

        for el in all_text:
            if el is element:
                continue
            # Check same horizontal band (full page width)
            el_cy = (el.top + el.bottom) / 2
            my_cy = (element.top + element.bottom) / 2
            y_diff = el_cy - my_cy

            if abs(y_diff) <= y_tolerance:
                # Same line — classify as left/right
                gap = el.x0 - element.x1 if el.x0 > element.x1 else element.x0 - el.x1
                direction = "right" if el.x0 >= element.x1 else "left"
                same_line.append((abs(gap), direction, max(0, gap), el))
            elif 0 < y_diff <= vertical_radius:
                gap = el.top - element.bottom
                below.append((gap, el))
            elif -vertical_radius <= y_diff < 0:
                gap = element.top - el.bottom
                above.append((gap, el))

        def _fmt(el, gap, direction):
            el_size = getattr(el, "size", 0)
            el_bold = " bold" if getattr(el, "bold", False) else ""
            return f'  {direction}: "{el.text}" ({gap:.0f}pt gap, {el_size:.0f}pt{el_bold})'

        has_neighbors = same_line or above or below
        if has_neighbors:
            parts.append("")
            parts.append("NEARBY")

            # Same line — sorted by distance
            same_line.sort(key=lambda x: x[0])
            for dist, direction, gap, el in same_line[:6]:
                parts.append(_fmt(el, gap, direction))

            # Above — sorted by proximity (closest first)
            above.sort(key=lambda x: x[0])
            for gap, el in above[:3]:
                parts.append(_fmt(el, gap, "above"))

            # Below — sorted by proximity
            below.sort(key=lambda x: x[0])
            for gap, el in below[:3]:
                parts.append(_fmt(el, gap, "below"))

        # Same style on page
        if size is not None:
            same_style = [
                el
                for el in all_text
                if abs(getattr(el, "size", 0) - size) < 0.5 and getattr(el, "bold", False) == bold
            ]
            if len(same_style) > 1:
                parts.append("")
                parts.append(f"SAME STYLE ON PAGE: {len(same_style)} elements")

    return "\n".join(parts)


def pdf_to_llm(
    pdf_or_collection,
    detail: str = "standard",
    include_text: bool = True,
    include_hints: str = "none",
) -> str:
    """Build LLM representation for a PDF or PageCollection (routing view)."""
    from natural_pdf.core.pdf import PDF

    if isinstance(pdf_or_collection, PDF):
        source = getattr(pdf_or_collection, "source", "PDF")
        pages = pdf_or_collection.pages
    else:
        source = "PageCollection"
        pages = pdf_or_collection

    parts = []
    parts.append(f"=== {source} ({len(pages)} pages) ===")
    parts.append("")

    for page in pages:
        text_els = page.find_all("text")
        word_count = len(text_els)
        h_lines = sum(1 for l in page.lines if l.is_horizontal)
        v_lines = sum(1 for l in page.lines if l.is_vertical)
        images = page.images
        img_coverage = 0
        if images:
            page_area = page.width * page.height
            img_coverage = (
                sum(i.width * i.height for i in images) / page_area * 100 if page_area else 0
            )

        # Get anchors: up to 3 from largest/boldest text
        anchors = []
        if text_els:
            by_size = sorted(
                text_els, key=lambda e: (-getattr(e, "size", 0), -int(getattr(e, "bold", False)))
            )
            seen = set()
            for el in by_size:
                t = el.text.strip()
                if t and t not in seen and len(t) > 1:
                    anchors.append(f'"{t}"')
                    seen.add(t)
                    if len(anchors) >= 3:
                        break

        # Build page summary line
        summary_parts = [f"{word_count} words"]
        if h_lines or v_lines:
            summary_parts.append(f"{h_lines}h+{v_lines}v lines")
        if img_coverage > 50:
            summary_parts.append(f"image {img_coverage:.0f}%")

        anchor_str = f" — {', '.join(anchors)}" if anchors else ""
        parts.append(f"  Page {page.number}: {', '.join(summary_parts)}{anchor_str}")

    return "\n".join(parts)
