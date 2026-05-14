"""Render cropped specs by asking pypdfium to rasterize only the crop area."""

from __future__ import annotations

from contextlib import contextmanager

METADATA = {
    "track": "rendering",
    "candidate": "direct_pdfium_crop",
    "cache_only": False,
    "hypothesis": "Crop-heavy render workflows can avoid full-page rasterization.",
}


def _render_crop_direct(page, crop_bbox, resolution):
    from natural_pdf.utils.visualization import pdf_render_lock, pypdfium2

    if pypdfium2 is None:
        raise RuntimeError("pypdfium2 is not available")

    x0, top, x1, bottom = crop_bbox
    crop = (x0, page.height - bottom, page.width - x1, top)

    with pdf_render_lock:
        doc = pypdfium2.PdfDocument(page._page.pdf.stream)
        pdf_page = doc[page.index]
        bitmap = pdf_page.render(scale=resolution / 72.0, crop=crop)
        image = bitmap.to_pil().convert("RGB")
        pdf_page.close()
        doc.close()

    return image


@contextmanager
def install():
    from natural_pdf.core.highlighting_service import HighlightingService, Image, render_ocr_overlay
    from natural_pdf.utils.visualization import (
        create_colorbar,
        create_legend,
        merge_images_with_legend,
    )

    original = HighlightingService._render_spec

    def patched(
        self,
        spec,
        resolution,
        width,
        labels,
        label_format,
        render_ocr=False,
        legend_position="right",
        spec_index=0,
        **kwargs,
    ):
        if not spec.crop_bbox:
            return original(
                self,
                spec,
                resolution,
                width,
                labels,
                label_format,
                render_ocr=render_ocr,
                legend_position=legend_position,
                spec_index=spec_index,
                **kwargs,
            )

        page = spec.page
        if not hasattr(page, "width") or not hasattr(page, "height"):
            return original(
                self,
                spec,
                resolution,
                width,
                labels,
                label_format,
                render_ocr=render_ocr,
                legend_position=legend_position,
                spec_index=spec_index,
                **kwargs,
            )

        target_width = width
        base_resolution = resolution if resolution is not None else 150
        if target_width is not None and page.width > 0:
            width_resolution = (target_width / page.width) * 72
            actual_resolution = max(width_resolution, base_resolution)
        else:
            actual_resolution = base_resolution
        scale_factor = actual_resolution / 72

        try:
            page_image = _render_crop_direct(page, spec.crop_bbox, actual_resolution)
        except Exception:
            return original(
                self,
                spec,
                resolution,
                width,
                labels,
                label_format,
                render_ocr=render_ocr,
                legend_position=legend_position,
                spec_index=spec_index,
                **kwargs,
            )

        if spec.highlights:
            page_image = self._apply_spec_highlights(
                page_image,
                spec.highlights,
                page,
                scale_factor,
                labels=labels,
                label_format=label_format,
                spec_index=spec_index,
                crop_offset=spec.crop_bbox[:2],
            )

        if render_ocr:
            page_image = render_ocr_overlay(page, page_image, scale_factor)

        if target_width is not None and page_image.width != target_width:
            aspect = page_image.height / page_image.width
            target_height = int(target_width * aspect)
            page_image = page_image.resize((target_width, target_height), Image.Resampling.LANCZOS)

        if spec.highlights and labels:
            quantitative_metadata = None
            for highlight_data in spec.highlights:
                if (
                    "quantitative_metadata" in highlight_data
                    and highlight_data["quantitative_metadata"]
                ):
                    quantitative_metadata = highlight_data["quantitative_metadata"]
                    break

            if quantitative_metadata:
                colorbar = create_colorbar(
                    values=quantitative_metadata["values"],
                    colormap=quantitative_metadata["colormap"],
                    bins=quantitative_metadata["bins"],
                    orientation=(
                        "horizontal" if legend_position in ["top", "bottom"] else "vertical"
                    ),
                )
                page_image = merge_images_with_legend(
                    page_image, colorbar, position=legend_position
                )
            else:
                spec_labels = {}
                for hl in spec.highlights:
                    label = hl.get("label")
                    color = hl.get("color")
                    if label and color:
                        processed_color = self._process_color_input(color)
                        if processed_color:
                            spec_labels[label] = processed_color
                        else:
                            spec_labels[label] = self._color_manager.get_color(label=label)

                if spec_labels:
                    legend = create_legend(spec_labels)
                    if legend:
                        page_image = merge_images_with_legend(
                            page_image, legend, position=legend_position
                        )

        return page_image

    HighlightingService._render_spec = patched
    try:
        yield
    finally:
        HighlightingService._render_spec = original
