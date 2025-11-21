from __future__ import annotations

import logging
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Union,
)

from natural_pdf.core.render_spec import RenderSpec

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from PIL.Image import Image as PILImage

    from natural_pdf.core.highlighting_service import HighlightingService

ColorInput = Union[str, Tuple[int, int, int]]

logger = logging.getLogger(__name__)


class SupportsVisualizable(Protocol):
    """Protocol describing the Visualizable contract required by RenderingService."""

    def _get_render_specs(
        self,
        mode: Literal["show", "render"] = "show",
        **kwargs: Any,
    ) -> List[RenderSpec]: ...

    def _get_highlighter(self) -> "HighlightingService": ...

    def _resolve_image_resolution(self, requested: Optional[float]) -> float: ...


class RenderingService:
    """Centralizes Visualizable.show/render/export behavior for all hosts."""

    def __init__(self, _context):
        self._context = _context

    def show(
        self,
        host: SupportsVisualizable,
        *,
        resolution: Optional[float] = None,
        width: Optional[int] = None,
        color: Optional[ColorInput] = None,
        labels: bool = True,
        label_format: Optional[str] = None,
        highlights: Optional[Union[List[Dict[str, Any]], bool]] = None,
        legend_position: str = "right",
        annotate: Optional[Union[str, Sequence[str]]] = None,
        layout: Optional[Literal["stack", "grid", "single"]] = None,
        stack_direction: Literal["vertical", "horizontal"] = "vertical",
        gap: int = 5,
        columns: Optional[int] = 6,
        limit: Optional[int] = 30,
        crop: Union[bool, int, str, Any] = False,
        crop_bbox: Optional[Tuple[float, float, float, float]] = None,
        **kwargs: Any,
    ) -> Optional["PILImage"]:
        annotate_list = self._normalize_annotate(annotate)
        columns = self._resolve_columns_alias(columns, kwargs, guard_value=6)

        if limit is not None:
            kwargs["max_pages"] = limit

        specs = host._get_render_specs(
            mode="show",
            color=color,
            highlights=highlights,
            crop=crop,
            crop_bbox=crop_bbox,
            annotate=annotate_list,
            **kwargs,
        )

        self._ensure_specs(host, specs, "show")

        if limit is not None and len(specs) > limit:
            logger.debug(
                "Limiting render specs for %s.show() to %s entries from %s",
                host.__class__.__name__,
                limit,
                len(specs),
            )
            specs = specs[:limit]

        if layout is None:
            layout = "grid" if len(specs) > 1 else "single"

        highlighter = host._get_highlighter()
        effective_resolution = host._resolve_image_resolution(resolution)

        return highlighter.unified_render(
            specs=specs,
            resolution=effective_resolution,
            width=width,
            labels=labels,
            label_format=label_format,
            legend_position=legend_position,
            layout=layout,
            stack_direction=stack_direction,
            gap=gap,
            columns=columns,
            **kwargs,
        )

    def render(
        self,
        host: SupportsVisualizable,
        *,
        resolution: Optional[float] = None,
        width: Optional[int] = None,
        layout: Literal["stack", "grid", "single"] = "stack",
        stack_direction: Literal["vertical", "horizontal"] = "vertical",
        gap: int = 5,
        columns: Optional[int] = None,
        crop: Union[bool, int, str, Any] = False,
        crop_bbox: Optional[Tuple[float, float, float, float]] = None,
        **kwargs: Any,
    ) -> Optional["PILImage"]:
        columns = self._resolve_columns_alias(columns, kwargs, guard_value=None)
        kwargs.pop("labels", None)

        specs = host._get_render_specs(mode="render", crop=crop, crop_bbox=crop_bbox, **kwargs)
        self._ensure_specs(host, specs, "render")

        highlighter = host._get_highlighter()
        effective_resolution = host._resolve_image_resolution(resolution)

        return highlighter.unified_render(
            specs=specs,
            resolution=effective_resolution,
            width=width,
            labels=False,
            layout=layout,
            stack_direction=stack_direction,
            gap=gap,
            columns=columns,
            **kwargs,
        )

    def export(
        self,
        host: SupportsVisualizable,
        path: Union[str, Path],
        *,
        resolution: Optional[float] = None,
        width: Optional[int] = None,
        layout: Literal["stack", "grid", "single"] = "stack",
        stack_direction: Literal["vertical", "horizontal"] = "vertical",
        gap: int = 5,
        columns: Optional[int] = None,
        crop: Union[bool, Literal["content"]] = False,
        crop_bbox: Optional[Tuple[float, float, float, float]] = None,
        format: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        image = self.render(
            host,
            resolution=resolution,
            width=width,
            layout=layout,
            stack_direction=stack_direction,
            gap=gap,
            columns=columns,
            crop=crop,
            crop_bbox=crop_bbox,
            **kwargs,
        )

        if image is None:
            raise ValueError(f"No image generated by {host.__class__.__name__}.render()")

        path_obj = Path(path)
        if format is None:
            format = path_obj.suffix.lstrip(".").upper()
            if format == "JPG":
                format = "JPEG"

        save_kwargs: Dict[str, Any] = {}
        if format == "JPEG":
            save_kwargs["quality"] = kwargs.get("quality", 95)
        elif format == "PNG":
            save_kwargs["compress_level"] = kwargs.get("compress_level", 6)

        image.save(path_obj, format=format, **save_kwargs)
        logger.info("Exported %s to %s", host.__class__.__name__, path_obj)

    def render_preview(
        self,
        host: SupportsVisualizable,
        *,
        page_index: int,
        temporary_highlights: List[Dict[str, Any]],
        resolution: Optional[float] = 144,
        width: Optional[int] = None,
        labels: bool = True,
        legend_position: str = "right",
        render_ocr: bool = False,
        crop_bbox: Optional[Tuple[float, float, float, float]] = None,
        **kwargs: Any,
    ) -> Optional["PILImage"]:
        """Render temporary highlights for preview purposes."""
        highlighter = host._get_highlighter()
        preview_kwargs = dict(kwargs)
        if width is not None:
            preview_kwargs["width"] = width
        effective_resolution = host._resolve_image_resolution(resolution)
        return highlighter.render_preview(
            page_index=page_index,
            temporary_highlights=temporary_highlights,
            resolution=effective_resolution,
            labels=labels,
            legend_position=legend_position,
            render_ocr=render_ocr,
            crop_bbox=crop_bbox,
            **preview_kwargs,
        )

    @staticmethod
    def _normalize_annotate(
        annotate: Optional[Union[str, Sequence[str]]],
    ) -> Optional[Sequence[str]]:
        if annotate is None:
            return None
        if isinstance(annotate, str):
            return [annotate]
        return annotate

    @staticmethod
    def _resolve_columns_alias(
        columns: Optional[int],
        kwargs: Dict[str, Any],
        *,
        guard_value: Optional[int],
    ) -> Optional[int]:
        alias = kwargs.pop("cols", None)
        if alias is None:
            return columns

        if guard_value is None and columns is None:
            columns = alias
            logger.info("Using 'cols' parameter as alias for 'columns': %s", columns)
        elif guard_value is not None and columns == guard_value:
            columns = alias
            logger.info("Using 'cols' parameter as alias for 'columns': %s", columns)
        else:
            # User explicitly set columns; warn about ignored alias
            logger.debug("'cols' alias ignored because columns were explicitly set.")
        return columns

    @staticmethod
    def _ensure_specs(host: SupportsVisualizable, specs: List[RenderSpec], mode: str) -> None:
        if not specs:
            raise RuntimeError(f"{host.__class__.__name__}.{mode}() generated no render specs")
