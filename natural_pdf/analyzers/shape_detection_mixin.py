"""Compatibility stub for the legacy shape detection mixin.

The implementation now lives in ``natural_pdf.services._shape_detection_impl``
and is exposed to hosts through ``ShapeDetectionService`` (registered as the
``shapes`` service).  This module remains so that existing imports of
``ShapeDetectionMixin`` and ``LINE_DETECTION_PARAM_DEFAULTS`` keep working.
"""

from __future__ import annotations

from typing import Any

from natural_pdf.services._shape_detection_impl import LINE_DETECTION_PARAM_DEFAULTS

__all__ = ["LINE_DETECTION_PARAM_DEFAULTS", "ShapeDetectionMixin"]


class ShapeDetectionMixin:
    """
    Deprecated shim. Shape detection (lines, blobs) is implemented in
    ``natural_pdf.services._shape_detection_impl``; hosts reach it through
    ``self.services.shapes``. This class simply forwards to those functions
    for any code that still mixes it in or calls it directly.
    """

    def detect_lines(self, *args: Any, **kwargs: Any) -> "ShapeDetectionMixin":
        from natural_pdf.services import _shape_detection_impl as _impl

        _impl.detect_lines(self, *args, **kwargs)
        return self

    def detect_blobs(self, *args: Any, **kwargs: Any) -> "ShapeDetectionMixin":
        from natural_pdf.services import _shape_detection_impl as _impl

        _impl.detect_blobs(self, *args, **kwargs)
        return self

    def _detect_line_element_data(self, *args: Any, **kwargs: Any) -> Any:
        from natural_pdf.services import _shape_detection_impl as _impl

        return _impl.detect_line_element_data(self, *args, **kwargs)

    def _detect_lines_projection(self, *args: Any, **kwargs: Any) -> "ShapeDetectionMixin":
        from natural_pdf.services import _shape_detection_impl as _impl

        _impl.detect_lines_projection(self, *args, **kwargs)
        return self

    def _detect_lines_lsd(self, *args: Any, **kwargs: Any) -> "ShapeDetectionMixin":
        from natural_pdf.services import _shape_detection_impl as _impl

        _impl.detect_lines_lsd(self, *args, **kwargs)
        return self

    def _get_image_for_detection(self, resolution: int) -> Any:
        from natural_pdf.services import _shape_detection_impl as _impl

        return _impl.get_image_for_detection(self, resolution)

    def _convert_line_to_element_data(self, *args: Any, **kwargs: Any) -> Any:
        from natural_pdf.services import _shape_detection_impl as _impl

        return _impl.convert_line_to_element_data(*args, **kwargs)

    def _find_lines_on_image_data(self, *args: Any, **kwargs: Any) -> Any:
        from natural_pdf.services import _shape_detection_impl as _impl

        return _impl.find_lines_on_image_data(*args, **kwargs)

    def _process_image_for_lines_lsd(self, *args: Any, **kwargs: Any) -> Any:
        from natural_pdf.services import _shape_detection_impl as _impl

        return _impl.process_image_for_lines_lsd(*args, **kwargs)
