from __future__ import annotations

from typing import Any, Iterable, List, Optional, Sequence, Tuple

from natural_pdf.utils.spatial import is_element_in_region


class RegionGeometryMixin:
    """Common geometric helpers shared by region-like objects."""

    def is_point_inside(self, x: float, y: float) -> bool:
        if not self.has_polygon:
            return self.x0 <= x <= self.x1 and self.top <= y <= self.bottom
        return self._is_point_in_polygon(x, y)

    def _is_point_in_polygon(self, x: float, y: float) -> bool:
        polygon = self.polygon
        inside = False
        j = len(polygon) - 1
        for i in range(len(polygon)):
            xi, yi = polygon[i]
            xj, yj = polygon[j]
            if ((yi > y) != (yj > y)) and (
                x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi  # avoid division by zero
            ):
                inside = not inside
            j = i
        return inside

    def is_element_center_inside(self, element) -> bool:
        cx = (element.x0 + element.x1) / 2.0
        cy = (element.top + element.bottom) / 2.0
        return self.is_point_inside(cx, cy)

    def _is_element_in_region(self, element, use_boundary_tolerance: bool = True) -> bool:
        from natural_pdf.elements.base import Element

        if not isinstance(element, Element):
            return False
        return is_element_in_region(element, self, strategy="center", check_page=True)

    def contains(self, element) -> bool:
        if not hasattr(element, "page") or element.page != self.page:
            return False
        if not all(hasattr(element, attr) for attr in ["x0", "x1", "top", "bottom"]):
            return False
        if not self.has_polygon:
            return (
                self.x0 <= element.x0
                and element.x1 <= self.x1
                and self.top <= element.top
                and element.bottom <= self.bottom
            )

        corners = [
            (element.x0, element.top),
            (element.x1, element.top),
            (element.x1, element.bottom),
            (element.x0, element.bottom),
        ]
        return all(self.is_point_inside(x, y) for x, y in corners)

    def intersects(self, element) -> bool:
        if not hasattr(element, "page") or element.page != self.page:
            return False
        if not all(hasattr(element, attr) for attr in ["x0", "x1", "top", "bottom"]):
            return False

        if not self.has_polygon:
            return (
                self.x0 < element.x1
                and self.x1 > element.x0
                and self.top < element.bottom
                and self.bottom > element.top
            )

        corners = [
            (element.x0, element.top),
            (element.x1, element.top),
            (element.x1, element.bottom),
            (element.x0, element.bottom),
        ]
        if any(self.is_point_inside(x, y) for x, y in corners):
            return True
        for x, y in self.polygon:
            if element.x0 <= x <= element.x1 and element.top <= y <= element.bottom:
                return True
        return (
            self.x0 < element.x1
            and self.x1 > element.x0
            and self.top < element.bottom
            and self.bottom > element.top
        )
