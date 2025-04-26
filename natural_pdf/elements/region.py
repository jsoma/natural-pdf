import logging
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

from pdfplumber.utils.geometry import get_bbox_overlap, merge_bboxes, objects_to_bbox

# New Imports
from pdfplumber.utils.text import TEXTMAP_KWARGS, WORD_EXTRACTOR_KWARGS, chars_to_textmap

from natural_pdf.elements.base import DirectionalMixin

# Import new utils
from natural_pdf.utils.text_extraction import filter_chars_spatially, generate_text_layout

from natural_pdf.ocr.utils import _apply_ocr_correction_to_elements  # Import utility

# --- Classification Imports --- #
from natural_pdf.classification.mixin import ClassificationMixin
from natural_pdf.classification.manager import ClassificationManager # Keep for type hint
# --- End Classification Imports --- #

if TYPE_CHECKING:
    from natural_pdf.core.page import Page
    from natural_pdf.elements.text import TextElement

# Import OCRManager conditionally to avoid circular imports
try:
    from natural_pdf.ocr import OCRManager
except ImportError:
    # OCRManager will be imported directly in methods that use it
    pass

logger = logging.getLogger(__name__)


class Region(DirectionalMixin, ClassificationMixin):
    """
    Represents a rectangular region on a page.
    """

    def __init__(
        self,
        page: "Page",
        bbox: Tuple[float, float, float, float],
        polygon: List[Tuple[float, float]] = None,
        parent=None,
    ):
        """
        Initialize a region.

        Args:
            page: Parent page
            bbox: Bounding box as (x0, top, x1, bottom)
            polygon: Optional list of coordinate points [(x1,y1), (x2,y2), ...] for non-rectangular regions
            parent: Optional parent region (for hierarchical document structure)
        """
        self._page = page
        self._bbox = bbox
        self._polygon = polygon
        self._multi_page_elements = None
        self._spans_pages = False
        self._page_range = None
        self.start_element = None
        self.end_element = None

        # --- ADDED --- Metadata store for mixins
        self.metadata: Dict[str, Any] = {}
        # --- END ADDED ---

        # Standard attributes for all elements
        self.object_type = "region"  # For selector compatibility

        # Layout detection attributes
        self.region_type = None
        self.normalized_type = None
        self.confidence = None
        self.model = None

        # Region management attributes
        self.name = None
        self.source = None  # Will be set by creation methods

        # Hierarchy support for nested document structure
        self.parent_region = parent
        self.child_regions = []
        self.text_content = None  # Direct text content (e.g., from Docling)
        self.associated_text_elements = []  # Native text elements that overlap with this region

    def _direction(
        self,
        direction: str,
        size: Optional[float] = None,
        cross_size: str = "full",
        include_element: bool = False,
        until: Optional[str] = None,
        include_endpoint: bool = True,
        **kwargs,
    ) -> "Region":
        """
        Protected helper method to create a region in a specified direction relative to this region.

        Args:
            direction: 'left', 'right', 'above', or 'below'
            size: Size in the primary direction (width for horizontal, height for vertical)
            cross_size: Size in the cross direction ('full' or 'element')
            include_element: Whether to include this region's area in the result
            until: Optional selector string to specify a boundary element
            include_endpoint: Whether to include the boundary element found by 'until'
            **kwargs: Additional parameters for the 'until' selector search

        Returns:
            Region object
        """
        import math  # Use math.inf for infinity

        is_horizontal = direction in ("left", "right")
        is_positive = direction in ("right", "below")  # right/below are positive directions
        pixel_offset = 1  # Offset for excluding elements/endpoints

        # 1. Determine initial boundaries based on direction and include_element
        if is_horizontal:
            # Initial cross-boundaries (vertical)
            y0 = 0 if cross_size == "full" else self.top
            y1 = self.page.height if cross_size == "full" else self.bottom

            # Initial primary boundaries (horizontal)
            if is_positive:  # right
                x0_initial = self.x0 if include_element else self.x1 + pixel_offset
                x1_initial = self.x1  # This edge moves
            else:  # left
                x0_initial = self.x0  # This edge moves
                x1_initial = self.x1 if include_element else self.x0 - pixel_offset
        else:  # Vertical
            # Initial cross-boundaries (horizontal)
            x0 = 0 if cross_size == "full" else self.x0
            x1 = self.page.width if cross_size == "full" else self.x1

            # Initial primary boundaries (vertical)
            if is_positive:  # below
                y0_initial = self.top if include_element else self.bottom + pixel_offset
                y1_initial = self.bottom  # This edge moves
            else:  # above
                y0_initial = self.top  # This edge moves
                y1_initial = self.bottom if include_element else self.top - pixel_offset

        # 2. Calculate the final primary boundary, considering 'size' or page limits
        if is_horizontal:
            if is_positive:  # right
                x1_final = min(
                    self.page.width,
                    x1_initial + (size if size is not None else (self.page.width - x1_initial)),
                )
                x0_final = x0_initial
            else:  # left
                x0_final = max(0, x0_initial - (size if size is not None else x0_initial))
                x1_final = x1_initial
        else:  # Vertical
            if is_positive:  # below
                y1_final = min(
                    self.page.height,
                    y1_initial + (size if size is not None else (self.page.height - y1_initial)),
                )
                y0_final = y0_initial
            else:  # above
                y0_final = max(0, y0_initial - (size if size is not None else y0_initial))
                y1_final = y1_initial

        # 3. Handle 'until' selector if provided
        target = None
        if until:
            all_matches = self.page.find_all(until, **kwargs)
            matches_in_direction = []

            # Filter and sort matches based on direction
            if direction == "above":
                matches_in_direction = [m for m in all_matches if m.bottom <= self.top]
                matches_in_direction.sort(key=lambda e: e.bottom, reverse=True)
            elif direction == "below":
                matches_in_direction = [m for m in all_matches if m.top >= self.bottom]
                matches_in_direction.sort(key=lambda e: e.top)
            elif direction == "left":
                matches_in_direction = [m for m in all_matches if m.x1 <= self.x0]
                matches_in_direction.sort(key=lambda e: e.x1, reverse=True)
            elif direction == "right":
                matches_in_direction = [m for m in all_matches if m.x0 >= self.x1]
                matches_in_direction.sort(key=lambda e: e.x0)

            if matches_in_direction:
                target = matches_in_direction[0]

                # Adjust the primary boundary based on the target
                if is_horizontal:
                    if is_positive:  # right
                        x1_final = target.x1 if include_endpoint else target.x0 - pixel_offset
                    else:  # left
                        x0_final = target.x0 if include_endpoint else target.x1 + pixel_offset
                else:  # Vertical
                    if is_positive:  # below
                        y1_final = target.bottom if include_endpoint else target.top - pixel_offset
                    else:  # above
                        y0_final = target.top if include_endpoint else target.bottom + pixel_offset

                # Adjust cross boundaries if cross_size is 'element'
                if cross_size == "element":
                    if is_horizontal:  # Adjust y0, y1
                        target_y0 = (
                            target.top if include_endpoint else target.bottom
                        )  # Use opposite boundary if excluding
                        target_y1 = target.bottom if include_endpoint else target.top
                        y0 = min(y0, target_y0)
                        y1 = max(y1, target_y1)
                    else:  # Adjust x0, x1
                        target_x0 = (
                            target.x0 if include_endpoint else target.x1
                        )  # Use opposite boundary if excluding
                        target_x1 = target.x1 if include_endpoint else target.x0
                        x0 = min(x0, target_x0)
                        x1 = max(x1, target_x1)

        # 4. Finalize bbox coordinates
        if is_horizontal:
            bbox = (x0_final, y0, x1_final, y1)
        else:
            bbox = (x0, y0_final, x1, y1_final)

        # Ensure valid coordinates (x0 <= x1, y0 <= y1)
        final_x0 = min(bbox[0], bbox[2])
        final_y0 = min(bbox[1], bbox[3])
        final_x1 = max(bbox[0], bbox[2])
        final_y1 = max(bbox[1], bbox[3])
        final_bbox = (final_x0, final_y0, final_x1, final_y1)

        # 5. Create and return Region
        region = Region(self.page, final_bbox)
        region.source_element = self
        region.includes_source = include_element
        # Optionally store the boundary element if found
        if target:
            region.boundary_element = target

        return region

    def above(
        self,
        height: Optional[float] = None,
        width: str = "full",
        include_element: bool = False,
        until: Optional[str] = None,
        include_endpoint: bool = True,
        **kwargs,
    ) -> "Region":
        """
        Select region above this region.

        Args:
            height: Height of the region above, in points
            width: Width mode - "full" for full page width or "element" for element width
            include_element: Whether to include this region in the result (default: False)
            until: Optional selector string to specify an upper boundary element
            include_endpoint: Whether to include the boundary element in the region (default: True)
            **kwargs: Additional parameters

        Returns:
            Region object representing the area above
        """
        return self._direction(
            direction="above",
            size=height,
            cross_size=width,
            include_element=include_element,
            until=until,
            include_endpoint=include_endpoint,
            **kwargs,
        )

    def below(
        self,
        height: Optional[float] = None,
        width: str = "full",
        include_element: bool = False,
        until: Optional[str] = None,
        include_endpoint: bool = True,
        **kwargs,
    ) -> "Region":
        """
        Select region below this region.

        Args:
            height: Height of the region below, in points
            width: Width mode - "full" for full page width or "element" for element width
            include_element: Whether to include this region in the result (default: False)
            until: Optional selector string to specify a lower boundary element
            include_endpoint: Whether to include the boundary element in the region (default: True)
            **kwargs: Additional parameters

        Returns:
            Region object representing the area below
        """
        return self._direction(
            direction="below",
            size=height,
            cross_size=width,
            include_element=include_element,
            until=until,
            include_endpoint=include_endpoint,
            **kwargs,
        )

    def left(
        self,
        width: Optional[float] = None,
        height: str = "full",
        include_element: bool = False,
        until: Optional[str] = None,
        include_endpoint: bool = True,
        **kwargs,
    ) -> "Region":
        """
        Select region to the left of this region.

        Args:
            width: Width of the region to the left, in points
            height: Height mode - "full" for full page height or "element" for element height
            include_element: Whether to include this region in the result (default: False)
            until: Optional selector string to specify a left boundary element
            include_endpoint: Whether to include the boundary element in the region (default: True)
            **kwargs: Additional parameters

        Returns:
            Region object representing the area to the left
        """
        return self._direction(
            direction="left",
            size=width,
            cross_size=height,
            include_element=include_element,
            until=until,
            include_endpoint=include_endpoint,
            **kwargs,
        )

    def right(
        self,
        width: Optional[float] = None,
        height: str = "full",
        include_element: bool = False,
        until: Optional[str] = None,
        include_endpoint: bool = True,
        **kwargs,
    ) -> "Region":
        """
        Select region to the right of this region.

        Args:
            width: Width of the region to the right, in points
            height: Height mode - "full" for full page height or "element" for element height
            include_element: Whether to include this region in the result (default: False)
            until: Optional selector string to specify a right boundary element
            include_endpoint: Whether to include the boundary element in the region (default: True)
            **kwargs: Additional parameters

        Returns:
            Region object representing the area to the right
        """
        return self._direction(
            direction="right",
            size=width,
            cross_size=height,
            include_element=include_element,
            until=until,
            include_endpoint=include_endpoint,
            **kwargs,
        )

    @property
    def type(self) -> str:
        """Element type."""
        # Return the specific type if detected (e.g., from layout analysis)
        # or 'region' as a default.
        return self.region_type or "region"  # Prioritize specific region_type if set

    @property
    def page(self) -> "Page":
        """Get the parent page."""
        return self._page

    @property
    def bbox(self) -> Tuple[float, float, float, float]:
        """Get the bounding box as (x0, top, x1, bottom)."""
        return self._bbox

    @property
    def x0(self) -> float:
        """Get the left coordinate."""
        return self._bbox[0]

    @property
    def top(self) -> float:
        """Get the top coordinate."""
        return self._bbox[1]

    @property
    def x1(self) -> float:
        """Get the right coordinate."""
        return self._bbox[2]

    @property
    def bottom(self) -> float:
        """Get the bottom coordinate."""
        return self._bbox[3]

    @property
    def width(self) -> float:
        """Get the width of the region."""
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        """Get the height of the region."""
        return self.bottom - self.top

    @property
    def has_polygon(self) -> bool:
        """Check if this region has polygon coordinates."""
        return self._polygon is not None and len(self._polygon) >= 3

    @property
    def polygon(self) -> List[Tuple[float, float]]:
        """Get polygon coordinates if available, otherwise return rectangle corners."""
        if self._polygon:
            return self._polygon
        else:
            # Create rectangle corners from bbox as fallback
            return [
                (self.x0, self.top),  # top-left
                (self.x1, self.top),  # top-right
                (self.x1, self.bottom),  # bottom-right
                (self.x0, self.bottom),  # bottom-left
            ]

    def _is_point_in_polygon(self, x: float, y: float) -> bool:
        """
        Check if a point is inside the polygon using ray casting algorithm.

        Args:
            x: X coordinate of the point
            y: Y coordinate of the point

        Returns:
            bool: True if the point is inside the polygon
        """
        if not self.has_polygon:
            return (self.x0 <= x <= self.x1) and (self.top <= y <= self.bottom)

        # Ray casting algorithm
        inside = False
        j = len(self.polygon) - 1

        for i in range(len(self.polygon)):
            if ((self.polygon[i][1] > y) != (self.polygon[j][1] > y)) and (
                x
                < (self.polygon[j][0] - self.polygon[i][0])
                * (y - self.polygon[i][1])
                / (self.polygon[j][1] - self.polygon[i][1])
                + self.polygon[i][0]
            ):
                inside = not inside
            j = i

        return inside

    def is_point_inside(self, x: float, y: float) -> bool:
        """
        Check if a point is inside this region using ray casting algorithm for polygons.

        Args:
            x: X coordinate of the point
            y: Y coordinate of the point

        Returns:
            bool: True if the point is inside the region
        """
        if not self.has_polygon:
            return (self.x0 <= x <= self.x1) and (self.top <= y <= self.bottom)

        # Ray casting algorithm
        inside = False
        j = len(self.polygon) - 1

        for i in range(len(self.polygon)):
            if ((self.polygon[i][1] > y) != (self.polygon[j][1] > y)) and (
                x
                < (self.polygon[j][0] - self.polygon[i][0])
                * (y - self.polygon[i][1])
                / (self.polygon[j][1] - self.polygon[i][1])
                + self.polygon[i][0]
            ):
                inside = not inside
            j = i

        return inside

    def _is_element_in_region(self, element: "Element", use_boundary_tolerance=True) -> bool:
        """
        Check if an element is within this region.

        Args:
            element: Element to check
            use_boundary_tolerance: Whether to apply a small tolerance for boundary elements

        Returns:
            True if the element is in the region, False otherwise
        """
        # If we have multi-page elements cached, check if the element is in the list
        if self._spans_pages and self._multi_page_elements is not None:
            return element in self._multi_page_elements

        # Check if element is on the same page
        if not hasattr(element, "page") or element.page != self._page:
            return False

        # Calculate element center
        # Ensure element has necessary attributes
        if not all(hasattr(element, attr) for attr in ["x0", "x1", "top", "bottom"]):
            return False  # Cannot determine position

        element_center_x = (element.x0 + element.x1) / 2
        element_center_y = (element.top + element.bottom) / 2

        # Check if center point is inside the region's geometry
        return self.is_point_inside(element_center_x, element_center_y)

    def highlight(
        self,
        label: Optional[str] = None,
        color: Optional[Union[Tuple, str]] = None,
        use_color_cycling: bool = False,
        include_attrs: Optional[List[str]] = None,
        existing: str = "append",
    ) -> "Region":
        """
        Highlight this region on the page.

        Args:
            label: Optional label for the highlight
            color: Color tuple/string for the highlight, or None to use automatic color
            use_color_cycling: Force color cycling even with no label (default: False)
            include_attrs: List of attribute names to display on the highlight (e.g., ['confidence', 'type'])
            existing: How to handle existing highlights ('append' or 'replace').

        Returns:
            Self for method chaining
        """
        # Access the highlighter service correctly
        highlighter = self.page._highlighter

        # Prepare common arguments
        highlight_args = {
            "page_index": self.page.index,
            "color": color,
            "label": label,
            "use_color_cycling": use_color_cycling,
            "element": self,  # Pass the region itself so attributes can be accessed
            "include_attrs": include_attrs,
            "existing": existing,
        }

        # Call the appropriate service method
        if self.has_polygon:
            highlight_args["polygon"] = self.polygon
            highlighter.add_polygon(**highlight_args)
        else:
            highlight_args["bbox"] = self.bbox
            highlighter.add(**highlight_args)

        return self

    def to_image(
        self,
        scale: float = 2.0,
        resolution: float = 150,
        crop_only: bool = False,
        include_highlights: bool = True,
        **kwargs,
    ) -> "Image.Image":
        """
        Generate an image of just this region.

        Args:
            resolution: Resolution in DPI for rendering (default: 150)
            crop_only: If True, only crop the region without highlighting its boundaries
            include_highlights: Whether to include existing highlights (default: True)
            **kwargs: Additional parameters for page.to_image()

        Returns:
            PIL Image of just this region
        """
        # First get the full page image with highlights if requested
        page_image = self._page.to_image(
            scale=scale, resolution=resolution, include_highlights=include_highlights, **kwargs
        )

        # Calculate the crop coordinates - apply resolution scaling factor
        # PDF coordinates are in points (1/72 inch), but image is scaled by resolution
        scale_factor = resolution / 72.0  # Scale based on DPI

        # Apply scaling to the coordinates
        x0 = int(self.x0 * scale_factor)
        top = int(self.top * scale_factor)
        x1 = int(self.x1 * scale_factor)
        bottom = int(self.bottom * scale_factor)

        # Ensure coords are valid for cropping (left < right, top < bottom)
        if x0 >= x1:
             logger.warning(
                 f"Region {self.bbox} resulted in non-positive width after scaling ({x0} >= {x1}). Cannot create image."
             )
             return None
        if top >= bottom:
             logger.warning(
                 f"Region {self.bbox} resulted in non-positive height after scaling ({top} >= {bottom}). Cannot create image."
             )
             return None

        # Crop the image to just this region
        region_image = page_image.crop((x0, top, x1, bottom))

        # If not crop_only, add a border to highlight the region boundaries
        if not crop_only:
            from PIL import ImageDraw

            # Create a 1px border around the region
            draw = ImageDraw.Draw(region_image)
            draw.rectangle(
                (0, 0, region_image.width - 1, region_image.height - 1),
                outline=(255, 0, 0),
                width=1,
            )

        return region_image

    def show(
        self,
        scale: float = 2.0,
        labels: bool = True,
        legend_position: str = "right",
        # Add a default color for standalone show
        color: Optional[Union[Tuple, str]] = "blue",
        label: Optional[str] = None,
    ) -> "Image.Image":
        """
        Show the page with just this region highlighted temporarily.

        Args:
            scale: Scale factor for rendering
            labels: Whether to include a legend for labels
            legend_position: Position of the legend
            color: Color to highlight this region (default: blue)
            label: Optional label for this region in the legend

        Returns:
            PIL Image of the page with only this region highlighted
        """
        if not self._page:
            raise ValueError("Region must be associated with a page to show.")

        # Use the highlighting service via the page's property
        service = self._page._highlighter

        # Determine the label if not provided
        display_label = (
            label if label is not None else f"Region ({self.type})" if self.type else "Region"
        )

        # Prepare temporary highlight data for just this region
        temp_highlight_data = {
            "page_index": self._page.index,
            "bbox": self.bbox,
            "polygon": self.polygon if self.has_polygon else None,
            "color": color,  # Use provided or default color
            "label": display_label,
            "use_color_cycling": False,  # Explicitly false for single preview
        }

        # Use render_preview to show only this highlight
        return service.render_preview(
            page_index=self._page.index,
            temporary_highlights=[temp_highlight_data],
            scale=scale,
            labels=labels,
            legend_position=legend_position,
        )

    def save(
        self, filename: str, scale: float = 2.0, labels: bool = True, legend_position: str = "right"
    ) -> "Region":
        """
        Save the page with this region highlighted to an image file.

        Args:
            filename: Path to save the image to
            scale: Scale factor for rendering
            labels: Whether to include a legend for labels
            legend_position: Position of the legend

        Returns:
            Self for method chaining
        """
        # Highlight this region if not already highlighted
        self.highlight()

        # Save the highlighted image
        self._page.save_image(filename, scale=scale, labels=labels, legend_position=legend_position)
        return self

    def save_image(
        self,
        filename: str,
        resolution: float = 150,
        crop_only: bool = False,
        include_highlights: bool = True,
        **kwargs,
    ) -> "Region":
        """
        Save an image of just this region to a file.

        Args:
            filename: Path to save the image to
            resolution: Resolution in DPI for rendering (default: 150)
            crop_only: If True, only crop the region without highlighting its boundaries
            include_highlights: Whether to include existing highlights (default: True)
            **kwargs: Additional parameters for page.to_image()

        Returns:
            Self for method chaining
        """
        # Get the region image
        image = self.to_image(
            resolution=resolution,
            crop_only=crop_only,
            include_highlights=include_highlights,
            **kwargs,
        )

        # Save the image
        image.save(filename)
        return self

    def get_elements(
        self, selector: Optional[str] = None, apply_exclusions=True, **kwargs
    ) -> List["Element"]:
        """
        Get all elements within this region.

        Args:
            selector: Optional selector to filter elements
            apply_exclusions: Whether to apply exclusion regions
            **kwargs: Additional parameters for element filtering

        Returns:
            List of elements in the region
        """
        # If we have multi-page elements, return those
        if self._spans_pages and self._multi_page_elements is not None:
            # TODO: Apply selector to multi-page elements if needed
            return self._multi_page_elements

        # Otherwise, get elements from the page
        if selector:
            # Find elements on the page matching the selector
            page_elements = self.page.find_all(
                selector, apply_exclusions=apply_exclusions, **kwargs
            )
            # Filter those elements to only include ones within this region
            return [e for e in page_elements if self._is_element_in_region(e)]
        else:
            # Get all elements from the page
            page_elements = self.page.get_elements(apply_exclusions=apply_exclusions)
            # Filter to elements in this region
            return [e for e in page_elements if self._is_element_in_region(e)]

    def extract_text(self, apply_exclusions=True, debug=False, **kwargs) -> str:
        """
        Extract text from this region, respecting page exclusions and using pdfplumber's
        layout engine (chars_to_textmap).

        Args:
            apply_exclusions: Whether to apply exclusion regions defined on the parent page.
            debug: Enable verbose debugging output for filtering steps.
            **kwargs: Additional layout parameters passed directly to pdfplumber's
                      `chars_to_textmap` function (e.g., layout, x_density, y_density).
                      See Page.extract_text docstring for more.

        Returns:
            Extracted text as string, potentially with layout-based spacing.
        """
        # Allow 'debug_exclusions' for backward compatibility
        debug = kwargs.get("debug", debug or kwargs.get("debug_exclusions", False))
        logger.debug(f"Region {self.bbox}: extract_text called with kwargs: {kwargs}")

        # --- Handle Docling source (priority) --- DEPRECATED or Adapt?
        # For now, let's bypass this and always use the standard extraction flow
        # based on contained elements to ensure consistency.
        # if self.model == 'docling' or hasattr(self, 'text_content'): ...

        # 1. Get Word Elements potentially within this region (initial broad phase)
        # Optimization: Could use spatial query if page elements were indexed
        page_words = self.page.words  # Get all words from the page

        # 2. Gather all character dicts from words potentially in region
        # We filter precisely in filter_chars_spatially
        all_char_dicts = []
        for word in page_words:
            # Quick bbox check to avoid processing words clearly outside
            if get_bbox_overlap(self.bbox, word.bbox) is not None:
                all_char_dicts.extend(getattr(word, "_char_dicts", []))

        if not all_char_dicts:
            logger.debug(f"Region {self.bbox}: No character dicts found overlapping region bbox.")
            return ""

        # 3. Get Relevant Exclusions (overlapping this region)
        apply_exclusions_flag = kwargs.get("apply_exclusions", apply_exclusions)
        exclusion_regions = []
        if apply_exclusions_flag and self._page._exclusions:
            all_page_exclusions = self._page._get_exclusion_regions(
                include_callable=True, debug=debug
            )
            overlapping_exclusions = []
            for excl in all_page_exclusions:
                if get_bbox_overlap(self.bbox, excl.bbox) is not None:
                    overlapping_exclusions.append(excl)
            exclusion_regions = overlapping_exclusions
            if debug:
                logger.debug(
                    f"Region {self.bbox}: Applying {len(exclusion_regions)} overlapping exclusions."
                )
        elif debug:
            logger.debug(f"Region {self.bbox}: Not applying exclusions.")

        # 4. Spatially Filter Characters using Utility
        # Pass self as the target_region for precise polygon checks etc.
        filtered_chars = filter_chars_spatially(
            char_dicts=all_char_dicts,
            exclusion_regions=exclusion_regions,
            target_region=self,  # Pass self!
            debug=debug,
        )

        # 5. Generate Text Layout using Utility
        result = generate_text_layout(
            char_dicts=filtered_chars,
            layout_context_bbox=self.bbox,  # Use region's bbox for context
            user_kwargs=kwargs,
        )

        logger.debug(f"Region {self.bbox}: extract_text finished, result length: {len(result)}.")
        return result

    def extract_table(
        self,
        method: str = None,
        table_settings: dict = None,
        use_ocr: bool = False,
        ocr_config: dict = None,
    ) -> List[List[str]]:
        """
        Extract a table from this region.

        Args:
            method: Method to use for extraction ('tatr', 'plumber', or None for auto-detection)
            table_settings: Settings for pdfplumber table extraction (used only with 'plumber' method)
            use_ocr: Whether to use OCR for text extraction (only applicable with 'tatr' method)
            ocr_config: OCR configuration parameters

        Returns:
            Table data as a list of rows, where each row is a list of cell values
        """
        # Default settings if none provided
        if table_settings is None:
            table_settings = {}

        # Auto-detect method if not specified
        if method is None:
            # If this is a TATR-detected region, use TATR method
            if hasattr(self, "model") and self.model == "tatr" and self.region_type == "table":
                method = "tatr"
            else:
                method = "plumber"

        # Use the selected method
        if method == "tatr":
            return self._extract_table_tatr(use_ocr=use_ocr, ocr_config=ocr_config)
        else:  # Default to pdfplumber
            return self._extract_table_plumber(table_settings)

    def _extract_table_plumber(self, table_settings: dict) -> List[List[str]]:
        """
        Extract table using pdfplumber's table extraction.

        Args:
            table_settings: Settings for pdfplumber table extraction

        Returns:
            Table data as a list of rows, where each row is a list of cell values
        """
        # Create a crop of the page for this region
        cropped = self.page._page.crop(self.bbox)

        # Extract table from the cropped area
        tables = cropped.extract_tables(table_settings)

        # Return the first table or an empty list if none found
        if tables:
            return tables[0]
        return []

    def _extract_table_tatr(self, use_ocr=False, ocr_config=None) -> List[List[str]]:
        """
        Extract table using TATR structure detection.

        Args:
            use_ocr: Whether to apply OCR to each cell for better text extraction
            ocr_config: Optional OCR configuration parameters

        Returns:
            Table data as a list of rows, where each row is a list of cell values
        """
        # Find all rows and headers in this table
        rows = self.page.find_all(f"region[type=table-row][model=tatr]")
        headers = self.page.find_all(f"region[type=table-column-header][model=tatr]")
        columns = self.page.find_all(f"region[type=table-column][model=tatr]")

        # Filter to only include rows/headers/columns that overlap with this table region
        def is_in_table(region):
            # Check for overlap - simplifying to center point for now
            region_center_x = (region.x0 + region.x1) / 2
            region_center_y = (region.top + region.bottom) / 2
            return (
                self.x0 <= region_center_x <= self.x1 and self.top <= region_center_y <= self.bottom
            )

        rows = [row for row in rows if is_in_table(row)]
        headers = [header for header in headers if is_in_table(header)]
        columns = [column for column in columns if is_in_table(column)]

        # Sort rows by vertical position (top to bottom)
        rows.sort(key=lambda r: r.top)

        # Sort columns by horizontal position (left to right)
        columns.sort(key=lambda c: c.x0)

        # Create table data structure
        table_data = []

        # Prepare OCR config if needed
        if use_ocr:
            # Default OCR config focuses on small text with low confidence
            default_ocr_config = {
                "enabled": True,
                "min_confidence": 0.1,  # Lower than default to catch more text
                "detection_params": {
                    "text_threshold": 0.1,  # Lower threshold for low-contrast text
                    "link_threshold": 0.1,  # Lower threshold for connecting text components
                },
            }

            # Merge with provided config if any
            if ocr_config:
                if isinstance(ocr_config, dict):
                    # Update default config with provided values
                    for key, value in ocr_config.items():
                        if (
                            isinstance(value, dict)
                            and key in default_ocr_config
                            and isinstance(default_ocr_config[key], dict)
                        ):
                            # Merge nested dicts
                            default_ocr_config[key].update(value)
                        else:
                            # Replace value
                            default_ocr_config[key] = value
                else:
                    # Not a dict, use as is
                    default_ocr_config = ocr_config

            # Use the merged config
            ocr_config = default_ocr_config

        # Add header row if headers were detected
        if headers:
            header_texts = []
            for header in headers:
                if use_ocr:
                    # Try OCR for better text extraction
                    ocr_elements = header.apply_ocr(**ocr_config)
                    if ocr_elements:
                        ocr_text = " ".join(e.text for e in ocr_elements).strip()
                        if ocr_text:
                            header_texts.append(ocr_text)
                            continue

                # Fallback to normal extraction
                header_texts.append(header.extract_text().strip())
            table_data.append(header_texts)

        # Process rows
        for row in rows:
            row_cells = []

            # If we have columns, use them to extract cells
            if columns:
                for column in columns:
                    # Create a cell region at the intersection of row and column
                    cell_bbox = (column.x0, row.top, column.x1, row.bottom)

                    # Create a region for this cell
                    from natural_pdf.elements.region import (  # Import here to avoid circular imports
                        Region,
                    )

                    cell_region = Region(self.page, cell_bbox)

                    # Extract text from the cell
                    if use_ocr:
                        # Apply OCR to the cell
                        ocr_elements = cell_region.apply_ocr(**ocr_config)
                        if ocr_elements:
                            # Get text from OCR elements
                            ocr_text = " ".join(e.text for e in ocr_elements).strip()
                            if ocr_text:
                                row_cells.append(ocr_text)
                                continue

                    # Fallback to normal extraction
                    cell_text = cell_region.extract_text().strip()
                    row_cells.append(cell_text)
            else:
                # No column information, just extract the whole row text
                if use_ocr:
                    # Try OCR on the whole row
                    ocr_elements = row.apply_ocr(**ocr_config)
                    if ocr_elements:
                        ocr_text = " ".join(e.text for e in ocr_elements).strip()
                        if ocr_text:
                            row_cells.append(ocr_text)
                            continue

                # Fallback to normal extraction
                row_cells.append(row.extract_text().strip())

            table_data.append(row_cells)

        return table_data

    def find(self, selector: str, apply_exclusions=True, **kwargs) -> Optional["Element"]:
        """
        Find the first element in this region matching the selector.

        Args:
            selector: CSS-like selector string
            apply_exclusions: Whether to apply exclusion regions
            **kwargs: Additional parameters for element filtering

        Returns:
            First matching element or None
        """
        elements = self.find_all(selector, apply_exclusions=apply_exclusions, **kwargs)
        return elements.first if elements else None  # Use .first property

    def find_all(
        self, selector: str, apply_exclusions=True, **kwargs
    ) -> "ElementCollection":  # Changed from _find_all
        """
        Find all elements in this region matching the selector.

        Args:
            selector: CSS-like selector string
            apply_exclusions: Whether to apply exclusion regions
            **kwargs: Additional parameters for element filtering

        Returns:
            ElementCollection with matching elements
        """
        from natural_pdf.elements.collections import ElementCollection

        # If we span multiple pages, filter our elements
        # TODO: Revisit multi-page region logic
        if self._spans_pages and self._multi_page_elements is not None:
            logger.warning("find_all on multi-page regions is not fully implemented.")
            # Temporary: Apply filter directly to cached elements
            from natural_pdf.selectors.parser import parse_selector, selector_to_filter_func

            try:
                selector_obj = parse_selector(selector)
                filter_func = selector_to_filter_func(selector_obj, **kwargs)
                matching = [el for el in self._multi_page_elements if filter_func(el)]
                return ElementCollection(matching)
            except Exception as e:
                logger.error(f"Error applying selector to multi-page region elements: {e}")
                return ElementCollection([])

        # Otherwise, get elements from the page and filter by selector and region
        page_elements = self.page.find_all(selector, apply_exclusions=apply_exclusions, **kwargs)
        # Use the precise _is_element_in_region check
        filtered_elements = [e for e in page_elements if self._is_element_in_region(e)]
        return ElementCollection(filtered_elements)

    def apply_ocr(self, replace=True, **ocr_params) -> "Region":
        """
        Apply OCR to this region and return the created text elements.

        Args:
            replace: If True (default), removes existing OCR elements in the region
                    before adding new ones. If False, adds new OCR elements without 
                    removing existing ones.
            **ocr_params: Keyword arguments passed to the OCR Manager.
                          Common parameters like `engine`, `languages`, `min_confidence`,
                          `device`, and `resolution` (for image rendering) should be
                          provided here. **The `languages` list must contain codes
                          understood by the specific engine selected.** No mapping
                          is performed. Engine-specific settings can be passed in
                          an `options` object (e.g., `options=EasyOCROptions(...)`).

        Returns:
            Self for method chaining.
        """
        # Ensure OCRManager is available
        if not hasattr(self.page._parent, "_ocr_manager") or self.page._parent._ocr_manager is None:
            logger.error("OCRManager not available on parent PDF. Cannot apply OCR to region.")
            return self

        # If replace is True, find and remove existing OCR elements in this region
        if replace:
            logger.info(f"Region {self.bbox}: Removing existing OCR elements before applying new OCR.")
            # Find all OCR elements in this region
            ocr_selector = "text[source=ocr]"
            ocr_elements = self.find_all(ocr_selector)
            
            if ocr_elements:
                logger.info(f"Region {self.bbox}: Found {len(ocr_elements)} existing OCR elements to remove.")
                # Remove these elements from their page
                removed_count = ocr_elements.remove()
                logger.info(f"Region {self.bbox}: Removed {removed_count} OCR elements.")
            else:
                logger.info(f"Region {self.bbox}: No existing OCR elements found to remove.")

        ocr_mgr = self.page._parent._ocr_manager

        # Determine rendering resolution from parameters
        final_resolution = ocr_params.get("resolution")
        if final_resolution is None and hasattr(self.page, "_parent") and self.page._parent:
            final_resolution = getattr(self.page._parent, "_config", {}).get("resolution", 150)
        elif final_resolution is None:
            final_resolution = 150
        logger.debug(
            f"Region {self.bbox}: Applying OCR with resolution {final_resolution} DPI and params: {ocr_params}"
        )

        # Render the page region to an image using the determined resolution
        try:
            region_image = self.to_image(
                resolution=final_resolution, include_highlights=False, crop_only=True
            )
            if not region_image:
                logger.error("Failed to render region to image for OCR.")
                return self
            logger.debug(f"Region rendered to image size: {region_image.size}")
        except Exception as e:
            logger.error(f"Error rendering region to image for OCR: {e}", exc_info=True)
            return self

        # Prepare args for the OCR Manager
        manager_args = {
            "images": region_image,
            "engine": ocr_params.get("engine"),
            "languages": ocr_params.get("languages"),
            "min_confidence": ocr_params.get("min_confidence"),
            "device": ocr_params.get("device"),
            "options": ocr_params.get("options"),
            "detect_only": ocr_params.get("detect_only"),
        }
        manager_args = {k: v for k, v in manager_args.items() if v is not None}

        # Run OCR on this region's image using the manager
        try:
            results = ocr_mgr.apply_ocr(**manager_args)
            if not isinstance(results, list):
                logger.error(
                    f"OCRManager returned unexpected type for single region image: {type(results)}"
                )
                return self
            logger.debug(f"Region OCR processing returned {len(results)} results.")
        except Exception as e:
            logger.error(f"Error during OCRManager processing for region: {e}", exc_info=True)
            return self

        # Convert results to TextElements
        scale_x = self.width / region_image.width if region_image.width > 0 else 1.0
        scale_y = self.height / region_image.height if region_image.height > 0 else 1.0
        logger.debug(f"Region OCR scaling factors (PDF/Img): x={scale_x:.2f}, y={scale_y:.2f}")
        created_elements = []
        for result in results:
            try:
                img_x0, img_top, img_x1, img_bottom = map(float, result["bbox"])
                pdf_height = (img_bottom - img_top) * scale_y
                page_x0 = self.x0 + (img_x0 * scale_x)
                page_top = self.top + (img_top * scale_y)
                page_x1 = self.x0 + (img_x1 * scale_x)
                page_bottom = self.top + (img_bottom * scale_y)
                element_data = {
                    "text": result["text"],
                    "x0": page_x0,
                    "top": page_top,
                    "x1": page_x1,
                    "bottom": page_bottom,
                    "width": page_x1 - page_x0,
                    "height": page_bottom - page_top,
                    "object_type": "word",
                    "source": "ocr",
                    "confidence": float(result.get("confidence", 0.0)),
                    "fontname": "OCR",
                    "size": round(pdf_height) if pdf_height > 0 else 10.0,
                    "page_number": self.page.number,
                    "bold": False,
                    "italic": False,
                    "upright": True,
                    "doctop": page_top + self.page._page.initial_doctop,
                }
                ocr_char_dict = element_data.copy()
                ocr_char_dict["object_type"] = "char"
                ocr_char_dict.setdefault("adv", ocr_char_dict.get("width", 0))
                element_data["_char_dicts"] = [ocr_char_dict]
                from natural_pdf.elements.text import TextElement

                elem = TextElement(element_data, self.page)
                created_elements.append(elem)
                self.page._element_mgr.add_element(elem, element_type="words")
                self.page._element_mgr.add_element(ocr_char_dict, element_type="chars")
            except Exception as e:
                logger.error(
                    f"Failed to convert region OCR result to element: {result}. Error: {e}",
                    exc_info=True,
                )
        logger.info(f"Region {self.bbox}: Added {len(created_elements)} elements from OCR.")
        return self

    def get_section_between(self, start_element=None, end_element=None, boundary_inclusion="both"):
        """
        Get a section between two elements within this region.

        Args:
            start_element: Element marking the start of the section
            end_element: Element marking the end of the section
            boundary_inclusion: How to include boundary elements: 'start', 'end', 'both', or 'none'

        Returns:
            Region representing the section
        """
        # Get elements only within this region first
        elements = self.get_elements()

        # If no elements, return self or empty region?
        if not elements:
            logger.warning(
                f"get_section_between called on region {self.bbox} with no contained elements."
            )
            # Return an empty region at the start of the parent region
            return Region(self.page, (self.x0, self.top, self.x0, self.top))

        # Sort elements in reading order
        elements.sort(key=lambda e: (e.top, e.x0))

        # Find start index
        start_idx = 0
        if start_element:
            try:
                start_idx = elements.index(start_element)
            except ValueError:
                # Start element not in region, use first element
                logger.debug("Start element not found in region, using first element.")
                start_element = elements[0]  # Use the actual first element
                start_idx = 0
        else:
            start_element = elements[0]  # Default start is first element

        # Find end index
        end_idx = len(elements) - 1
        if end_element:
            try:
                end_idx = elements.index(end_element)
            except ValueError:
                # End element not in region, use last element
                logger.debug("End element not found in region, using last element.")
                end_element = elements[-1]  # Use the actual last element
                end_idx = len(elements) - 1
        else:
            end_element = elements[-1]  # Default end is last element

        # Adjust indexes based on boundary inclusion
        start_element_for_bbox = start_element
        end_element_for_bbox = end_element

        if boundary_inclusion == "none":
            start_idx += 1
            end_idx -= 1
            start_element_for_bbox = elements[start_idx] if start_idx <= end_idx else None
            end_element_for_bbox = elements[end_idx] if start_idx <= end_idx else None
        elif boundary_inclusion == "start":
            end_idx -= 1
            end_element_for_bbox = elements[end_idx] if start_idx <= end_idx else None
        elif boundary_inclusion == "end":
            start_idx += 1
            start_element_for_bbox = elements[start_idx] if start_idx <= end_idx else None

        # Ensure valid indexes
        start_idx = max(0, start_idx)
        end_idx = min(len(elements) - 1, end_idx)

        # If no valid elements in range, return empty region
        if start_idx > end_idx or start_element_for_bbox is None or end_element_for_bbox is None:
            logger.debug("No valid elements in range for get_section_between.")
            # Return an empty region positioned at the start element boundary
            anchor = start_element if start_element else self
            return Region(self.page, (anchor.x0, anchor.top, anchor.x0, anchor.top))

        # Get elements in range based on adjusted indices
        section_elements = elements[start_idx : end_idx + 1]

        # Create bounding box around the ELEMENTS included based on indices
        x0 = min(e.x0 for e in section_elements)
        top = min(e.top for e in section_elements)
        x1 = max(e.x1 for e in section_elements)
        bottom = max(e.bottom for e in section_elements)

        # Create new region
        section = Region(self.page, (x0, top, x1, bottom))
        # Store the original boundary elements for reference
        section.start_element = start_element
        section.end_element = end_element

        return section

    def get_sections(
        self, start_elements=None, end_elements=None, boundary_inclusion="both"
    ) -> List["Region"]:
        """
        Get sections within this region based on start/end elements.

        Args:
            start_elements: Elements or selector string that mark the start of sections
            end_elements: Elements or selector string that mark the end of sections
            boundary_inclusion: How to include boundary elements: 'start', 'end', 'both', or 'none'

        Returns:
            List of Region objects representing the extracted sections
        """
        from natural_pdf.elements.collections import ElementCollection

        # Process string selectors to find elements WITHIN THIS REGION
        if isinstance(start_elements, str):
            start_elements = self.find_all(start_elements)  # Use region's find_all
            if hasattr(start_elements, "elements"):
                start_elements = start_elements.elements

        if isinstance(end_elements, str):
            end_elements = self.find_all(end_elements)  # Use region's find_all
            if hasattr(end_elements, "elements"):
                end_elements = end_elements.elements

        # Ensure start_elements is a list (or similar iterable)
        if start_elements is None or not hasattr(start_elements, "__iter__"):
            logger.warning(
                "get_sections requires valid start_elements (selector or list). Returning empty."
            )
            return []
        # Ensure end_elements is a list if provided
        if end_elements is not None and not hasattr(end_elements, "__iter__"):
            logger.warning("end_elements must be iterable if provided. Ignoring.")
            end_elements = []
        elif end_elements is None:
            end_elements = []

        # If no start elements found within the region, return empty list
        if not start_elements:
            return []

        # Sort all elements within the region in reading order
        all_elements_in_region = self.get_elements()
        all_elements_in_region.sort(key=lambda e: (e.top, e.x0))

        if not all_elements_in_region:
            return []  # Cannot create sections if region is empty

        # Map elements to their indices in the sorted list
        element_to_index = {el: i for i, el in enumerate(all_elements_in_region)}

        # Mark section boundaries using indices from the sorted list
        section_boundaries = []

        # Add start element indexes
        for element in start_elements:
            idx = element_to_index.get(element)
            if idx is not None:
                section_boundaries.append({"index": idx, "element": element, "type": "start"})
            # else: Element found by selector might not be geometrically in region? Log warning?

        # Add end element indexes if provided
        for element in end_elements:
            idx = element_to_index.get(element)
            if idx is not None:
                section_boundaries.append({"index": idx, "element": element, "type": "end"})

        # Sort boundaries by index (document order within the region)
        section_boundaries.sort(key=lambda x: x["index"])

        # Generate sections
        sections = []
        current_start_boundary = None

        for i, boundary in enumerate(section_boundaries):
            # If it's a start boundary and we don't have a current start
            if boundary["type"] == "start" and current_start_boundary is None:
                current_start_boundary = boundary

            # If it's an end boundary and we have a current start
            elif boundary["type"] == "end" and current_start_boundary is not None:
                # Create a section from current_start to this boundary
                start_element = current_start_boundary["element"]
                end_element = boundary["element"]
                # Use the helper, ensuring elements are from within the region
                section = self.get_section_between(start_element, end_element, boundary_inclusion)
                sections.append(section)
                current_start_boundary = None  # Reset

            # If it's another start boundary and we have a current start (split by starts only)
            elif (
                boundary["type"] == "start"
                and current_start_boundary is not None
                and not end_elements
            ):
                # End the previous section just before this start boundary
                start_element = current_start_boundary["element"]
                # Find the element immediately preceding this start in the sorted list
                end_idx = boundary["index"] - 1
                if end_idx >= 0 and end_idx >= current_start_boundary["index"]:
                    end_element = all_elements_in_region[end_idx]
                    section = self.get_section_between(
                        start_element, end_element, boundary_inclusion
                    )
                    sections.append(section)
                # Else: Section started and ended by consecutive start elements? Create empty?
                # For now, just reset and start new section

                # Start the new section
                current_start_boundary = boundary

        # Handle the last section if we have a current start
        if current_start_boundary is not None:
            start_element = current_start_boundary["element"]
            # End at the last element within the region
            end_element = all_elements_in_region[-1]
            section = self.get_section_between(start_element, end_element, boundary_inclusion)
            sections.append(section)

        return sections

    def create_cells(self):
        """
        Create cell regions for a detected table by intersecting its
        row and column regions, and add them to the page.

        Assumes child row and column regions are already present on the page.

        Returns:
            Self for method chaining.
        """
        # Ensure this is called on a table region
        if self.region_type not in (
            "table",
            "tableofcontents",
        ):  # Allow for ToC which might have structure
            raise ValueError(
                f"create_cells should be called on a 'table' or 'tableofcontents' region, not '{self.region_type}'"
            )

        # Find rows and columns associated with this page
        # Remove the model-specific filter
        rows = self.page.find_all("region[type=table-row]")
        columns = self.page.find_all("region[type=table-column]")

        # Filter to only include those that overlap with this table region
        def is_in_table(element):
            # Use a simple overlap check (more robust than just center point)
            # Check if element's bbox overlaps with self.bbox
            return (
                hasattr(element, "bbox")
                and element.x0 < self.x1  # Ensure element has bbox
                and element.x1 > self.x0
                and element.top < self.bottom
                and element.bottom > self.top
            )

        table_rows = [r for r in rows if is_in_table(r)]
        table_columns = [c for c in columns if is_in_table(c)]

        if not table_rows or not table_columns:
            # Use page's logger if available
            logger_instance = getattr(self._page, "logger", logger)
            logger_instance.warning(
                f"Region {self.bbox}: Cannot create cells. No overlapping row or column regions found."
            )
            return self  # Return self even if no cells created

        # Sort rows and columns
        table_rows.sort(key=lambda r: r.top)
        table_columns.sort(key=lambda c: c.x0)

        # Create cells and add them to the page's element manager
        created_count = 0
        for row in table_rows:
            for column in table_columns:
                # Calculate intersection bbox for the cell
                cell_x0 = max(row.x0, column.x0)
                cell_y0 = max(row.top, column.top)
                cell_x1 = min(row.x1, column.x1)
                cell_y1 = min(row.bottom, column.bottom)

                # Only create a cell if the intersection is valid (positive width/height)
                if cell_x1 > cell_x0 and cell_y1 > cell_y0:
                    # Create cell region at the intersection
                    cell = self.page.create_region(cell_x0, cell_y0, cell_x1, cell_y1)
                    # Set metadata
                    cell.source = "derived"
                    cell.region_type = "table-cell"  # Explicitly set type
                    cell.normalized_type = "table-cell"  # And normalized type
                    # Inherit model from the parent table region
                    cell.model = self.model
                    cell.parent_region = self  # Link cell to parent table region

                    # Add the cell region to the page's element manager
                    self.page._element_mgr.add_region(cell)
                    created_count += 1

        # Optional: Add created cells to the table region's children
        # self.child_regions.extend(cells_created_in_this_call) # Needs list management

        logger_instance = getattr(self._page, "logger", logger)
        logger_instance.info(
            f"Region {self.bbox} (Model: {self.model}): Created and added {created_count} cell regions."
        )

        return self  # Return self for chaining

    def ask(
        self,
        question: str,
        min_confidence: float = 0.1,
        model: str = None,
        debug: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Ask a question about the region content using document QA.

        This method uses a document question answering model to extract answers from the region content.
        It leverages both textual content and layout information for better understanding.

        Args:
            question: The question to ask about the region content
            min_confidence: Minimum confidence threshold for answers (0.0-1.0)
            model: Optional model name to use for QA (if None, uses default model)
            **kwargs: Additional parameters to pass to the QA engine

        Returns:
            Dictionary with answer details: {
                "answer": extracted text,
                "confidence": confidence score,
                "found": whether an answer was found,
                "page_num": page number,
                "region": reference to this region,
                "source_elements": list of elements that contain the answer (if found)
            }
        """
        try:
            from natural_pdf.qa.document_qa import get_qa_engine
        except ImportError:
            logger.error(
                "Question answering requires optional dependencies. Install with `pip install natural-pdf[qa]`"
            )
            return {
                "answer": None,
                "confidence": 0.0,
                "found": False,
                "page_num": self.page.number,
                "source_elements": [],
                "region": self,
            }

        # Get or initialize QA engine with specified model
        try:
            qa_engine = get_qa_engine(model_name=model) if model else get_qa_engine()
        except Exception as e:
            logger.error(f"Failed to initialize QA engine (model: {model}): {e}", exc_info=True)
            return {
                "answer": None,
                "confidence": 0.0,
                "found": False,
                "page_num": self.page.number,
                "source_elements": [],
                "region": self,
            }

        # Ask the question using the QA engine
        try:
            return qa_engine.ask_pdf_region(
                self, question, min_confidence=min_confidence, debug=debug, **kwargs
            )
        except Exception as e:
            logger.error(f"Error during qa_engine.ask_pdf_region: {e}", exc_info=True)
            return {
                "answer": None,
                "confidence": 0.0,
                "found": False,
                "page_num": self.page.number,
                "source_elements": [],
                "region": self,
            }

    def add_child(self, child):
        """
        Add a child region to this region.

        Used for hierarchical document structure when using models like Docling
        that understand document hierarchy.

        Args:
            child: Region object to add as a child

        Returns:
            Self for method chaining
        """
        self.child_regions.append(child)
        child.parent_region = self
        return self

    def get_children(self, selector=None):
        """
        Get immediate child regions, optionally filtered by selector.

        Args:
            selector: Optional selector to filter children

        Returns:
            List of child regions matching the selector
        """
        import logging

        logger = logging.getLogger("natural_pdf.elements.region")

        if selector is None:
            return self.child_regions

        # Use existing selector parser to filter
        from natural_pdf.selectors.parser import parse_selector, selector_to_filter_func

        try:
            selector_obj = parse_selector(selector)
            filter_func = selector_to_filter_func(selector_obj)  # Removed region=self
            matched = [child for child in self.child_regions if filter_func(child)]
            logger.debug(
                f"get_children: found {len(matched)} of {len(self.child_regions)} children matching '{selector}'"
            )
            return matched
        except Exception as e:
            logger.error(f"Error applying selector in get_children: {e}", exc_info=True)
            return []  # Return empty list on error

    def get_descendants(self, selector=None):
        """
        Get all descendant regions (children, grandchildren, etc.), optionally filtered by selector.

        Args:
            selector: Optional selector to filter descendants

        Returns:
            List of descendant regions matching the selector
        """
        import logging

        logger = logging.getLogger("natural_pdf.elements.region")

        all_descendants = []
        queue = list(self.child_regions)  # Start with direct children

        while queue:
            current = queue.pop(0)
            all_descendants.append(current)
            # Add current's children to the queue for processing
            if hasattr(current, "child_regions"):
                queue.extend(current.child_regions)

        logger.debug(f"get_descendants: found {len(all_descendants)} total descendants")

        # Filter by selector if provided
        if selector is not None:
            from natural_pdf.selectors.parser import parse_selector, selector_to_filter_func

            try:
                selector_obj = parse_selector(selector)
                filter_func = selector_to_filter_func(selector_obj)  # Removed region=self
                matched = [desc for desc in all_descendants if filter_func(desc)]
                logger.debug(f"get_descendants: filtered to {len(matched)} matching '{selector}'")
                return matched
            except Exception as e:
                logger.error(f"Error applying selector in get_descendants: {e}", exc_info=True)
                return []  # Return empty list on error

        return all_descendants

    # Removed recursive=True, find_all on region shouldn't be recursive by default
    # Renamed _find_all back to find_all
    # def find_all(self, selector, apply_exclusions=True, **kwargs):
    # See implementation above near get_elements

    def __repr__(self) -> str:
        """String representation of the region."""
        poly_info = " (Polygon)" if self.has_polygon else ""
        name_info = f" name='{self.name}'" if self.name else ""
        type_info = f" type='{self.region_type}'" if self.region_type else ""
        source_info = f" source='{self.source}'" if self.source else ""
        return f"<Region{name_info}{type_info}{source_info} bbox={self.bbox}{poly_info}>"

    def correct_ocr(
        self,
        correction_callback: Callable[[Any], Optional[str]],
    ) -> "Region":  # Return self for chaining
        """
        Applies corrections to OCR-generated text elements within this region
        using a user-provided callback function.

        Finds text elements within this region whose 'source' attribute starts
        with 'ocr' and calls the `correction_callback` for each, passing the
        element itself.

        The `correction_callback` should contain the logic to:
        1. Determine if the element needs correction.
        2. Perform the correction (e.g., call an LLM).
        3. Return the new text (`str`) or `None`.

        If the callback returns a string, the element's `.text` is updated.
        Metadata updates (source, confidence, etc.) should happen within the callback.

        Args:
            correction_callback: A function accepting an element and returning
                                 `Optional[str]` (new text or None).

        Returns:
            Self for method chaining.
        """
        # Find OCR elements specifically within this region
        # Note: We typically want to correct even if the element falls in an excluded area
        target_elements = self.find_all(selector="text[source=ocr]", apply_exclusions=False)

        # Delegate to the utility function
        _apply_ocr_correction_to_elements(
            elements=target_elements,  # Pass the ElementCollection directly
            correction_callback=correction_callback,
            caller_info=f"Region({self.bbox})",  # Pass caller info
        )

        return self  # Return self for chaining

    # --- Classification Mixin Implementation --- #
    def _get_classification_manager(self) -> "ClassificationManager":
        if not hasattr(self.page, 'pdf') or not hasattr(self.page.pdf, '_classification_manager'):
            raise AttributeError("ClassificationManager not accessible via page.pdf._classification_manager")
        return self.page.pdf._classification_manager

    def _get_classification_content(self, model_type: str) -> Union[str, "Image"]: # Use "Image" for lazy import
        if model_type == 'text':
            text_content = self.extract_text(layout=False) # Simple join for classification
            if not text_content or text_content.isspace():
                raise ValueError("Cannot classify region with 'text' model: No text content found.")
            return text_content
        elif model_type == 'vision':
            # Get resolution from manager/kwargs if possible, else default
            # We access manager via the method to ensure it's available
            manager = self._get_classification_manager()
            default_resolution = 150 # Manager doesn't store default res, set here
            # Note: classify() passes resolution via **kwargs if user specifies
            resolution = kwargs.get('resolution', default_resolution) if 'kwargs' in locals() else default_resolution

            img = self.to_image(
                resolution=resolution,
                include_highlights=False, # No highlights for classification input
                crop_only=True # Just the region content
            )
            if img is None:
                raise ValueError("Cannot classify region with 'vision' model: Failed to render image.")
            return img
        else:
            raise ValueError(f"Unsupported model_type for classification: {model_type}")

    def _get_metadata_storage(self) -> Dict[str, Any]:
        # Ensure metadata exists
        if not hasattr(self, 'metadata') or self.metadata is None:
            self.metadata = {}
        return self.metadata

    @property
    def category(self) -> Optional[str]:
        """Returns the top category label from the most recent classification, or None."""
        results = self.classification_results
        if results and results.get('scores'):
            return results['scores'][0].get('label')
        return None

    @property
    def category_confidence(self) -> Optional[float]:
        """Returns the top category confidence score from the most recent classification, or None."""
        results = self.classification_results
        if results and results.get('scores'):
            return results['scores'][0].get('confidence')
        return None

    @property
    def classification_results(self) -> Optional[Dict]:
        """Returns the full results dictionary from the most recent classification, or None."""
        return self._get_metadata_storage().get('classification')
    # --- End Classification Mixin Implementation --- #
