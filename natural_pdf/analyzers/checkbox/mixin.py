"""Checkbox detection mixin for Page and Region classes."""

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

if TYPE_CHECKING:
    from natural_pdf.analyzers.checkbox.checkbox_options import CheckboxOptions
    from natural_pdf.elements.element_collection import ElementCollection
    from natural_pdf.elements.region import Region

logger = logging.getLogger(__name__)


class CheckboxDetectionMixin:
    """Mixin to add checkbox detection capabilities to Page and Region classes."""

    def detect_checkboxes(
        self,
        engine: Optional[str] = None,
        options: Optional[Union["CheckboxOptions", Dict[str, Any]]] = None,
        confidence: Optional[float] = None,
        resolution: Optional[int] = None,
        device: Optional[str] = None,
        existing: str = "replace",
        limit: Optional[int] = None,
        **kwargs,
    ) -> "ElementCollection[Region]":
        """
        Detect checkboxes in the page or region.

        This method identifies checkboxes and their states (checked/unchecked) using
        computer vision models. Detected checkboxes are added as Region objects with
        type="checkbox" and can be accessed via selectors like page.find_all('checkbox').

        Args:
            engine: Name of the detection engine (default: 'rtdetr' for wendys model)
            options: CheckboxOptions instance or dict of options for advanced configuration
            confidence: Minimum confidence threshold (default: 0.02 for DETR models)
            resolution: DPI for rendering pages to images (default: 150)
            device: Device for inference ('cpu', 'cuda', 'mps', etc.)
            existing: How to handle existing checkbox regions: 'replace' (default) or 'append'
            limit: Maximum number of checkboxes to detect (useful when you know the expected count)
            **kwargs: Additional engine-specific arguments

        Returns:
            ElementCollection containing detected checkbox Region objects with attributes:
            - region_type: "checkbox"
            - is_checked: bool indicating if checkbox is checked
            - checkbox_state: "checked" or "unchecked"
            - confidence: detection confidence score

        Examples:
            # Basic detection
            checkboxes = page.detect_checkboxes()

            # Find checked boxes
            checked = page.find_all('checkbox:checked')
            unchecked = page.find_all('checkbox:unchecked')

            # Limit to expected number
            checkboxes = page.detect_checkboxes(limit=10)

            # High confidence detection
            checkboxes = page.detect_checkboxes(confidence=0.9)

            # GPU acceleration
            checkboxes = page.detect_checkboxes(device='cuda')

            # Custom model
            from natural_pdf import CheckboxOptions
            options = CheckboxOptions(model_repo="your-org/your-checkbox-model")
            checkboxes = page.detect_checkboxes(options=options)
        """
        # Lazy import to avoid circular dependencies
        from natural_pdf.analyzers.checkbox.checkbox_analyzer import CheckboxAnalyzer

        # Create analyzer
        analyzer = CheckboxAnalyzer(self)

        # Run detection
        regions = analyzer.detect_checkboxes(
            engine=engine,
            options=options,
            confidence=confidence,
            resolution=resolution,
            device=device,
            existing=existing,
            limit=limit,
            **kwargs,
        )

        # Return as ElementCollection
        from natural_pdf.elements.element_collection import ElementCollection

        return ElementCollection(regions)
