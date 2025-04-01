import logging
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)

@dataclass
class TextStyleOptions:
    """Options for configuring text style analysis."""

    # Properties to consider when grouping elements by style
    group_by: List[str] = field(default_factory=lambda: ['size', 'fontname', 'is_bold', 'is_italic', 'color'])

    # Tolerance for comparing font sizes (e.g., 0.5 rounds to nearest 0.5 point)
    size_tolerance: float = 0.5

    # If True, ignores text color during grouping
    ignore_color: bool = False

    # If True, ignores small variations often found in font names (e.g., '+ArialMT')
    normalize_fontname: bool = True

    # If True, generates descriptive labels (e.g., "12pt-Bold-Arial")
    # If False, uses simple numeric labels ("Style 1")
    descriptive_labels: bool = True

    # Prefix for generated labels (used if descriptive_labels is False or as fallback)
    label_prefix: str = "Style"

    # Format string for descriptive labels. Placeholders match keys in style_properties dict.
    # Example: "{size}pt {weight}{style} {family} ({color})"
    # Available keys: size, fontname, is_bold, is_italic, color, weight, style, family
    label_format: str = "{size}pt {weight}{style} {family}" # Default format without color


    def __post_init__(self):
        # Validate size_tolerance
        if self.size_tolerance <= 0:
            logger.warning(f"size_tolerance must be positive, setting to 0.1. Original value: {self.size_tolerance}")
            self.size_tolerance = 0.1

        # Ensure 'size' is always considered if tolerance is relevant
        if 'size' not in self.group_by and self.size_tolerance > 0:
            logger.debug("Adding 'size' to group_by keys because size_tolerance is set.")
            if 'size' not in self.group_by: self.group_by.append('size')

        if self.ignore_color and 'color' in self.group_by:
            logger.debug("Removing 'color' from group_by keys because ignore_color is True.")
            self.group_by = [key for key in self.group_by if key != 'color']
        elif not self.ignore_color and 'color' not in self.group_by:
             # If color isn't ignored, ensure it's included if requested in label format?
             # For now, just rely on explicit group_by setting.
             pass

        # Basic validation for group_by keys
        allowed_keys = {'size', 'fontname', 'is_bold', 'is_italic', 'color'}
        invalid_keys = set(self.group_by) - allowed_keys
        if invalid_keys:
            logger.warning(f"Invalid keys found in group_by: {invalid_keys}. Allowed keys: {allowed_keys}. Ignoring invalid keys.")
            self.group_by = [key for key in self.group_by if key in allowed_keys] 