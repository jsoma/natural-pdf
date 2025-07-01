"""
Text structure analyzer for natural-pdf.
"""

import logging
import re
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import jenkspy  # Added import for jenkspy

from natural_pdf.analyzers.text_options import TextStyleOptions

if TYPE_CHECKING:
    from natural_pdf.core.page import Page
    from natural_pdf.elements.base import Element
    from natural_pdf.elements.element_collection import ElementCollection

logger = logging.getLogger(__name__)

# Simple regex to remove common PDF font prefixes like "ABCDEF+"
FONT_PREFIX_RE = re.compile(r"^[A-Z]{6}\+")

# Common font weight/style keywords
FONT_WEIGHTS = {
    "bold": "Bold",
    "black": "Bold",
    "heavy": "Bold",
    "medium": "",
    "light": "Light",
    "thin": "Thin",
}
FONT_STYLES = {"italic": "Italic", "oblique": "Italic"}

# Constants for automatic font size bucketing
MAX_UNIQUE_SIZES_FOR_JENKS_INPUT = (
    3000  # Max unique sizes to feed directly into Jenks; uses sampling above this
)
DEFAULT_MAX_AUTO_BUCKETS = 7  # Max number of buckets to try when font_size_buckets='auto'
MIN_BUCKETS_FOR_AUTO = 2


class TextStyleAnalyzer:
    """
    Analyzes and groups text elements by their style properties based on configuration.

    This analyzer groups text elements based on specified font properties
    (controlled by TextStyleOptions) and adds 'style_label', 'style_key',
    and 'style_properties' attributes to each processed text element.
    """

    def __init__(self, options: Optional[TextStyleOptions] = None):
        """
        Initialize the text style analyzer.

        Args:
            options: Configuration options for the analysis. Uses default if None.
        """
        self.options = options or TextStyleOptions()
        logger.debug(f"Initialized TextStyleAnalyzer with options: {self.options}")

        # To store the font size bucket mapper if bucketing is active
        self._font_size_bucket_mapper = None
        self._font_size_bucket_count = 0

    def _calculate_jenks_breaks(self, data: List[float], num_classes: int) -> List[float]:
        if not data or num_classes <= 1:
            return []

        unique_data = sorted(
            list(set(data))
        )  # jenkspy works best with unique, sorted data for clarity of breaks
        if len(unique_data) < 2 or len(unique_data) < num_classes:
            # Not enough unique data points to form meaningful breaks for the requested number of classes
            # or no way to make breaks if fewer than 2 unique points.
            # If len(unique_data) == 1, and num_classes > 1, jenkspy might error or give trivial breaks.
            # If num_classes is 1, we already returned [].
            # If len(unique_data) < num_classes, it means we cannot have num_classes distinct groups based on these unique points.
            # The calling function _get_font_size_bucket_mapper already adjusts num_classes if it's > len(unique_data)
            # so this condition here is a safeguard or handles cases where data is extremely sparse.
            if (
                len(unique_data) > 1 and num_classes > 1
            ):  # Try to make at least one break if possible
                # Fallback: create breaks between all unique points if Jenks is not suitable
                # This ensures we get some division if possible, up to num_classes-1 breaks
                # breaks = [(unique_data[i] + unique_data[i+1]) / 2.0 for i in range(len(unique_data)-1)]
                # return sorted(list(set(breaks)))[:num_classes-1]
                # However, with jenkspy, it might be better to let it try and handle its output.
                # If jenkspy cannot form num_classes, it might return fewer breaks or specific values.
                pass  # Let jenkspy attempt it, its behavior for sparse data is specific to its C implementation.
            else:
                return []  # Cannot form breaks

        try:
            # jenkspy.jenks_breaks returns all boundaries, including min and max of data
            # e.g., for n_classes=5, it returns 6 values: [min, break1, break2, break3, break4, max]
            all_boundaries = jenkspy.jenks_breaks(
                unique_data, n_classes=num_classes
            )  # Use unique_data

            # We need the inner breaks: [break1, break2, break3, break4]
            if len(all_boundaries) > 2:  # Ensure there are inner breaks
                inner_breaks = all_boundaries[1:-1]
                return sorted(list(set(inner_breaks)))  # Ensure breaks are unique and sorted
            else:
                # This case implies n_classes=1 or data was so uniform jenkspy couldn't break it
                return []
        except Exception as e:
            logger.warning(
                f"jenkspy.jenks_breaks failed with {num_classes} classes for data (first 10 shown): {unique_data[:10]}. Error: {e}. Falling back to no breaks for this k."
            )
            return []  # Fallback if jenkspy fails

    def _calculate_gvf(self, data: List[float], breaks: List[float]) -> float:
        if not data:
            return 0.0
        overall_mean = sum(data) / len(data)
        sdam = sum([(x - overall_mean) ** 2 for x in data])
        if sdam == 0:
            return 1.0  # Perfect fit if all data points are the same

        sdcm = 0.0
        all_breaks = [-float("inf")] + breaks + [float("inf")]
        for i in range(len(all_breaks) - 1):
            lower_bound = all_breaks[i]
            upper_bound = all_breaks[i + 1]
            cluster = [x for x in data if x > lower_bound and x <= upper_bound]
            if not cluster:
                continue
            cluster_mean = sum(cluster) / len(cluster)
            sdcm += sum([(x - cluster_mean) ** 2 for x in cluster])

        return (sdam - sdcm) / sdam if sdam > 0 else 1.0

    def _get_font_size_bucket_mapper(
        self, all_font_sizes: List[float], config: Union[List[float], int, str]
    ) -> Tuple[Optional[Callable[[float], int]], int]:
        if not all_font_sizes:
            return None, 0

        unique_font_sizes = sorted(list(set(s for s in all_font_sizes if s is not None)))
        if not unique_font_sizes:
            return None, 0

        # Apply sampling if too many unique font sizes for Jenks input
        jenks_input_data = unique_font_sizes
        if len(unique_font_sizes) > MAX_UNIQUE_SIZES_FOR_JENKS_INPUT:
            logger.debug(
                f"Sampling {MAX_UNIQUE_SIZES_FOR_JENKS_INPUT} from {len(unique_font_sizes)} unique font sizes for Jenks."
            )
            # Simple uniform sampling from sorted unique values
            indices = [
                int(i * (len(unique_font_sizes) - 1) / (MAX_UNIQUE_SIZES_FOR_JENKS_INPUT - 1))
                for i in range(MAX_UNIQUE_SIZES_FOR_JENKS_INPUT)
            ]
            jenks_input_data = [unique_font_sizes[i] for i in indices]
            jenks_input_data = sorted(list(set(jenks_input_data)))  # Ensure still sorted and unique

        breaks: List[float] = []
        num_buckets = 0

        if isinstance(config, list):  # Explicit boundaries
            breaks = sorted(list(set(config)))  # Ensure sorted and unique
            num_buckets = len(breaks) + 1
        elif isinstance(config, int):  # User-defined number of buckets
            num_buckets_to_find = config
            if num_buckets_to_find <= 0:
                logger.warning(f"Invalid number of buckets ({config}), disabling bucketing.")
                return None, 0
            if num_buckets_to_find == 1:
                return (lambda size: 0), 1  # All in one bucket
            if (
                not jenks_input_data
                or len(jenks_input_data) < num_buckets_to_find
                and len(jenks_input_data) > 0
            ):
                logger.debug(
                    f"Not enough unique font sizes ({len(jenks_input_data)}) to create {num_buckets_to_find} distinct buckets based on input data. Adjusting."
                )
                # Fallback to fewer buckets if not enough unique data points to separate
                num_buckets_to_find = max(
                    1, len(jenks_input_data) - 1 if len(jenks_input_data) > 1 else 1
                )
                if num_buckets_to_find == 1:
                    return (lambda size: 0), 1

            breaks = self._calculate_jenks_breaks(jenks_input_data, num_buckets_to_find)
            num_buckets = len(breaks) + 1
        elif config == "auto":
            best_gvf = -1.0
            best_breaks = []
            best_k = 0
            # Iterate from MIN_BUCKETS_FOR_AUTO up to a max (or len of data if smaller)
            max_k_to_try = min(
                DEFAULT_MAX_AUTO_BUCKETS,
                len(jenks_input_data) if jenks_input_data else MIN_BUCKETS_FOR_AUTO,
            )
            if len(jenks_input_data) == 1:  # Only one unique font size
                return (lambda size: 0), 1

            for k_buckets in range(MIN_BUCKETS_FOR_AUTO, max_k_to_try + 1):
                if k_buckets > len(
                    jenks_input_data
                ):  # Cannot have more buckets than unique data points
                    break
                current_breaks = self._calculate_jenks_breaks(jenks_input_data, k_buckets)
                if (
                    len(current_breaks) != k_buckets - 1
                ):  # Jenks couldn't find enough distinct breaks
                    # This can happen if data points are too few or clustered.
                    # If we requested k_buckets, we expect k_buckets-1 breaks.
                    # If we get fewer, it implies the effective number of buckets is less.
                    # We should only proceed if number of breaks matches k_buckets-1 for a valid GVF.
                    if (
                        k_buckets > 1 and not current_breaks
                    ):  # requested multiple buckets but got no breaks
                        continue
                    # else: proceed with fewer breaks which means fewer effective buckets for GVF.

                gvf = self._calculate_gvf(jenks_input_data, current_breaks)
                # Simple strategy: pick k with highest GVF.
                # More sophisticated: look for an elbow or significant GVF jump.
                if gvf > best_gvf:
                    best_gvf = gvf
                    best_breaks = current_breaks
                    best_k = len(current_breaks) + 1  # Number of buckets is breaks + 1

            breaks = best_breaks
            num_buckets = best_k if best_k > 0 else 1  # Ensure at least 1 bucket
            if num_buckets == 1 and breaks:  # If only 1 bucket, there should be no breaks
                breaks = []
            logger.debug(
                f"Auto bucketing: Chose {num_buckets} buckets with GVF {best_gvf:.4f}. Breaks: {breaks}"
            )

        else:
            return None, 0  # Invalid config or no bucketing

        if not breaks and num_buckets > 1 and len(unique_font_sizes) > 1:
            # This can happen if Jenks fails to find breaks for N > 1 buckets but config specified N > 1
            # Or if auto chose num_buckets > 1 but ended up with no breaks.
            # Fallback to treating all as one bucket if no breaks were determined for multiple requested buckets.
            logger.debug(
                f"No breaks determined for {num_buckets} requested buckets. Treating as 1 bucket."
            )
            num_buckets = 1
        elif num_buckets <= 1 and breaks:  # Contradiction: 1 bucket should have no breaks
            breaks = []
            num_buckets = 1

        final_breaks = sorted(list(set(breaks)))  # Ensure unique and sorted

        if not final_breaks and len(unique_font_sizes) > 1 and num_buckets > 1:
            # If still no breaks but we expect multiple buckets (e.g. config=2, unique_sizes=[10,12])
            # This implies Jenks failed to produce breaks. Fallback to simpler split for 2 buckets.
            if num_buckets == 2 and len(unique_font_sizes) >= 2:
                mid_point = (unique_font_sizes[0] + unique_font_sizes[-1]) / 2.0
                final_breaks = [mid_point]
                logger.debug(f"Jenks failed for 2 buckets, using midpoint break: {final_breaks}")
            else:  # For >2 buckets and no breaks, it defaults to 1 bucket effectively.
                num_buckets = 1
        elif final_breaks and num_buckets <= 1:
            num_buckets = len(final_breaks) + 1  # Recalculate num_buckets from actual breaks

        if num_buckets <= 1:  # If effectively one bucket (or no data to bucket)
            return (lambda size: 0), 1

        # Create a mapper function
        def mapper(size: float) -> int:
            if size is None:
                return -1  # Or some other indicator for unbucketable
            # Find which bucket the size falls into
            # bisect_left finds insertion point, which corresponds to bucket index
            bucket_index = 0
            for i, break_val in enumerate(final_breaks):
                if size <= break_val:
                    return i
            return len(final_breaks)  # Belongs to the last bucket

        return mapper, num_buckets

    def analyze(
        self, page: "Page", options: Optional[TextStyleOptions] = None
    ) -> "ElementCollection":
        from natural_pdf.elements.element_collection import ElementCollection

        current_options = options or self.options
        logger.info(
            f"Starting text style analysis for page {page.number} with options: {current_options}"
        )

        text_elements = page.words
        if not text_elements:
            text_elements = page.find_all("text").elements

        if not text_elements:
            logger.warning(f"Page {page.number} has no text elements to analyze.")
            return ElementCollection([])

        # --- Font Size Bucketing Setup ---
        self._font_size_bucket_mapper = None
        self._font_size_bucket_count = 0
        bucketing_config = getattr(current_options, "font_size_buckets", None)

        if bucketing_config is not None:
            all_page_font_sizes = [
                el.size for el in text_elements if hasattr(el, "size") and el.size is not None
            ]
            if all_page_font_sizes:
                self._font_size_bucket_mapper, self._font_size_bucket_count = (
                    self._get_font_size_bucket_mapper(all_page_font_sizes, bucketing_config)
                )
                if self._font_size_bucket_mapper:
                    logger.debug(
                        f"Font size bucketing active with {self._font_size_bucket_count} buckets for page {page.number}."
                    )
            else:
                logger.debug("No font sizes found on page for bucketing.")
        # --- End Bucketing Setup ---

        style_cache: Dict[Tuple, Dict[str, Any]] = {}
        processed_elements: List["Element"] = []
        group_by_keys = sorted(current_options.group_by)

        for element in text_elements:
            if not hasattr(element, "text") or not hasattr(element, "size"):
                logger.debug(f"Skipping element without text/size: {element}")
                continue

            try:
                style_properties = self._extract_style_properties(element, current_options)
                style_key = self._create_style_key(style_properties, group_by_keys)

                if style_key not in style_cache:
                    label = self._generate_style_label(
                        style_properties,
                        current_options,
                        len(style_cache) + 1,
                        self._font_size_bucket_count,
                    )
                    style_cache[style_key] = {"label": label, "properties": style_properties}
                    logger.debug(
                        f"New style detected (Key: {style_key}): Label='{label}', Props={style_properties}"
                    )

                element.style_label = style_cache[style_key]["label"]
                element.style_key = style_key
                element.style_properties = style_cache[style_key]["properties"]
                element.font_bucket_name = style_cache[style_key]["properties"].get(
                    "font_bucket_name"
                )
                processed_elements.append(element)
            except Exception as e:
                logger.warning(
                    f"Error processing element {element} for text style: {e}", exc_info=True
                )

        page._text_styles_summary = style_cache
        logger.info(
            f"Finished text style analysis for page {page.number}. Found {len(style_cache)} unique styles."
        )
        return ElementCollection(processed_elements)

    def _extract_style_properties(
        self, element: "Element", options: TextStyleOptions
    ) -> Dict[str, Any]:
        properties = {}
        original_size = getattr(element, "size", None)
        rounded_size = None

        properties["original_size"] = original_size

        if original_size is not None:
            rounding_factor = 1.0 / options.size_tolerance
            rounded_size = round(original_size * rounding_factor) / rounding_factor
        properties["size"] = rounded_size  # For display in labels
        properties["rounded_size"] = rounded_size  # Explicit storage

        # Font size bucketing logic
        properties["font_bucket_id"] = None
        properties["font_bucket_name"] = None  # Initialize font_bucket_name
        size_for_keying = rounded_size

        if self._font_size_bucket_mapper and original_size is not None:
            bucket_id = self._font_size_bucket_mapper(original_size)
            properties["font_bucket_id"] = bucket_id
            properties["font_bucket_name"] = self._get_bucket_name(
                bucket_id, self._font_size_bucket_count
            )
            size_for_keying = bucket_id

        properties["size_for_keying"] = size_for_keying

        font_name = None
        normalized_font_name = None
        if hasattr(element, "fontname") and element.fontname is not None:
            font_name = element.fontname
            normalized_font_name = self._normalize_font_name(font_name, options)
        properties["fontname"] = normalized_font_name if options.normalize_fontname else font_name

        # Font characteristics (derived from normalized name if available)
        name_to_check = normalized_font_name or font_name or ""
        name_lower = name_to_check.lower()
        is_bold = (
            "bold" in name_lower
            or "black" in name_lower
            or "heavy" in name_lower
            or name_to_check.endswith("-B")
        )
        is_italic = (
            "italic" in name_lower or "oblique" in name_lower or name_to_check.endswith("-I")
        )

        properties["is_bold"] = is_bold
        properties["is_italic"] = is_italic

        # Text color
        color = None
        if (
            not options.ignore_color
            and hasattr(element, "non_stroking_color")
            and element.non_stroking_color is not None
        ):
            raw_color = element.non_stroking_color
            # Convert color to a hashable form (tuple)
            if isinstance(raw_color, (list, tuple)):
                color = tuple(round(c, 3) for c in raw_color)  # Round color components
            else:
                # Handle simple grayscale or other non-list representations if needed
                try:
                    color = round(float(raw_color), 3)
                except (ValueError, TypeError):
                    color = str(raw_color)  # Fallback to string if cannot convert
            # Normalize common colors (optional, could be complex)
            # Example: (0.0, 0.0, 0.0) -> 'black', (1.0, 1.0, 1.0) -> 'white'
            if color == (0.0, 0.0, 0.0) or color == 0.0:
                color = "black"
            if color == (1.0, 1.0, 1.0) or color == 1.0:
                color = "white"
        properties["color"] = color

        return properties

    def _normalize_font_name(self, fontname: str, options: TextStyleOptions) -> str:
        """Basic normalization of font names."""
        if not options.normalize_fontname:
            return fontname
        # Remove common subset prefixes like "ABCDEF+"
        name = FONT_PREFIX_RE.sub("", fontname)
        # Could add more rules here, e.g., removing version numbers, standardizing separators
        return name

    def _parse_font_name(self, normalized_fontname: str) -> Dict[str, str]:
        """Attempt to parse family, weight, and style from a font name. Very heuristic."""
        if not normalized_fontname:
            return {"family": "Unknown", "weight": "", "style": ""}

        parts = re.split(r"[-,_ ]", normalized_fontname)
        family_parts = []
        weight = ""
        style = ""

        for part in parts:
            part_lower = part.lower()
            found = False
            # Check weights
            for key, val in FONT_WEIGHTS.items():
                if key in part_lower:
                    weight = val
                    found = True
                    break
            if found:
                continue  # Skip part if it was a weight

            # Check styles
            for key, val in FONT_STYLES.items():
                if key in part_lower:
                    style = val
                    found = True
                    break
            if found:
                continue  # Skip part if it was a style

            # If not weight or style, assume it's part of the family name
            if part:  # Avoid empty strings from multiple delimiters
                family_parts.append(part)

        family = "".join(family_parts) or "Unknown"  # Join remaining parts
        # Simple cleanup: Remove "MT" often appended? Maybe too aggressive.
        # if family.endswith("MT"): family = family[:-2]

        return {"family": family, "weight": weight, "style": style}

    def _create_style_key(self, properties: Dict[str, Any], group_by_keys: List[str]) -> Tuple:
        key_parts = []
        for key in group_by_keys:
            if key == "size":
                value = properties.get("size_for_keying")  # Use the correct size value for keying
            else:
                value = properties.get(key)

            if isinstance(value, list):
                value = tuple(value)
            key_parts.append(value)
        return tuple(key_parts)

    def _generate_style_label(
        self,
        properties: Dict[str, Any],
        options: TextStyleOptions,
        style_index: int,
        num_font_buckets: int = 0,
    ) -> str:
        if not options.descriptive_labels:
            # If bucketing is active and only 1 bucket, it's not very informative
            is_meaningful_bucketing = (
                self._font_size_bucket_mapper is not None and num_font_buckets > 1
            )
            bucket_id = properties.get("font_bucket_id")
            if is_meaningful_bucketing and bucket_id is not None:
                return f"{options.label_prefix} (Bucket {bucket_id + 1}) {style_index}"
            return f"{options.label_prefix} {style_index}"

        try:
            font_details = self._parse_font_name(properties.get("fontname", ""))
            bucket_label_part = ""
            bucket_id = properties.get("font_bucket_id")

            # Only add bucket info if bucketing is active and meaningful (more than 1 bucket)
            if (
                self._font_size_bucket_mapper is not None
                and num_font_buckets > 1
                and bucket_id is not None
            ):
                bucket_label_part = f" (Bucket {bucket_id + 1})"  # Simple numeric label for now

            label_data = {
                "size": properties.get("rounded_size", "?"),  # Use rounded_size for display
                "fontname": properties.get("fontname", "Unknown"),
                "is_bold": properties.get("is_bold", False),
                "is_italic": properties.get("is_italic", False),
                "color": properties.get("color", ""),
                "family": font_details["family"],
                # Use parsed weight/style if available, otherwise fallback to is_bold/is_italic flags
                "weight": font_details["weight"] or ("Bold" if properties.get("is_bold") else ""),
                "style": font_details["style"] or ("Italic" if properties.get("is_italic") else ""),
            }
            # Ensure style has a space separator if both weight and style exist
            if label_data["weight"] and label_data["style"]:
                label_data["style"] = " " + label_data["style"]

            label_data["bucket_info"] = bucket_label_part

            # Handle color formatting for label
            color_val = label_data["color"]
            if isinstance(color_val, tuple):
                color_str = f"rgb{color_val}"  # Basic tuple representation
            elif isinstance(color_val, str):
                color_str = color_val  # Already string ('black', 'white', or fallback)
            else:
                color_str = str(color_val)  # Other types
            label_data["color_str"] = color_str

            # Format the label, handle potential missing keys in format string gracefully
            # Add {bucket_info} to default format string if not already customized by user?
            # For now, user would need to add {bucket_info} to their custom label_format if they want it.
            current_label_format = options.label_format
            bucket_name_for_label = properties.get("font_bucket_name")

            # Construct a bucket_info string if a bucket name exists and it's not already in the format
            # And if there are multiple buckets to make it meaningful.
            bucket_info_str = ""
            if bucket_name_for_label and num_font_buckets > 1:
                bucket_info_str = f" ({bucket_name_for_label})"

            if "{bucket_info}" not in current_label_format and bucket_info_str:
                current_label_format += " {bucket_info}"  # Placeholder name for format_map

            # Populate label_data with the actual bucket string for the {bucket_info} placeholder
            label_data["bucket_info"] = bucket_info_str

            label = current_label_format.format_map(defaultdict(str, label_data))
            return label.strip().replace("  ", " ")
        except Exception as e:
            logger.warning(
                f"Error generating descriptive label for style {properties}: {e}. Falling back to numeric label."
            )
            # Fallback to numeric label on error
            return f"{options.label_prefix} {style_index}"

    def _get_bucket_name(self, bucket_id: Optional[int], total_buckets: int) -> Optional[str]:
        if bucket_id is None or not (0 <= bucket_id < total_buckets):
            return None  # Or "N/A"

        if total_buckets <= 0:  # Should not happen if called correctly
            return f"Invalid Bucket {bucket_id}"

        # Predefined human-readable names for up to 8 buckets
        # Buckets are 0-indexed internally, names correspond to that index.
        bucket_name_sets = {
            1: ["standard"],
            2: ["small", "large"],
            3: ["small", "medium", "large"],
            4: ["small", "medium", "large", "x-large"],
            5: ["x-small", "small", "medium", "large", "x-large"],
            6: ["x-small", "small", "medium", "large", "x-large", "xx-large"],
            7: ["xx-small", "x-small", "small", "medium", "large", "x-large", "xx-large"],
            8: [
                "xx-small",
                "x-small",
                "small",
                "medium",
                "large",
                "x-large",
                "xx-large",
                "xxx-large",
            ],
        }

        if total_buckets in bucket_name_sets:
            names = bucket_name_sets[total_buckets]
            if 0 <= bucket_id < len(names):
                return names[bucket_id]
            else:  # Should not happen if bucket_id is valid for total_buckets
                return f"Size Group {bucket_id}"
        else:  # Fallback for more than 8 buckets or unhandled cases
            return f"Size Group {bucket_id}"
