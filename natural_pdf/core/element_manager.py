"""
Element Manager for natural-pdf.

This class handles the loading, creation, and management of PDF elements like
characters, words, rectangles, and lines extracted from a page.
"""

import logging
import re
from itertools import groupby
from typing import Any, Dict, List, Optional, Tuple, Union

from pdfplumber.utils.text import WordExtractor

from natural_pdf.elements.line import LineElement
from natural_pdf.elements.rect import RectangleElement
from natural_pdf.elements.text import TextElement

logger = logging.getLogger(__name__)


class NaturalWordExtractor(WordExtractor):
    """
    Custom WordExtractor that splits words based on specified character attributes
    in addition to pdfplumber's default spatial logic.
    """

    def __init__(self, word_split_attributes: List[str], extra_attrs: List[str], *args, **kwargs):
        """
        Initialize the extractor.

        Args:
            word_split_attributes: List of character attributes (keys in char dict)
                                   that should trigger a word split if they differ
                                   between adjacent characters.
            extra_attrs: List of character attributes (keys in char dict)
                         to copy from the first char of a word into the
                         resulting word dictionary.
            *args: Positional arguments passed to WordExtractor parent.
            **kwargs: Keyword arguments passed to WordExtractor parent.
        """
        self.word_split_attributes = word_split_attributes or []
        # Remove our custom arg before passing to parent
        # (Though WordExtractor likely ignores unknown kwargs)
        # Ensure it's removed if it exists in kwargs
        if "word_split_attributes" in kwargs:
            del kwargs["word_split_attributes"]
        # Pass extra_attrs to the parent constructor
        kwargs["extra_attrs"] = extra_attrs
        super().__init__(*args, **kwargs)

    def char_begins_new_word(
        self,
        prev_char: Dict[str, Any],
        curr_char: Dict[str, Any],
        direction: str,
        x_tolerance: float,
        y_tolerance: float,
    ) -> bool:
        """
        Determine if curr_char begins a new word, considering spatial and
        attribute differences.
        """
        # 1. Check pdfplumber's spatial logic first
        spatial_split = super().char_begins_new_word(
            prev_char, curr_char, direction, x_tolerance, y_tolerance
        )
        if spatial_split:
            return True

        # 2. Check for differences in specified attributes
        if self.word_split_attributes:
            for attr in self.word_split_attributes:
                # Use .get() for safety, although _prepare_char_dicts should ensure presence
                if prev_char.get(attr) != curr_char.get(attr):
                    logger.debug(
                        f"Splitting word due to attribute mismatch on '{attr}': {prev_char.get(attr)} != {curr_char.get(attr)}"
                    )
                    return True  # Attribute mismatch forces a new word

        # If both spatial and attribute checks pass, it's the same word
        return False


class ElementManager:
    """
    Manages the loading, creation, and retrieval of elements from a PDF page.

    This class centralizes the element management functionality previously
    contained in the Page class, providing better separation of concerns.
    """

    def __init__(self, page, font_attrs=None):
        """
        Initialize the ElementManager.

        Args:
            page: The parent Page object
            font_attrs: Font attributes to consider when grouping characters into words.
                       Default: ['fontname', 'size', 'bold', 'italic']
                       None: Only consider spatial relationships
                       List: Custom attributes to consider
        """
        self._page = page
        self._elements = None  # Lazy-loaded
        # Default to splitting by fontname, size, bold, italic if not specified
        # Renamed internal variable for clarity
        self._word_split_attributes = (
            ["fontname", "size", "bold", "italic"] if font_attrs is None else font_attrs
        )

    def load_elements(self):
        """
        Load all elements from the page (lazy loading).
        Uses NaturalWordExtractor for word grouping.
        """
        if self._elements is not None:
            return

        logger.debug(f"Page {self._page.number}: Loading elements...")

        # 1. Prepare character dictionaries (native + OCR) with necessary attributes
        prepared_char_dicts = self._prepare_char_dicts()
        logger.debug(
            f"Page {self._page.number}: Prepared {len(prepared_char_dicts)} character dictionaries."
        )

        # 2. Instantiate the custom word extractor
        # Get config settings from the parent PDF or use defaults
        pdf_config = getattr(self._page._parent, "_config", {})
        xt = pdf_config.get("x_tolerance", 3)
        yt = pdf_config.get("y_tolerance", 3)
        use_flow = pdf_config.get("use_text_flow", False)

        # Define which attributes to preserve on the merged word object
        # Should include split attributes + any others needed for filtering (like color)
        attributes_to_preserve = list(set(self._word_split_attributes + ["non_stroking_color"]))

        # Pass our configured attributes for splitting
        extractor = NaturalWordExtractor(
            word_split_attributes=self._word_split_attributes,
            extra_attrs=attributes_to_preserve,
            x_tolerance=xt,
            y_tolerance=yt,
            keep_blank_chars=True,
            use_text_flow=use_flow,
            # Assuming default directions are okay, configure if needed
            # line_dir=..., char_dir=...
        )

        # 3. Generate words using the extractor
        generated_words = []
        if prepared_char_dicts:
            # Sort chars primarily by upright status, then page reading order
            # Grouping by upright is crucial for WordExtractor's direction logic
            sorted_chars_for_extraction = sorted(
                prepared_char_dicts,
                key=lambda c: (c.get("upright", True), round(c.get("top", 0)), c.get("x0", 0)),
            )

            word_tuples = extractor.iter_extract_tuples(sorted_chars_for_extraction)
            for word_dict, char_list in word_tuples:
                # Convert the generated word_dict to a TextElement
                word_dict["_char_dicts"] = char_list
                word_element = self._create_word_element(word_dict)
                generated_words.append(word_element)
        logger.debug(
            f"Page {self._page.number}: Generated {len(generated_words)} words using NaturalWordExtractor."
        )

        # 4. Load other elements (rects, lines)
        rect_elements = [RectangleElement(r, self._page) for r in self._page._page.rects]
        line_elements = [LineElement(l, self._page) for l in self._page._page.lines]
        logger.debug(
            f"Page {self._page.number}: Loaded {len(rect_elements)} rects, {len(line_elements)} lines."
        )

        # 5. Create the final elements dictionary
        self._elements = {
            # Store original char elements if needed (e.g., for visualization/debugging)
            # We re-create them here from the prepared dicts
            "chars": [TextElement(c_dict, self._page) for c_dict in prepared_char_dicts],
            "words": generated_words,
            "rects": rect_elements,
            "lines": line_elements,
        }

        # Add regions if they exist
        if hasattr(self._page, "_regions") and (
            "detected" in self._page._regions or "named" in self._page._regions
        ):
            regions = []
            if "detected" in self._page._regions:
                regions.extend(self._page._regions["detected"])
            if "named" in self._page._regions:
                regions.extend(self._page._regions["named"].values())
            self._elements["regions"] = regions
            logger.debug(f"Page {self._page.number}: Added {len(regions)} regions.")
        else:
            self._elements["regions"] = []  # Ensure key exists

        logger.debug(f"Page {self._page.number}: Element loading complete.")

    def _prepare_char_dicts(self) -> List[Dict[str, Any]]:
        """
        Prepares a list of character dictionaries from native PDF characters,
        augmenting them with necessary attributes like bold/italic flags.
        This method focuses ONLY on native characters. OCR results are
        handled separately by create_text_elements_from_ocr.

        Returns:
            List of augmented native character dictionaries.
        """
        prepared_dicts = []
        processed_native_ids = set()  # To track processed native chars

        # 1. Process Native PDF Characters
        native_chars = self._page._page.chars or []
        logger.debug(f"Page {self._page.number}: Preparing {len(native_chars)} native char dicts.")
        for i, char_dict in enumerate(native_chars):
            # Create a temporary TextElement for analysis ONLY
            # We need to ensure the char_dict has necessary keys first
            if not all(k in char_dict for k in ["x0", "top", "x1", "bottom", "text"]):
                logger.warning(f"Skipping native char dict due to missing keys: {char_dict}")
                continue

            temp_element = TextElement(char_dict, self._page)

            # Augment the original dictionary
            augmented_dict = char_dict.copy()  # Work on a copy
            augmented_dict["bold"] = temp_element.bold
            augmented_dict["italic"] = temp_element.italic
            augmented_dict["source"] = "native"
            # Copy color if it exists
            if "non_stroking_color" in char_dict:
                augmented_dict["non_stroking_color"] = char_dict["non_stroking_color"]
            # Ensure basic required keys are present
            augmented_dict.setdefault("upright", True)
            augmented_dict.setdefault("fontname", "Unknown")
            augmented_dict.setdefault("size", 0)

            prepared_dicts.append(augmented_dict)
            # Use a unique identifier if available (e.g., tuple of key properties)
            # Simple approach: use index for now, assuming list order is stable here
            processed_native_ids.add(i)

        # 2. Remove OCR Processing from this method
        # OCR results will be added later via create_text_elements_from_ocr

        logger.debug(
            f"Page {self._page.number}: Total prepared native char dicts: {len(prepared_dicts)}"
        )
        return prepared_dicts

    def _create_word_element(self, word_dict: Dict[str, Any]) -> TextElement:
        """
        Create a TextElement (type 'word') from a word dictionary generated
        by NaturalWordExtractor/pdfplumber.

        Args:
            word_dict: Dictionary representing the word, including geometry,
                       text, and attributes copied from the first char
                       (e.g., fontname, size, bold, italic).

        Returns:
            TextElement representing the word.
        """
        # word_dict already contains calculated geometry (x0, top, x1, bottom, etc.)
        # and text content. We just need to ensure our required fields exist
        # and potentially set the source.

        # Start with a copy of the word_dict
        element_data = word_dict.copy()

        # Ensure required TextElement fields are present or add defaults
        element_data.setdefault("object_type", "word")  # Set type to 'word'
        element_data.setdefault("page_number", self._page.number)
        # Determine source based on attributes present (e.g., if 'confidence' exists, it's likely OCR)
        # This assumes the word_dict carries over some hint from its chars.
        # A simpler approach: assume 'native' unless fontname is 'OCR'.
        element_data.setdefault(
            "source", "ocr" if element_data.get("fontname") == "OCR" else "native"
        )
        element_data.setdefault(
            "confidence", 1.0 if element_data["source"] == "native" else 0.0
        )  # Default confidence

        # Bold/italic should already be in word_dict if they were split attributes,
        # copied from the first (representative) char by pdfplumber's merge_chars.
        # Ensure they exist for TextElement initialization.
        element_data.setdefault("bold", False)
        element_data.setdefault("italic", False)

        # Ensure fontname and size exist
        element_data.setdefault("fontname", "Unknown")
        element_data.setdefault("size", 0)

        # Store the constituent char dicts (passed alongside word_dict from extractor)
        # We need to modify the caller (load_elements) to pass this.
        # For now, assume it might be passed in word_dict for placeholder.
        element_data["_char_dicts"] = word_dict.get("_char_dicts", [])  # Store char list

        return TextElement(element_data, self._page)

    def create_text_elements_from_ocr(self, ocr_results, scale_x=None, scale_y=None):
        """
        Convert OCR results to TextElement objects AND adds them to the manager's
        'words' and 'chars' lists.

        This method should be called AFTER initial elements (native) might have
        been loaded, as it appends to the existing lists.

        Args:
            ocr_results: List of OCR results dictionaries with 'text', 'bbox', 'confidence'.
                         Confidence can be None for detection-only results.
            scale_x: Factor to convert image x-coordinates to PDF coordinates.
            scale_y: Factor to convert image y-coordinates to PDF coordinates.

        Returns:
            List of created TextElement word objects that were added.
        """
        added_word_elements = []
        if self._elements is None:
            # Trigger loading of native elements if not already done
            logger.debug(
                f"Page {self._page.number}: create_text_elements_from_ocr triggering initial load_elements."
            )
            self.load_elements()

        # Ensure scales are valid numbers
        scale_x = float(scale_x) if scale_x is not None else 1.0
        scale_y = float(scale_y) if scale_y is not None else 1.0

        logger.debug(
            f"Page {self._page.number}: Adding {len(ocr_results)} OCR results as elements. Scale: x={scale_x:.2f}, y={scale_y:.2f}"
        )

        # Ensure the target lists exist in the _elements dict
        if self._elements is None:
            logger.error(
                f"Page {self._page.number}: _elements dictionary is None after load_elements call in create_text_elements_from_ocr. Cannot add OCR elements."
            )
            return []  # Cannot proceed

        if "words" not in self._elements:
            self._elements["words"] = []
        if "chars" not in self._elements:
            self._elements["chars"] = []

        for result in ocr_results:
            try:
                x0_img, top_img, x1_img, bottom_img = map(float, result["bbox"])
                height_img = bottom_img - top_img
                pdf_x0 = x0_img * scale_x
                pdf_top = top_img * scale_y
                pdf_x1 = x1_img * scale_x
                pdf_bottom = bottom_img * scale_y
                pdf_height = (bottom_img - top_img) * scale_y

                # Handle potential None confidence
                raw_confidence = result.get("confidence")
                confidence_value = (
                    float(raw_confidence) if raw_confidence is not None else None
                )  # Keep None if it was None
                ocr_text = result.get("text")  # Get text, will be None if detect_only

                # Create the TextElement for the word
                word_element_data = {
                    "text": ocr_text,
                    "x0": pdf_x0,
                    "top": pdf_top,
                    "x1": pdf_x1,
                    "bottom": pdf_bottom,
                    "width": (x1_img - x0_img) * scale_x,
                    "height": pdf_height,
                    "object_type": "word",  # Treat OCR results as whole words
                    "source": "ocr",
                    "confidence": confidence_value,  # Use the handled confidence
                    "fontname": "OCR",  # Use consistent OCR fontname
                    "size": (
                        round(pdf_height) if pdf_height > 0 else 10.0
                    ),  # Use calculated PDF height for size
                    "page_number": self._page.number,
                    "bold": False,
                    "italic": False,
                    "upright": True,
                    "doctop": pdf_top + self._page._page.initial_doctop,
                }

                # Create the representative char dict for this OCR word
                ocr_char_dict = word_element_data.copy()
                ocr_char_dict["object_type"] = "char"
                ocr_char_dict.setdefault("adv", ocr_char_dict.get("width", 0))

                # Add the char dict list to the word data before creating TextElement
                word_element_data["_char_dicts"] = [ocr_char_dict]  # Store itself as its only char

                word_elem = TextElement(word_element_data, self._page)
                added_word_elements.append(word_elem)

                # Append the word element to the manager's list
                self._elements["words"].append(word_elem)

                # Only add a representative char dict if text actually exists
                if ocr_text is not None:
                    # This char dict represents the entire OCR word as a single 'char'.
                    char_dict_data = ocr_char_dict  # Use the one we already created
                    char_dict_data["object_type"] = "char"  # Mark as char type
                    char_dict_data.setdefault("adv", char_dict_data.get("width", 0))
                    self._elements["chars"].append(char_dict_data)  # Append the dictionary

            except (KeyError, ValueError, TypeError) as e:
                logger.error(f"Failed to process OCR result: {result}. Error: {e}", exc_info=True)
                continue

        logger.info(
            f"Page {self._page.number}: Appended {len(added_word_elements)} TextElements (words) and corresponding char dicts from OCR results."
        )
        return added_word_elements

    def add_element(self, element, element_type="words"):
        """
        Add an element to the managed elements.

        Args:
            element: The element to add
            element_type: The type of element ('words', 'chars', etc.)

        Returns:
            True if added successfully, False otherwise
        """
        # Load elements if not already loaded
        self.load_elements()

        # Add to the appropriate list
        if element_type in self._elements:
            # Avoid adding duplicates
            if element not in self._elements[element_type]:
                self._elements[element_type].append(element)
                return True
            else:
                # logger.debug(f"Element already exists in {element_type}: {element}")
                return False  # Indicate it wasn't newly added

        return False

    def add_region(self, region, name=None):
        """
        Add a region to the managed elements.

        Args:
            region: The region to add
            name: Optional name for the region

        Returns:
            True if added successfully, False otherwise
        """
        # Load elements if not already loaded
        self.load_elements()

        # Make sure regions is in _elements
        if "regions" not in self._elements:
            self._elements["regions"] = []

        # Add to elements for selector queries
        if region not in self._elements["regions"]:
            self._elements["regions"].append(region)
            return True

        return False

    def get_elements(self, element_type=None):
        """
        Get all elements of the specified type, or all elements if type is None.

        Args:
            element_type: Optional element type ('words', 'chars', 'rects', 'lines', 'regions' etc.)

        Returns:
            List of elements
        """
        # Load elements if not already loaded
        self.load_elements()

        if element_type:
            return self._elements.get(element_type, [])

        # Combine all element types
        all_elements = []
        for elements in self._elements.values():
            all_elements.extend(elements)

        return all_elements

    def get_all_elements(self):
        """
        Get all elements from all types.

        Returns:
            List of all elements
        """
        # Load elements if not already loaded
        self.load_elements()

        # Combine all element types
        all_elements = []
        if self._elements:  # Ensure _elements is not None
            for elements in self._elements.values():
                if isinstance(elements, list):  # Ensure we only extend lists
                    all_elements.extend(elements)
        return all_elements

    @property
    def chars(self):
        """Get all character elements."""
        self.load_elements()
        return self._elements.get("chars", [])

    @property
    def words(self):
        """Get all word elements."""
        self.load_elements()
        return self._elements.get("words", [])

    @property
    def rects(self):
        """Get all rectangle elements."""
        self.load_elements()
        return self._elements.get("rects", [])

    @property
    def lines(self):
        """Get all line elements."""
        self.load_elements()
        return self._elements.get("lines", [])

    @property
    def regions(self):
        """Get all region elements."""
        self.load_elements()
        return self._elements.get("regions", [])

    def remove_ocr_elements(self):
        """
        Remove all elements with source="ocr" from the elements dictionary.
        This should be called before adding new OCR elements if replacement is desired.

        Returns:
            int: Number of OCR elements removed
        """
        # Load elements if not already loaded
        self.load_elements()

        removed_count = 0

        # Filter out OCR elements from words
        if "words" in self._elements:
            original_len = len(self._elements["words"])
            self._elements["words"] = [
                word for word in self._elements["words"] if getattr(word, "source", None) != "ocr"
            ]
            removed_count += original_len - len(self._elements["words"])

        # Filter out OCR elements from chars
        if "chars" in self._elements:
            original_len = len(self._elements["chars"])
            self._elements["chars"] = [
                char
                for char in self._elements["chars"]
                if (isinstance(char, dict) and char.get("source") != "ocr")
                or (not isinstance(char, dict) and getattr(char, "source", None) != "ocr")
            ]
            removed_count += original_len - len(self._elements["chars"])

        logger.info(f"Page {self._page.number}: Removed {removed_count} OCR elements.")
        return removed_count

    def remove_element(self, element, element_type="words"):
        """
        Remove a specific element from the managed elements.

        Args:
            element: The element to remove
            element_type: The type of element ('words', 'chars', etc.)

        Returns:
            bool: True if removed successfully, False otherwise
        """
        # Load elements if not already loaded
        self.load_elements()

        # Check if the collection exists
        if element_type not in self._elements:
            logger.warning(f"Cannot remove element: collection '{element_type}' does not exist")
            return False

        # Try to remove the element
        try:
            if element in self._elements[element_type]:
                self._elements[element_type].remove(element)
                logger.debug(f"Removed element from {element_type}: {element}")
                return True
            else:
                logger.debug(f"Element not found in {element_type}: {element}")
                return False
        except Exception as e:
            logger.error(f"Error removing element from {element_type}: {e}", exc_info=True)
            return False

    def has_elements(self) -> bool:
        """
        Check if any significant elements (words, rects, lines, regions)
        have been loaded or added.

        Returns:
            True if any elements exist, False otherwise.
        """
        self.load_elements()

        for key in ["words", "rects", "lines", "regions"]:
            if self._elements.get(key):
                return True

        return False
