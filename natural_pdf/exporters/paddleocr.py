import logging
import os
import random
import shutil
from typing import TYPE_CHECKING, List, Optional, Set, Tuple, Union

from tqdm import tqdm

from natural_pdf.exporters.base import FinetuneExporter

# Need to import this utility
from natural_pdf.utils.identifiers import generate_short_path_hash

if TYPE_CHECKING:
    from natural_pdf.collections.pdf_collection import PDFCollection
    from natural_pdf.core.pdf import PDF
    from natural_pdf.elements.text import TextElement

logger = logging.getLogger(__name__)

DEFAULT_SELECTOR_CORRECTED = "text[source^=manifest]"  # Match manifest-import etc.


class PaddleOCRRecognitionExporter(FinetuneExporter):
    """
    Exports data for fine-tuning a PaddleOCR text recognition model.

    Creates a directory structure with cropped text images and label files
    (`train.txt`, `val.txt`, or `label.txt`) suitable for PaddleOCR training.
    Optionally includes a Jupyter Notebook guide for fine-tuning on Colab.
    """

    def __init__(
        self,
        resolution: int = 150,
        padding: int = 2,
        selector: Optional[str] = None,
        corrected_only: bool = False,
        split_ratio: Optional[float] = 0.9,
        include_guide: bool = True,
        random_seed: Optional[int] = 42,
    ):
        """
        Initialize the PaddleOCR Recognition Exporter.

        Args:
            resolution: DPI resolution for rendering text region images (default: 150).
            padding: Padding (in points) to add around text element bbox before cropping (default: 2).
            selector: CSS-like selector to filter which TextElements to export.
                      If None and corrected_only is False, all 'text' elements are considered.
            corrected_only: If True, overrides selector and exports only elements likely
                            originating from a correction manifest (selector="text[source=manifest]").
                            (default: False).
            split_ratio: Ratio for splitting data into training/validation sets (e.g., 0.9 for 90% train).
                         If None, creates a single `label.txt` file (default: 0.9).
            include_guide: If True, includes a template Jupyter Notebook guide for fine-tuning
                           in the output directory (default: True).
            random_seed: Seed for the random number generator used for train/val split shuffling,
                         ensuring reproducibility (default: 42).
        """
        if corrected_only and selector:
            logger.warning(
                f"Both 'corrected_only=True' and 'selector=\"{selector}\"' were provided. "
                f"Using corrected_only=True (selector='{DEFAULT_SELECTOR_CORRECTED}')."
            )
            self.selector = DEFAULT_SELECTOR_CORRECTED
        elif corrected_only:
            self.selector = DEFAULT_SELECTOR_CORRECTED
        elif selector:
            self.selector = selector
        else:
            self.selector = "text"  # Default to all text elements if nothing else specified

        self.resolution = resolution
        self.padding = padding
        self.split_ratio = split_ratio
        self.include_guide = include_guide
        self.random_seed = random_seed

        logger.info(
            f"Initialized PaddleOCRRecognitionExporter: selector='{self.selector}', resolution={resolution}, "
            f"padding={padding}, split_ratio={split_ratio}, include_guide={include_guide}"
        )

    def export(
        self,
        source: Union["PDF", "PDFCollection", List["PDF"]],
        output_dir: str,
        **kwargs,  # Allow for potential future args
    ):
        """
        Exports text elements from the source PDF(s) to the specified output directory
        in PaddleOCR text recognition format.

        Args:
            source: The PDF object, PDFCollection, or list of PDF objects to process.
            output_dir: The path to the directory where the exported files will be saved.
                        The directory will be created if it doesn't exist.
            **kwargs: Optional keyword arguments (currently unused).
        """
        # --- 1. Setup and Validation ---
        pdfs_to_process = self._resolve_source_pdfs(source)
        if not pdfs_to_process:
            logger.error("No valid PDF sources found. Aborting export.")
            return

        try:
            os.makedirs(output_dir, exist_ok=True)
            images_dir = os.path.join(output_dir, "images")
            os.makedirs(images_dir, exist_ok=True)
        except OSError as e:
            logger.error(f"Failed to create output directory '{output_dir}': {e}", exc_info=True)
            raise

        # --- 2. Collect Elements and Render Images ---
        labels: List[Tuple[str, str]] = []  # List of (relative_image_path, text_label)
        char_set: Set[str] = set()
        elements_processed = 0
        elements_skipped = 0

        logger.info(
            f"Processing {len(pdfs_to_process)} PDF(s) to find elements matching selector: '{self.selector}'"
        )

        for pdf in tqdm(pdfs_to_process, desc="Processing PDFs"):
            # Need to ensure pdf.path exists and is string
            if not hasattr(pdf, "path") or not isinstance(pdf.path, str):
                logger.warning(f"Skipping PDF object without a valid path attribute: {pdf}")
                continue
            pdf_hash = generate_short_path_hash(pdf.path)
            try:
                # Find elements using the specified selector
                # Need to check if pdf has find_all method
                if not hasattr(pdf, "find_all"):
                    logger.warning(
                        f"PDF object {pdf.path} does not have find_all method. Skipping."
                    )
                    continue

                elements = pdf.find_all(
                    self.selector, apply_exclusions=False
                )  # Usually want all text, even if excluded
                if not elements:
                    logger.debug(f"No elements matching '{self.selector}' found in {pdf.path}")
                    continue

                for i, element in enumerate(
                    tqdm(
                        elements,
                        desc=f"Exporting '{os.path.basename(pdf.path)}'",
                        leave=False,
                        position=1,
                    )
                ):
                    # Ensure it's a TextElement with necessary methods/attributes
                    # Removed check for to_image as it's called after expand()
                    if not (
                        hasattr(element, "page")
                        and hasattr(element, "text")
                        and hasattr(element, "expand")
                    ):
                        logger.warning(f"Skipping invalid/non-text element {i} in {pdf.path}")
                        elements_skipped += 1
                        continue

                    element_text = element.text
                    # Skip elements with no text, non-string text, or newlines
                    if (
                        not element_text
                        or not isinstance(element_text, str)
                        or "\n" in element_text
                    ):
                        if "\n" in str(element_text):
                            reason = "contains newline"
                        elif not element_text:
                            reason = "empty text"
                        else:
                            reason = "invalid text type"
                        logger.debug(
                            f"Skipping element {i} in {pdf.path} page {getattr(element.page, 'number', 'N/A')} because {reason}."
                        )
                        elements_skipped += 1
                        continue

                    # Use page index if available, otherwise fallback or skip? Fallback to 0 for now.
                    page_index = getattr(element.page, "index", 0)
                    image_filename = f"{pdf_hash}_p{page_index}_e{i}.png"
                    relative_image_path = os.path.join("images", image_filename)
                    absolute_image_path = os.path.join(output_dir, relative_image_path)

                    try:
                        # Expand region, render, and save image
                        region = element.expand(self.padding)
                        img = region.to_image(
                            resolution=self.resolution, crop_only=True, include_highlights=False
                        )
                        img.save(absolute_image_path, "PNG")

                        # Add to labels and character set
                        labels.append(
                            (relative_image_path.replace(os.path.sep, "/"), element_text)
                        )  # Use forward slashes for labels
                        char_set.update(element_text)
                        elements_processed += 1

                    except Exception as e:
                        page_num_str = getattr(
                            element.page, "number", "N/A"
                        )  # Get page number safely
                        logger.error(
                            f"Failed to process/save image for element {i} in {pdf.path} page {page_num_str}: {e}",
                            exc_info=False,  # Keep log cleaner
                        )
                        elements_skipped += 1

            except Exception as e:
                logger.error(f"Failed to process PDF {pdf.path}: {e}", exc_info=True)
                # Continue with other PDFs if possible

        if elements_processed == 0:
            logger.error(
                f"No text elements were successfully processed and exported matching '{self.selector}'. Aborting."
            )
            # Clean up potentially created directories? Or leave them empty? Let's leave them.
            return

        logger.info(f"Processed {elements_processed} text elements, skipped {elements_skipped}.")

        # --- 3. Generate Dictionary File (`dict.txt`) ---
        dict_path = os.path.join(output_dir, "dict.txt")
        try:
            # Log the character set before sorting/writing
            logger.debug(f"Exporter final char_set before sorting: {repr(char_set)}")
            # PaddleOCR typically doesn't require special tokens like <UNK> or <BLK> in the dict
            # for recognition models, but this might depend on the specific base model.
            # Start with just the characters found.
            sorted_chars = sorted(list(char_set), reverse=True)
            with open(dict_path, "w", encoding="utf-8") as f_dict:
                for char in sorted_chars:
                    # Ensure we don't write empty strings or just newlines as dictionary entries
                    if char and char != "\n":
                        f_dict.write(char + "\n")
            logger.info(f"Created dictionary file with {len(sorted_chars)} characters: {dict_path}")
        except Exception as e:
            logger.error(f"Failed to write dictionary file '{dict_path}': {e}", exc_info=True)
            raise  # Re-raise as this is critical

        # --- 4. Generate Label Files (`train.txt`, `val.txt` or `label.txt`) ---
        if self.split_ratio is not None and 0 < self.split_ratio < 1:
            if self.random_seed is not None:
                random.seed(self.random_seed)
            random.shuffle(labels)
            split_index = int(len(labels) * self.split_ratio)
            train_labels = labels[:split_index]
            val_labels = labels[split_index:]

            try:
                train_path = os.path.join(output_dir, "train.txt")
                with open(train_path, "w", encoding="utf-8") as f_train:
                    for img_path, text in train_labels:
                        f_train.write(f"{img_path}\t{text}\n")  # Use literal tabs and newlines
                logger.info(
                    f"Created training label file with {len(train_labels)} entries: {train_path}"
                )

                val_path = os.path.join(output_dir, "val.txt")
                with open(val_path, "w", encoding="utf-8") as f_val:
                    for img_path, text in val_labels:
                        f_val.write(f"{img_path}\t{text}\n")  # Use literal tabs and newlines
                logger.info(
                    f"Created validation label file with {len(val_labels)} entries: {val_path}"
                )
            except Exception as e:
                logger.error(f"Failed to write train/validation label files: {e}", exc_info=True)
                raise
        else:
            # Create a single label file
            label_path = os.path.join(output_dir, "label.txt")
            try:
                with open(label_path, "w", encoding="utf-8") as f_label:
                    for img_path, text in labels:
                        f_label.write(f"{img_path}\t{text}\n")  # Use literal tabs and newlines
                logger.info(f"Created single label file with {len(labels)} entries: {label_path}")
            except Exception as e:
                logger.error(f"Failed to write label file '{label_path}': {e}", exc_info=True)
                raise

        # --- 5. Include Guide Notebook ---
        if self.include_guide:
            self._copy_guide_notebook(output_dir)

        logger.info(f"PaddleOCR recognition data export completed successfully to '{output_dir}'.")

    def _copy_guide_notebook(self, output_dir: str):
        """Locates, converts (md->ipynb), and copies the guide notebook."""
        try:
            # Try importing conversion library
            import jupytext
            from nbformat import write as write_notebook
        except ImportError:
            logger.warning(
                "Could not import 'jupytext' or 'nbformat'. Skipping guide notebook generation. "
                "Install with 'pip install natural-pdf[dev]' or 'pip install jupytext nbformat'."
            )
            return

        try:
            # Locate the template .md file relative to this script
            exporter_dir = os.path.dirname(os.path.abspath(__file__))
            # Go up two levels (exporters -> natural_pdf) then down to templates/finetune
            template_dir = os.path.abspath(
                os.path.join(exporter_dir, "..", "templates", "finetune")
            )
            template_md_path = os.path.join(template_dir, "fine_tune_paddleocr.md")
            output_ipynb_path = os.path.join(output_dir, "fine_tune_paddleocr.ipynb")

            if not os.path.exists(template_md_path):
                logger.error(
                    f"Guide template not found at expected location: {template_md_path}. Trying alternate path."
                )
                # Try path relative to workspace root as fallback if run from project root
                alt_template_path = os.path.abspath(
                    os.path.join("natural_pdf", "templates", "finetune", "fine_tune_paddleocr.md")
                )
                if os.path.exists(alt_template_path):
                    template_md_path = alt_template_path
                    logger.info(f"Found guide template at alternate path: {template_md_path}")
                else:
                    logger.error(
                        f"Guide template also not found at: {alt_template_path}. Cannot copy guide."
                    )
                    return

            # Convert Markdown to Notebook object using jupytext
            logger.debug(f"Reading guide template from: {template_md_path}")
            notebook = jupytext.read(template_md_path)  # Reads md and returns NotebookNode

            # Write the Notebook object to the output .ipynb file
            logger.debug(f"Writing guide notebook to: {output_ipynb_path}")
            with open(output_ipynb_path, "w", encoding="utf-8") as f_nb:
                write_notebook(notebook, f_nb)

            logger.info(f"Copied and converted fine-tuning guide notebook to: {output_ipynb_path}")

        except Exception as e:
            logger.error(f"Failed to copy/convert guide notebook: {e}", exc_info=True)
