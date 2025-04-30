# natural_pdf/ocr/engine_doctr.py
import importlib.util
import logging
from typing import Any, List, Optional

import numpy as np
from PIL import Image

from .engine import OCREngine, TextRegion
from .ocr_options import BaseOCROptions, DoctrOCROptions

logger = logging.getLogger(__name__)


class DoctrOCREngine(OCREngine):
    """docTR engine implementation."""

    def __init__(self):
        super().__init__()
        self._model = None  # Will hold the doctr ocr_predictor
        self._detection_model = None  # Will hold detection_predictor if detect_only is used
        self._orientation_model = None  # Will hold page_orientation_predictor if enabled

    def is_available(self) -> bool:
        """Check if doctr is installed."""
        return importlib.util.find_spec("doctr") is not None

    def _initialize_model(
        self, languages: List[str], device: str, options: Optional[BaseOCROptions]
    ):
        """Initialize the doctr model."""
        if not self.is_available():
            raise ImportError(
                "Doctr engine requires the 'python-doctr' package. "
                "Install with: pip install python-doctr[torch] or python-doctr[tf]"
            )

        try:
            import doctr.models

            self.logger.info("doctr.models imported successfully.")
        except ImportError as e:
            self.logger.error(f"Failed to import doctr: {e}")
            raise

        # Cast to DoctrOCROptions or use default
        doctr_opts = options if isinstance(options, DoctrOCROptions) else DoctrOCROptions()

        # Check if CUDA is available in device string
        use_cuda = device.lower().startswith("cuda") if device else False

        # Prepare OCR predictor arguments
        predictor_args = {
            "det_arch": doctr_opts.det_arch,
            "reco_arch": doctr_opts.reco_arch,
            "pretrained": doctr_opts.pretrained,
            "assume_straight_pages": doctr_opts.assume_straight_pages,
            "export_as_straight_boxes": doctr_opts.export_as_straight_boxes,
        }
        # Filter out None values
        predictor_args = {k: v for k, v in predictor_args.items() if v is not None}

        self.logger.debug(f"doctr ocr_predictor constructor args: {predictor_args}")
        try:
            # Create the main OCR predictor (doesn't accept batch_size)
            self._model = doctr.models.ocr_predictor(**predictor_args)

            # Apply CUDA if available
            if use_cuda:
                self._model = self._model.cuda()

            self.logger.info("doctr ocr_predictor created successfully")

            # Now initialize the detection-only model
            try:
                detection_args = {
                    "arch": doctr_opts.det_arch,
                    "pretrained": doctr_opts.pretrained,
                    "assume_straight_pages": doctr_opts.assume_straight_pages,
                    "symmetric_pad": doctr_opts.symmetric_pad,
                    "preserve_aspect_ratio": doctr_opts.preserve_aspect_ratio,
                    "batch_size": doctr_opts.batch_size,
                }
                self._detection_model = doctr.models.detection_predictor(**detection_args)

                # Apply CUDA if available
                if use_cuda:
                    self._detection_model = self._detection_model.cuda()

                # Configure postprocessing parameters if provided
                if doctr_opts.bin_thresh is not None:
                    self._detection_model.model.postprocessor.bin_thresh = doctr_opts.bin_thresh
                if doctr_opts.box_thresh is not None:
                    self._detection_model.model.postprocessor.box_thresh = doctr_opts.box_thresh

                self.logger.info("doctr detection_predictor created successfully")
            except Exception as e:
                self.logger.error(f"Failed to create detection_predictor: {e}")
                self._detection_model = None

            # Initialize orientation predictor if enabled
            if doctr_opts.use_orientation_predictor:
                try:
                    self._orientation_model = doctr.models.page_orientation_predictor(
                        pretrained=True, batch_size=doctr_opts.batch_size
                    )
                    if use_cuda:
                        self._orientation_model = self._orientation_model.cuda()
                    self.logger.info("doctr page_orientation_predictor created successfully")
                except Exception as e:
                    self.logger.error(f"Failed to create page_orientation_predictor: {e}")
                    self._orientation_model = None

        except Exception as e:
            self.logger.error(f"Failed to create doctr models: {e}")
            raise

        # Doctr doesn't explicitly use language list in ocr_predictor initialization
        if languages and languages != [self.DEFAULT_LANGUAGES[0]]:
            logger.warning(
                f"Doctr engine currently doesn't support language selection during initialization. Using its default language capabilities for model: {doctr_opts.reco_arch}"
            )

    def _preprocess_image(self, image: Image.Image) -> np.ndarray:
        """Convert PIL Image to RGB numpy array for doctr."""
        # Ensure the image is in RGB mode
        if image.mode != "RGB":
            image = image.convert("RGB")
        # Convert to numpy array
        return np.array(image)

    def _process_single_image(
        self, image: np.ndarray, detect_only: bool, options: Optional[DoctrOCROptions]
    ) -> Any:
        """Process a single image with doctr."""
        if self._model is None:
            raise RuntimeError("Doctr model not initialized")

        # Capture image dimensions for denormalization
        height, width = image.shape[:2]

        # Cast options to DoctrOCROptions or use default
        doctr_opts = options if isinstance(options, DoctrOCROptions) else DoctrOCROptions()

        # Check if we need to detect orientation first
        if self._orientation_model is not None and options and options.use_orientation_predictor:
            try:
                # Process with orientation predictor
                # For orientation predictor, we need to pass a batch of images
                orientations = self._orientation_model([image])
                orientation = orientations[1][0]  # Get the orientation angle
                logger.info(f"Detected page orientation: {orientation} degrees")
                # Note: doctr handles rotation internally for detection/recognition
            except Exception as e:
                logger.error(f"Error detecting orientation: {e}")

        # Process differently based on detect_only flag
        if detect_only and self._detection_model is not None:
            try:
                # Apply threshold settings at runtime for this detection
                if doctr_opts.bin_thresh is not None:
                    original_bin_thresh = self._detection_model.model.postprocessor.bin_thresh
                    self._detection_model.model.postprocessor.bin_thresh = doctr_opts.bin_thresh
                    logger.debug(f"Temporarily set bin_thresh to {doctr_opts.bin_thresh}")

                if doctr_opts.box_thresh is not None:
                    original_box_thresh = self._detection_model.model.postprocessor.box_thresh
                    self._detection_model.model.postprocessor.box_thresh = doctr_opts.box_thresh
                    logger.debug(f"Temporarily set box_thresh to {doctr_opts.box_thresh}")

                # Use the dedicated detection model with a list of numpy arrays
                result = self._detection_model([image])

                # Restore original thresholds
                if doctr_opts.bin_thresh is not None:
                    self._detection_model.model.postprocessor.bin_thresh = original_bin_thresh

                if doctr_opts.box_thresh is not None:
                    self._detection_model.model.postprocessor.box_thresh = original_box_thresh

                # Return tuple of (result, dimensions)
                return (result, (height, width))
            except Exception as e:
                logger.error(f"Error in detection_predictor: {e}")
                # Fall back to OCR predictor if detection fails
                logger.warning("Falling back to OCR predictor for detection")

        # Process with full OCR model, passing a list of numpy arrays directly
        try:
            # For full OCR, we should also apply the thresholds
            if (
                detect_only
                and doctr_opts.bin_thresh is not None
                and hasattr(self._model.det_predictor.model.postprocessor, "bin_thresh")
            ):
                original_bin_thresh = self._model.det_predictor.model.postprocessor.bin_thresh
                self._model.det_predictor.model.postprocessor.bin_thresh = doctr_opts.bin_thresh

            if (
                detect_only
                and doctr_opts.box_thresh is not None
                and hasattr(self._model.det_predictor.model.postprocessor, "box_thresh")
            ):
                original_box_thresh = self._model.det_predictor.model.postprocessor.box_thresh
                self._model.det_predictor.model.postprocessor.box_thresh = doctr_opts.box_thresh

            result = self._model([image])

            # Restore original thresholds
            if (
                detect_only
                and doctr_opts.bin_thresh is not None
                and hasattr(self._model.det_predictor.model.postprocessor, "bin_thresh")
            ):
                self._model.det_predictor.model.postprocessor.bin_thresh = original_bin_thresh

            if (
                detect_only
                and doctr_opts.box_thresh is not None
                and hasattr(self._model.det_predictor.model.postprocessor, "box_thresh")
            ):
                self._model.det_predictor.model.postprocessor.box_thresh = original_box_thresh

            # Return tuple of (result, dimensions)
            return (result, (height, width))
        except Exception as e:
            logger.error(f"Error in OCR prediction: {e}")
            raise

    def _standardize_results(
        self, raw_results: Any, min_confidence: float, detect_only: bool
    ) -> List[TextRegion]:
        """Convert doctr results to standardized TextRegion objects."""
        standardized_regions = []

        # Extract results and dimensions
        if isinstance(raw_results, tuple) and len(raw_results) == 2:
            results, dimensions = raw_results
            image_height, image_width = dimensions
        else:
            # Fallback if dimensions aren't provided
            results = raw_results
            image_width = 1
            image_height = 1
            logger.warning("Image dimensions not provided, using normalized coordinates")

        # Handle detection-only results differently
        if detect_only and self._detection_model is not None and not hasattr(results, "pages"):
            # Import doctr utils for detach_scores if needed
            try:
                from doctr.utils.geometry import detach_scores
            except ImportError:
                logger.error("Failed to import doctr.utils.geometry.detach_scores")
                return standardized_regions

            # Extract coordinates and scores from detection results
            for result in results:
                # Detection results structure is different from ocr_predictor
                if "words" in result:
                    try:
                        # Detach the coordinates and scores
                        detached_coords, prob_scores = detach_scores([result.get("words")])

                        for i, coords in enumerate(detached_coords[0]):
                            score = (
                                prob_scores[0][i]
                                if prob_scores and len(prob_scores[0]) > i
                                else 0.0
                            )

                            if score >= min_confidence:
                                try:
                                    # Handle both straight and rotated boxes
                                    if coords.shape == (
                                        4,
                                    ):  # Straight box as [xmin, ymin, xmax, ymax]
                                        xmin, ymin, xmax, ymax = coords.tolist()
                                        # Denormalize coordinates
                                        bbox = (
                                            float(xmin * image_width),
                                            float(ymin * image_height),
                                            float(xmax * image_width),
                                            float(ymax * image_height),
                                        )
                                    else:  # Polygon points
                                        # Get bounding box from polygon
                                        coords_list = coords.tolist()
                                        x_coords = [p[0] * image_width for p in coords_list]
                                        y_coords = [p[1] * image_height for p in coords_list]
                                        bbox = (
                                            float(min(x_coords)),
                                            float(min(y_coords)),
                                            float(max(x_coords)),
                                            float(max(y_coords)),
                                        )

                                    # In detection mode, we don't have text or confidence score
                                    standardized_regions.append(TextRegion(bbox, None, score))
                                except Exception as e:
                                    logger.error(f"Error processing detection result: {e}")
                    except Exception as e:
                        logger.error(f"Error detaching scores: {e}")

            return standardized_regions

        # Process standard OCR results
        if not hasattr(results, "pages") or not results.pages:
            logger.warning("Doctr result object does not contain pages.")
            return standardized_regions

        # Process results page by page (we typically process one image at a time)
        for page in results.pages:
            # Extract information from blocks, lines, words
            for block in page.blocks:
                for line in block.lines:
                    for word in line.words:
                        if word.confidence >= min_confidence:
                            try:
                                # doctr geometry is ((x_min, y_min), (x_max, y_max)) as relative coordinates
                                x_min, y_min = word.geometry[0]
                                x_max, y_max = word.geometry[1]

                                # Denormalize coordinates to absolute pixel values
                                bbox = (
                                    float(x_min * image_width),
                                    float(y_min * image_height),
                                    float(x_max * image_width),
                                    float(y_max * image_height),
                                )

                                # Skip text content if detect_only is True
                                text = None if detect_only else word.value
                                confidence = None if detect_only else word.confidence

                                standardized_regions.append(TextRegion(bbox, text, confidence))
                            except (ValueError, TypeError, IndexError) as e:
                                logger.error(
                                    f"Could not standardize bounding box/word from doctr result: {word}"
                                )
                                logger.error(f"Error: {e}")

        return standardized_regions

    def get_default_options(self) -> DoctrOCROptions:
        """Return the default options specific to this engine."""
        return DoctrOCROptions()
