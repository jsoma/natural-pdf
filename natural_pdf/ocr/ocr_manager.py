# ocr_manager.py
import copy  # For deep copying options
import logging
from typing import Any, Dict, List, Optional, Type, Union

from PIL import Image

# Import engine classes and options
from .engine import OCREngine
from .engine_easyocr import EasyOCREngine
from .engine_paddle import PaddleOCREngine
from .engine_surya import SuryaOCREngine
from .ocr_options import OCROptions
from .ocr_options import BaseOCROptions, EasyOCROptions, PaddleOCROptions, SuryaOCROptions

logger = logging.getLogger(__name__)


class OCRManager:
    """Manages OCR engine selection, configuration, and execution."""

    # Registry mapping engine names to classes and default options
    ENGINE_REGISTRY: Dict[str, Dict[str, Any]] = {
        "easyocr": {"class": EasyOCREngine, "options_class": EasyOCROptions},
        "paddle": {"class": PaddleOCREngine, "options_class": PaddleOCROptions},
        "surya": {"class": SuryaOCREngine, "options_class": SuryaOCROptions},  # <-- Add Surya
        # Add other engines here
    }

    def __init__(self):
        """Initializes the OCR Manager."""
        self._engine_instances: Dict[str, OCREngine] = {}  # Cache for engine instances
        logger.info("OCRManager initialized.")

    def _get_engine_instance(self, engine_name: str) -> OCREngine:
        """Retrieves or creates an instance of the specified OCR engine."""
        engine_name = engine_name.lower()
        if engine_name not in self.ENGINE_REGISTRY:
            raise ValueError(
                f"Unknown OCR engine: '{engine_name}'. Available: {list(self.ENGINE_REGISTRY.keys())}"
            )

        if engine_name not in self._engine_instances:
            logger.info(f"Creating instance of engine: {engine_name}")
            engine_class = self.ENGINE_REGISTRY[engine_name]["class"]
            engine_instance = engine_class()  # Instantiate first
            if not engine_instance.is_available():
                # Check availability before storing
                # Construct helpful error message with install hint
                install_hint = f"pip install 'natural-pdf[{engine_name}]'"
                raise RuntimeError(
                    f"Engine '{engine_name}' is not available. Please install the required dependencies: {install_hint}"
                )
            self._engine_instances[engine_name] = engine_instance  # Store if available

        return self._engine_instances[engine_name]

    def apply_ocr(
        self,
        images: Union[Image.Image, List[Image.Image]],
        # --- Explicit Common Parameters ---
        engine: Optional[str] = None,
        languages: Optional[List[str]] = None,
        min_confidence: Optional[float] = None,
        device: Optional[str] = None,
        detect_only: bool = False,
        # --- Engine-Specific Options ---
        options: Optional[Any] = None, # e.g. EasyOCROptions(), PaddleOCROptions()
    ) -> Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]]:
        """
        Applies OCR to a single image or a batch of images.

        Args:
            images: A single PIL Image or a list of PIL Images to process.
            engine: Name of the engine (e.g., 'easyocr', 'paddle', 'surya').
                    Defaults to 'easyocr' if not specified.
            languages: List of language codes (e.g., ['en', 'fr'], ['en', 'german']).
                       **Passed directly to the engine.** Must be codes understood
                       by the specific engine. No mapping is performed by the manager.
            min_confidence: Minimum confidence threshold (0.0-1.0).
                            Passed directly to the engine.
            device: Device string (e.g., 'cpu', 'cuda').
                    Passed directly to the engine.
            detect_only: If True, only detect text regions, do not perform OCR.
            options: An engine-specific options object (e.g., EasyOCROptions) or dict
                     containing additional parameters specific to the chosen engine.
                     Passed directly to the engine.

        Returns:
            If input is a single image: List of result dictionaries.
            If input is a list of images: List of lists of result dictionaries.

        Raises:
            ValueError: If the engine name is invalid.
            TypeError: If input 'images' is not valid or options type is incompatible.
            RuntimeError: If the selected engine is not available or processing fails.
        """
        # --- Validate input type ---
        is_batch = isinstance(images, list)
        if not is_batch and not isinstance(images, Image.Image):
            raise TypeError("Input 'images' must be a PIL Image or a list of PIL Images.")

        # --- Determine Engine --- 
        selected_engine_name = (engine or "easyocr").lower()
        if selected_engine_name not in self.ENGINE_REGISTRY:
            raise ValueError(
                f"Unknown OCR engine: '{selected_engine_name}'. Available: {list(self.ENGINE_REGISTRY.keys())}"
            )
        logger.debug(f"Selected engine: '{selected_engine_name}'")

        # --- Prepare Options --- 
        final_options = copy.deepcopy(options) if options is not None else None
        
        # Type check options object if provided
        if final_options is not None:
            options_class = self.ENGINE_REGISTRY[selected_engine_name].get("options_class", BaseOCROptions)
            if not isinstance(final_options, options_class):
                 # Allow dicts to be passed directly too, assuming engine handles them
                if not isinstance(final_options, dict):
                     raise TypeError(
                        f"Provided options type '{type(final_options).__name__}' is not compatible with engine '{selected_engine_name}'. Expected '{options_class.__name__}' or dict."
                    )

        # --- Get Engine Instance and Process --- 
        try:
            engine_instance = self._get_engine_instance(selected_engine_name)
            processing_mode = "batch" if is_batch else "single image"
            logger.info(f"Processing {processing_mode} with engine '{selected_engine_name}'...")
            logger.debug(f"  Engine Args: languages={languages}, min_confidence={min_confidence}, device={device}, options={final_options}")

            # Call the engine's process_image, passing common args and options object
            # **ASSUMPTION**: Engine process_image signatures are updated to accept these common args.
            results = engine_instance.process_image(
                images=images, 
                languages=languages,
                min_confidence=min_confidence,
                device=device,
                detect_only=detect_only,
                options=final_options
            )

            # Log result summary based on mode
            if is_batch:
                # Ensure results is a list before trying to get lengths
                if isinstance(results, list):
                    num_results_per_image = [
                        len(res_list) if isinstance(res_list, list) else -1 for res_list in results
                    ]  # Handle potential errors returning non-lists
                    logger.info(
                        f"Processing complete. Found results per image: {num_results_per_image}"
                    )
                else:
                    logger.error(
                        f"Processing complete but received unexpected result type for batch: {type(results)}"
                    )
            else:
                # Ensure results is a list
                if isinstance(results, list):
                    logger.info(f"Processing complete. Found {len(results)} results.")
                else:
                    logger.error(
                        f"Processing complete but received unexpected result type for single image: {type(results)}"
                    )
            return results  # Return type matches input type due to engine logic

        except (ImportError, RuntimeError, ValueError, TypeError) as e:
            logger.error(
                f"OCR processing failed for engine '{selected_engine_name}': {e}", exc_info=True
            )
            raise  # Re-raise expected errors
        except Exception as e:
            logger.error(f"An unexpected error occurred during OCR processing: {e}", exc_info=True)
            raise  # Re-raise unexpected errors

    def get_available_engines(self) -> List[str]:
        """Returns a list of registered engine names that are currently available."""
        available = []
        for name, registry_entry in self.ENGINE_REGISTRY.items():
            try:
                # Temporarily instantiate to check availability without caching
                engine_class = registry_entry["class"]
                if engine_class().is_available():
                    available.append(name)
            except Exception as e:
                logger.debug(
                    f"Engine '{name}' check failed: {e}"
                )  # Log check failures at debug level
                pass  # Ignore engines that fail to instantiate or check
        return available
