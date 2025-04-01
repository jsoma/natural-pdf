# ocr_manager.py
import logging
from typing import Dict, List, Any, Optional, Union, Type
from PIL import Image
import copy # For deep copying options

# Import engine classes and options
from .engine import OCREngine
from .engine_easyocr import EasyOCREngine
from .engine_paddle import PaddleOCREngine
from .engine_surya import SuryaOCREngine # <-- Import Surya Engine
from .ocr_options import (
    BaseOCROptions, EasyOCROptions, PaddleOCROptions, SuryaOCROptions, OCROptions # <-- Import Surya Options
)

logger = logging.getLogger(__name__)

class OCRManager:
    """Manages OCR engine selection, configuration, and execution."""

    # Registry mapping engine names to classes and default options
    ENGINE_REGISTRY: Dict[str, Dict[str, Any]] = {
        'easyocr': {'class': EasyOCREngine, 'options_class': EasyOCROptions},
        'paddle': {'class': PaddleOCREngine, 'options_class': PaddleOCROptions},
        'surya': {'class': SuryaOCREngine, 'options_class': SuryaOCROptions}, # <-- Add Surya
        # Add other engines here
    }

    # Define the limited set of kwargs allowed for the simple apply_ocr call
    SIMPLE_MODE_ALLOWED_KWARGS = {
        'engine', 'languages', 'min_confidence', 'device'
        # Add image pre-processing args like 'resolution', 'width' if handled here
    }

    def __init__(self):
        """Initializes the OCR Manager."""
        self._engine_instances: Dict[str, OCREngine] = {} # Cache for engine instances
        logger.info("OCRManager initialized.")

    def _get_engine_instance(self, engine_name: str) -> OCREngine:
        """Retrieves or creates an instance of the specified OCR engine."""
        engine_name = engine_name.lower()
        if engine_name not in self.ENGINE_REGISTRY:
            raise ValueError(f"Unknown OCR engine: '{engine_name}'. Available: {list(self.ENGINE_REGISTRY.keys())}")

        # Surya engine might manage its own predictor state, consider if caching instance is always right
        # For now, we cache the engine instance itself.
        if engine_name not in self._engine_instances:
            logger.info(f"Creating instance of engine: {engine_name}")
            engine_class = self.ENGINE_REGISTRY[engine_name]['class']
            engine_instance = engine_class() # Instantiate first
            if not engine_instance.is_available():
                 # Check availability before storing
                 raise RuntimeError(f"Engine '{engine_name}' is not available. Please check dependencies.")
            self._engine_instances[engine_name] = engine_instance # Store if available

        return self._engine_instances[engine_name]

    def apply_ocr(
        self,
        images: Union[Image.Image, List[Image.Image]], # Accept single or list
        engine: Optional[str] = 'easyocr', # Default engine
        options: Optional[OCROptions] = None,
        **kwargs
    ) -> Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]]: # Return single or list of lists
        """
        Applies OCR to a single image or a batch of images using either simple
        keyword arguments or an options object.

        Args:
            images: A single PIL Image or a list of PIL Images to process.
            engine: Name of the engine to use (e.g., 'easyocr', 'paddle', 'surya').
                    Ignored if 'options' object is provided. Defaults to 'easyocr'.
            options: An instance of EasyOCROptions, PaddleOCROptions, or SuryaOCROptions
                     for detailed configuration. If provided, simple kwargs (languages, etc.)
                     and the 'engine' arg are ignored.
            **kwargs: For simple mode, accepts: 'languages', 'min_confidence', 'device'.
                      Other kwargs will raise a TypeError unless 'options' is provided.

        Returns:
            If input is a single image: List of result dictionaries.
            If input is a list of images: List of lists of result dictionaries,
                                          corresponding to each input image.

        Raises:
            ValueError: If the engine name is invalid.
            TypeError: If unexpected keyword arguments are provided in simple mode,
                       or if input 'images' is not a PIL Image or list of PIL Images.
            RuntimeError: If the selected engine is not available.
        """
        final_options: BaseOCROptions
        selected_engine_name: str

        # --- Validate input type ---
        is_batch = isinstance(images, list)
        if not is_batch and not isinstance(images, Image.Image):
             raise TypeError("Input 'images' must be a PIL Image or a list of PIL Images.")
        # Allow engines to handle non-PIL images in list if they support it/log warnings
        # if is_batch and not all(isinstance(img, Image.Image) for img in images):
        #     logger.warning("Batch may contain items that are not PIL Images.")


        # --- Determine Options and Engine ---
        if options is not None:
            # Advanced Mode
            logger.debug(f"Using advanced mode with options object: {type(options).__name__}")
            final_options = copy.deepcopy(options) # Prevent modification of original
            found_engine = False
            for name, registry_entry in self.ENGINE_REGISTRY.items():
                # Check if options object is an instance of the registered options class
                if isinstance(options, registry_entry['options_class']):
                    selected_engine_name = name
                    found_engine = True
                    break
            if not found_engine:
                 raise TypeError(f"Provided options object type '{type(options).__name__}' does not match any registered engine options.")
            if kwargs:
                logger.warning(f"Keyword arguments {list(kwargs.keys())} were provided alongside 'options' and will be ignored.")
        else:
            # Simple Mode
            selected_engine_name = engine.lower() if engine else 'easyocr' # Fallback default
            logger.debug(f"Using simple mode with engine: '{selected_engine_name}' and kwargs: {kwargs}")

            if selected_engine_name not in self.ENGINE_REGISTRY:
                 raise ValueError(f"Unknown OCR engine: '{selected_engine_name}'. Available: {list(self.ENGINE_REGISTRY.keys())}")

            unexpected_kwargs = set(kwargs.keys()) - self.SIMPLE_MODE_ALLOWED_KWARGS
            if unexpected_kwargs:
                raise TypeError(f"Got unexpected keyword arguments in simple mode: {list(unexpected_kwargs)}. Use the 'options' parameter for detailed configuration.")

            # Get the *correct* options class for the selected engine
            options_class = self.ENGINE_REGISTRY[selected_engine_name]['options_class']

            # Create options instance using provided simple kwargs or defaults
            simple_args = {
                'languages': kwargs.get('languages', ['en']),
                'min_confidence': kwargs.get('min_confidence', 0.5),
                'device': kwargs.get('device', 'cpu')
                # Note: 'extra_args' isn't populated in simple mode
            }
            final_options = options_class(**simple_args)
            logger.debug(f"Constructed options for simple mode: {final_options}")


        # --- Get Engine Instance and Process ---
        try:
            engine_instance = self._get_engine_instance(selected_engine_name)
            processing_mode = "batch" if is_batch else "single image"
            logger.info(f"Processing {processing_mode} with engine '{selected_engine_name}'...")

            # Call the engine's process_image, passing single image or list
            results = engine_instance.process_image(images, final_options)

            # Log result summary based on mode
            if is_batch:
                 # Ensure results is a list before trying to get lengths
                 if isinstance(results, list):
                     num_results_per_image = [len(res_list) if isinstance(res_list, list) else -1 for res_list in results] # Handle potential errors returning non-lists
                     logger.info(f"Processing complete. Found results per image: {num_results_per_image}")
                 else:
                     logger.error(f"Processing complete but received unexpected result type for batch: {type(results)}")
            else:
                 # Ensure results is a list
                 if isinstance(results, list):
                      logger.info(f"Processing complete. Found {len(results)} results.")
                 else:
                      logger.error(f"Processing complete but received unexpected result type for single image: {type(results)}")
            return results # Return type matches input type due to engine logic

        except (ImportError, RuntimeError, ValueError, TypeError) as e:
             logger.error(f"OCR processing failed for engine '{selected_engine_name}': {e}", exc_info=True)
             raise # Re-raise expected errors
        except Exception as e:
             logger.error(f"An unexpected error occurred during OCR processing: {e}", exc_info=True)
             raise # Re-raise unexpected errors


    def get_available_engines(self) -> List[str]:
        """Returns a list of registered engine names that are currently available."""
        available = []
        for name, registry_entry in self.ENGINE_REGISTRY.items():
             try:
                 # Temporarily instantiate to check availability without caching
                 engine_class = registry_entry['class']
                 if engine_class().is_available():
                     available.append(name)
             except Exception as e:
                 logger.debug(f"Engine '{name}' check failed: {e}") # Log check failures at debug level
                 pass # Ignore engines that fail to instantiate or check
        return available

