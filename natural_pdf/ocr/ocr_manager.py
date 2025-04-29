# ocr_manager.py
import copy  # For deep copying options
import logging
from typing import Any, Dict, List, Optional, Type, Union
<<<<<<< HEAD
import threading # Import threading for lock
import time # Import time for timing
=======
>>>>>>> ea72b84d (A hundred updates, a thousand updates)

from PIL import Image

# Import engine classes and options
from .engine import OCREngine
from .engine_easyocr import EasyOCREngine
from .engine_paddle import PaddleOCREngine
<<<<<<< HEAD
from .engine_surya import SuryaOCREngine
from .ocr_options import OCROptions
=======
from .engine_surya import SuryaOCREngine  # <-- Import Surya Engine
from .ocr_options import OCROptions  # <-- Import Surya Options
>>>>>>> ea72b84d (A hundred updates, a thousand updates)
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

<<<<<<< HEAD
    def __init__(self):
        """Initializes the OCR Manager."""
        self._engine_instances: Dict[str, OCREngine] = {}  # Cache for engine instances
        self._engine_locks: Dict[str, threading.Lock] = {} # Lock per engine type for initialization
        self._engine_inference_locks: Dict[str, threading.Lock] = {} # Lock per engine type for inference
=======
    # Define the limited set of kwargs allowed for the simple apply_ocr call
    SIMPLE_MODE_ALLOWED_KWARGS = {
        "engine",
        "languages",
        "min_confidence",
        "device",
        # Add image pre-processing args like 'resolution', 'width' if handled here
    }

    def __init__(self):
        """Initializes the OCR Manager."""
        self._engine_instances: Dict[str, OCREngine] = {}  # Cache for engine instances
>>>>>>> ea72b84d (A hundred updates, a thousand updates)
        logger.info("OCRManager initialized.")

    def _get_engine_instance(self, engine_name: str) -> OCREngine:
        """Retrieves or creates an instance of the specified OCR engine, ensuring thread-safe initialization."""
        engine_name = engine_name.lower()
        if engine_name not in self.ENGINE_REGISTRY:
            raise ValueError(
                f"Unknown OCR engine: '{engine_name}'. Available: {list(self.ENGINE_REGISTRY.keys())}"
            )

<<<<<<< HEAD
        # Quick check if instance already exists (avoid lock contention)
        if engine_name in self._engine_instances:
            return self._engine_instances[engine_name]
=======
        # Surya engine might manage its own predictor state, consider if caching instance is always right
        # For now, we cache the engine instance itself.
        if engine_name not in self._engine_instances:
            logger.info(f"Creating instance of engine: {engine_name}")
            engine_class = self.ENGINE_REGISTRY[engine_name]["class"]
            engine_instance = engine_class()  # Instantiate first
            if not engine_instance.is_available():
                # Check availability before storing
                # Construct helpful error message with install hint
                install_hint = f"pip install 'natural-pdf[{engine_name}]'"
                # Handle potential special cases if extra name differs from engine name (none currently)
                # if engine_name == 'some_engine': install_hint = "pip install 'natural-pdf[some_extra]'"
                raise RuntimeError(
                    f"Engine '{engine_name}' is not available. Please install the required dependencies: {install_hint}"
                )
            self._engine_instances[engine_name] = engine_instance  # Store if available
>>>>>>> ea72b84d (A hundred updates, a thousand updates)

        # Get or create the lock for this engine type
        if engine_name not in self._engine_locks:
            self._engine_locks[engine_name] = threading.Lock()
        
        engine_init_lock = self._engine_locks[engine_name]

        # Acquire lock to safely check and potentially initialize the engine
        with engine_init_lock:
            # Double-check if another thread initialized it while we waited for the lock
            if engine_name in self._engine_instances:
                return self._engine_instances[engine_name]

            # If still not initialized, create it now under the lock
            logger.info(f"[{threading.current_thread().name}] Creating shared instance of engine: {engine_name}")
            engine_class = self.ENGINE_REGISTRY[engine_name]["class"]
            start_time = time.monotonic() # Optional: time initialization
            try:
                engine_instance = engine_class()  # Instantiate first
                if not engine_instance.is_available():
                    # Check availability before storing
                    install_hint = f"pip install 'natural-pdf[{engine_name}]'"
                    raise RuntimeError(
                        f"Engine '{engine_name}' is not available. Please install the required dependencies: {install_hint}"
                    )
                # Store the shared instance
                self._engine_instances[engine_name] = engine_instance
                end_time = time.monotonic()
                logger.info(f"[{threading.current_thread().name}] Shared instance of {engine_name} created successfully (Duration: {end_time - start_time:.2f}s).")
                return engine_instance
            except Exception as e:
                 # Ensure we don't leave a partial state if init fails
                 logger.error(f"[{threading.current_thread().name}] Failed to create shared instance of {engine_name}: {e}", exc_info=True)
                 # Remove potentially partial entry if exists
                 if engine_name in self._engine_instances: del self._engine_instances[engine_name]
                 raise # Re-raise the exception after logging

    def _get_engine_inference_lock(self, engine_name: str) -> threading.Lock:
        """Gets or creates the inference lock for a given engine type."""
        engine_name = engine_name.lower()
        # Assume engine_name is valid as it's checked before this would be called
        if engine_name not in self._engine_inference_locks:
            # Create lock if it doesn't exist (basic thread safety for dict access)
            # A more robust approach might lock around this check/creation too,
            # but contention here is less critical than for engine init or inference itself.
            self._engine_inference_locks[engine_name] = threading.Lock()
        return self._engine_inference_locks[engine_name]

    def apply_ocr(
        self,
<<<<<<< HEAD
        images: Union[Image.Image, List[Image.Image]],
        # --- Explicit Common Parameters ---
        engine: Optional[str] = None,
        languages: Optional[List[str]] = None,
        min_confidence: Optional[float] = None,
        device: Optional[str] = None,
        detect_only: bool = False,
        # --- Engine-Specific Options ---
        options: Optional[Any] = None,  # e.g. EasyOCROptions(), PaddleOCROptions()
    ) -> Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]]:
=======
        images: Union[Image.Image, List[Image.Image]],  # Accept single or list
        engine: Optional[str] = "easyocr",  # Default engine
        options: Optional[OCROptions] = None,
        **kwargs,
    ) -> Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]]:  # Return single or list of lists
>>>>>>> ea72b84d (A hundred updates, a thousand updates)
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
<<<<<<< HEAD

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
            options_class = self.ENGINE_REGISTRY[selected_engine_name].get(
                "options_class", BaseOCROptions
            )
            if not isinstance(final_options, options_class):
                # Allow dicts to be passed directly too, assuming engine handles them
                if not isinstance(final_options, dict):
                    raise TypeError(
                        f"Provided options type '{type(final_options).__name__}' is not compatible with engine '{selected_engine_name}'. Expected '{options_class.__name__}' or dict."
                    )

=======
        # Allow engines to handle non-PIL images in list if they support it/log warnings
        # if is_batch and not all(isinstance(img, Image.Image) for img in images):
        #     logger.warning("Batch may contain items that are not PIL Images.")

        # --- Determine Options and Engine ---
        if options is not None:
            # Advanced Mode
            logger.debug(f"Using advanced mode with options object: {type(options).__name__}")
            final_options = copy.deepcopy(options)  # Prevent modification of original
            found_engine = False
            for name, registry_entry in self.ENGINE_REGISTRY.items():
                # Check if options object is an instance of the registered options class
                if isinstance(options, registry_entry["options_class"]):
                    selected_engine_name = name
                    found_engine = True
                    break
            if not found_engine:
                raise TypeError(
                    f"Provided options object type '{type(options).__name__}' does not match any registered engine options."
                )
            if kwargs:
                logger.warning(
                    f"Keyword arguments {list(kwargs.keys())} were provided alongside 'options' and will be ignored."
                )
        else:
            # Simple Mode
            selected_engine_name = engine.lower() if engine else "easyocr"  # Fallback default
            logger.debug(
                f"Using simple mode with engine: '{selected_engine_name}' and kwargs: {kwargs}"
            )

            if selected_engine_name not in self.ENGINE_REGISTRY:
                raise ValueError(
                    f"Unknown OCR engine: '{selected_engine_name}'. Available: {list(self.ENGINE_REGISTRY.keys())}"
                )

            unexpected_kwargs = set(kwargs.keys()) - self.SIMPLE_MODE_ALLOWED_KWARGS
            if unexpected_kwargs:
                raise TypeError(
                    f"Got unexpected keyword arguments in simple mode: {list(unexpected_kwargs)}. Use the 'options' parameter for detailed configuration."
                )

            # Get the *correct* options class for the selected engine
            options_class = self.ENGINE_REGISTRY[selected_engine_name]["options_class"]

            # Create options instance using provided simple kwargs or defaults
            simple_args = {
                "languages": kwargs.get("languages", ["en"]),
                "min_confidence": kwargs.get("min_confidence", 0.5),
                "device": kwargs.get("device", "cpu"),
                # Note: 'extra_args' isn't populated in simple mode
            }
            final_options = options_class(**simple_args)
            logger.debug(f"Constructed options for simple mode: {final_options}")

>>>>>>> ea72b84d (A hundred updates, a thousand updates)
        # --- Get Engine Instance and Process ---
        try:
            engine_instance = self._get_engine_instance(selected_engine_name)
            processing_mode = "batch" if is_batch else "single image"
            # Log thread name for clarity during parallel calls
            thread_id = threading.current_thread().name
            logger.info(f"[{thread_id}] Processing {processing_mode} using shared engine instance '{selected_engine_name}'...")
            logger.debug(
                f"  Engine Args: languages={languages}, min_confidence={min_confidence}, device={device}, options={final_options}"
            )

            # Log image dimensions before processing
            if is_batch:
                image_dims = [f"{img.width}x{img.height}" for img in images if hasattr(img, 'width') and hasattr(img, 'height')]
                logger.debug(f"[{thread_id}] Processing batch of {len(images)} images with dimensions: {image_dims}")
            elif hasattr(images, 'width') and hasattr(images, 'height'):
                logger.debug(f"[{thread_id}] Processing single image with dimensions: {images.width}x{images.height}")
            else:
                logger.warning(f"[{thread_id}] Could not determine dimensions of input image(s).")

            # Acquire lock specifically for the inference call
            inference_lock = self._get_engine_inference_lock(selected_engine_name)
            logger.debug(f"[{thread_id}] Attempting to acquire inference lock for {selected_engine_name}...")
            inference_wait_start = time.monotonic()
            with inference_lock:
                inference_acquired_time = time.monotonic()
                logger.debug(f"[{thread_id}] Acquired inference lock for {selected_engine_name} (waited {inference_acquired_time - inference_wait_start:.2f}s). Calling process_image...")
                inference_start_time = time.monotonic()

                results = engine_instance.process_image(
                    images=images,
                    languages=languages,
                    min_confidence=min_confidence,
                    device=device,
                    detect_only=detect_only,
                    options=final_options,
                )
                inference_end_time = time.monotonic()
                logger.debug(f"[{thread_id}] process_image call finished for {selected_engine_name} (Duration: {inference_end_time - inference_start_time:.2f}s). Releasing lock.")

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
