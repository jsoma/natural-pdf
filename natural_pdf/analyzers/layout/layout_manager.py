# layout_manager.py
import logging
from typing import Dict, List, Any, Optional, Union, Type
from PIL import Image
import copy

# --- Import detector classes and options ---
# Use try-except blocks for robustness if some detectors might be missing dependencies
try:
    from .base import LayoutDetector
except ImportError:
    LayoutDetector = type('LayoutDetector', (), {})

try:
    from .yolo import YOLODocLayoutDetector
except ImportError:
    YOLODocLayoutDetector = None

try:
    from .tatr import TableTransformerDetector
except ImportError:
    TableTransformerDetector = None

try:
    from .paddle import PaddleLayoutDetector
except ImportError:
    PaddleLayoutDetector = None

try:
    from .surya import SuryaLayoutDetector
except ImportError:
    SuryaLayoutDetector = None

try:
    from .docling import DoclingLayoutDetector
except ImportError:
    DoclingLayoutDetector = None
    
from .layout_options import (
    BaseLayoutOptions, YOLOLayoutOptions, TATRLayoutOptions,
    PaddleLayoutOptions, SuryaLayoutOptions, DoclingLayoutOptions, LayoutOptions
)

logger = logging.getLogger(__name__)

class LayoutManager:
    """Manages layout detector selection, configuration, and execution."""

    # Registry mapping engine names to classes and default options
    ENGINE_REGISTRY: Dict[str, Dict[str, Any]] = {}

    # Populate registry only with available detectors
    if YOLODocLayoutDetector: ENGINE_REGISTRY['yolo'] = {'class': YOLODocLayoutDetector, 'options_class': YOLOLayoutOptions}
    if TableTransformerDetector: ENGINE_REGISTRY['tatr'] = {'class': TableTransformerDetector, 'options_class': TATRLayoutOptions}
    if PaddleLayoutDetector: ENGINE_REGISTRY['paddle'] = {'class': PaddleLayoutDetector, 'options_class': PaddleLayoutOptions}
    if SuryaLayoutDetector: ENGINE_REGISTRY['surya'] = {'class': SuryaLayoutDetector, 'options_class': SuryaLayoutOptions}
    if DoclingLayoutDetector: ENGINE_REGISTRY['docling'] = {'class': DoclingLayoutDetector, 'options_class': DoclingLayoutOptions}

    # Define the limited set of kwargs allowed for the simple analyze_layout call
    SIMPLE_MODE_ALLOWED_KWARGS = {
        'engine', 'confidence', 'classes', 'exclude_classes', 'device'
    }

    def __init__(self):
        """Initializes the Layout Manager."""
        # Cache for detector instances (different from model cache inside detector)
        self._detector_instances: Dict[str, LayoutDetector] = {}
        logger.info(f"LayoutManager initialized. Available engines: {list(self.ENGINE_REGISTRY.keys())}")

    def _get_engine_instance(self, engine_name: str) -> LayoutDetector:
        """Retrieves or creates an instance of the specified layout detector."""
        engine_name = engine_name.lower()
        if engine_name not in self.ENGINE_REGISTRY:
            raise ValueError(f"Unknown layout engine: '{engine_name}'. Available: {list(self.ENGINE_REGISTRY.keys())}")

        if engine_name not in self._detector_instances:
            logger.info(f"Creating instance of layout engine: {engine_name}")
            engine_class = self.ENGINE_REGISTRY[engine_name]['class']
            detector_instance = engine_class() # Instantiate
            if not detector_instance.is_available():
                 # Check availability before storing
                 raise RuntimeError(f"Layout engine '{engine_name}' is not available. Please check dependencies.")
            self._detector_instances[engine_name] = detector_instance # Store if available

        return self._detector_instances[engine_name]

    def analyze_layout(
        self,
        image: Image.Image,
        engine: Optional[str] = None, # Default engine handled below
        options: Optional[LayoutOptions] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Analyzes layout of a single image using simple args or an options object.

        Args:
            image: The PIL Image to analyze.
            engine: Name of the engine (e.g., 'yolo', 'tatr'). Ignored if 'options' provided.
                    Defaults to the first available engine if None.
            options: Specific LayoutOptions object for advanced configuration.
            **kwargs: For simple mode, accepts: 'confidence', 'classes',
                      'exclude_classes', 'device'.

        Returns:
            A list of standardized detection dictionaries.
        """
        final_options: BaseLayoutOptions
        selected_engine_name: str

        if not isinstance(image, Image.Image):
             raise TypeError("Input 'image' must be a PIL Image.")

        available_engines = self.get_available_engines()
        if not available_engines:
             raise RuntimeError("No layout engines are available. Please check dependencies.")

        # Determine default engine if not specified
        default_engine = engine if engine else available_engines[0]

        # --- Determine Options and Engine ---
        if options is not None:
            # Advanced Mode: An options object was provided directly (or constructed by LayoutAnalyzer)
            # Use this object directly, do not deep copy or reconstruct.
            logger.debug(f"LayoutManager: Using provided options object: {type(options).__name__}")
            final_options = options # Use the provided object directly
            found_engine = False
            for name, registry_entry in self.ENGINE_REGISTRY.items():
                if isinstance(options, registry_entry['options_class']):
                    selected_engine_name = name
                    found_engine = True
                    break
            if not found_engine:
                 raise TypeError(f"Provided options object type '{type(options).__name__}' does not match any registered layout engine options.")
            # Ignore simple kwargs if options object is present
            if kwargs:
                logger.warning(f"Keyword arguments {list(kwargs.keys())} were provided alongside an 'options' object and will be ignored.")
        else:
            # Simple Mode: No options object provided initially. 
            # Determine engine from kwargs or default, then construct options.
            selected_engine_name = default_engine.lower()
            logger.debug(f"LayoutManager: Using simple mode. Engine: '{selected_engine_name}', kwargs: {kwargs}")

            if selected_engine_name not in self.ENGINE_REGISTRY:
                 raise ValueError(f"Unknown or unavailable layout engine: '{selected_engine_name}'. Available: {available_engines}")

            unexpected_kwargs = set(kwargs.keys()) - self.SIMPLE_MODE_ALLOWED_KWARGS
            if unexpected_kwargs:
                raise TypeError(f"Got unexpected keyword arguments in simple mode: {list(unexpected_kwargs)}. Use the 'options' parameter for detailed configuration.")

            options_class = self.ENGINE_REGISTRY[selected_engine_name]['options_class']
            # Use BaseLayoutOptions defaults unless overridden by kwargs
            base_defaults = BaseLayoutOptions()
            simple_args = {
                'confidence': kwargs.get('confidence', base_defaults.confidence),
                'classes': kwargs.get('classes'),
                'exclude_classes': kwargs.get('exclude_classes'),
                'device': kwargs.get('device', base_defaults.device)
            }
            # Filter out None values before passing to constructor
            simple_args_filtered = {k: v for k, v in simple_args.items() if v is not None}
            final_options = options_class(**simple_args_filtered)
            logger.debug(f"LayoutManager: Constructed options for simple mode: {final_options}")


        # --- Get Engine Instance and Process ---
        try:
            engine_instance = self._get_engine_instance(selected_engine_name)
            logger.info(f"Analyzing layout with engine '{selected_engine_name}'...")

            # Call the engine's detect method
            detections = engine_instance.detect(image, final_options)

            logger.info(f"Layout analysis complete. Found {len(detections)} regions.")
            return detections

        except (ImportError, RuntimeError, ValueError, TypeError) as e:
             logger.error(f"Layout analysis failed for engine '{selected_engine_name}': {e}", exc_info=True)
             raise # Re-raise expected errors
        except Exception as e:
             logger.error(f"An unexpected error occurred during layout analysis: {e}", exc_info=True)
             raise # Re-raise unexpected errors


    def get_available_engines(self) -> List[str]:
        """Returns a list of registered layout engine names that are currently available."""
        available = []
        for name, registry_entry in self.ENGINE_REGISTRY.items():
             try:
                 engine_class = registry_entry['class']
                 # Check availability without full instantiation if possible
                 if hasattr(engine_class, 'is_available') and callable(engine_class.is_available):
                      # Create temporary instance only for check if needed, or use classmethod
                      if engine_class().is_available(): # Assumes instance needed for check
                           available.append(name)
                 else:
                      # Assume available if class exists (less robust)
                      available.append(name)
             except Exception as e:
                 logger.debug(f"Layout engine '{name}' check failed: {e}")
                 pass
        return available

