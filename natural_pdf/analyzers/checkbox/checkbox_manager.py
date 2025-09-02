"""Manager for checkbox detection engines."""

import logging
from typing import Any, Dict, List, Optional, Type, Union

from PIL import Image

from .base import CheckboxDetector
from .checkbox_options import CheckboxOptions, RTDETRCheckboxOptions

logger = logging.getLogger(__name__)


def _lazy_import_rtdetr_detector():
    """Lazy import RT-DETR detector to avoid heavy dependencies at module load."""
    from .rtdetr import RTDETRCheckboxDetector

    return RTDETRCheckboxDetector


class CheckboxManager:
    """Manages checkbox detection engines and provides a unified interface."""

    # Registry of available engines
    ENGINE_REGISTRY = {
        "rtdetr": {
            "class": _lazy_import_rtdetr_detector,
            "options_class": RTDETRCheckboxOptions,
        },
        "wendys": {  # Alias for the default model
            "class": _lazy_import_rtdetr_detector,
            "options_class": RTDETRCheckboxOptions,
        },
    }

    def __init__(self):
        """Initialize the checkbox manager."""
        self.logger = logging.getLogger(__name__)
        self._detector_cache: Dict[str, CheckboxDetector] = {}

    def detect_checkboxes(
        self,
        image: Image.Image,
        engine: Optional[str] = None,
        options: Optional[Union[CheckboxOptions, Dict[str, Any]]] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Detect checkboxes in an image using the specified engine.

        Args:
            image: PIL Image to analyze
            engine: Name of the detection engine (default: 'rtdetr')
            options: CheckboxOptions instance or dict of options
            **kwargs: Additional options to override

        Returns:
            List of detection dictionaries
        """
        # Determine engine and options
        if options is None:
            if engine is None:
                engine = "rtdetr"  # Default engine
            options = self._create_options(engine, **kwargs)
        elif isinstance(options, dict):
            if engine is None:
                engine = "rtdetr"
            options = self._create_options(engine, **options, **kwargs)
        else:
            # options is a CheckboxOptions instance
            # Determine engine from options type if not specified
            if engine is None:
                engine = self._get_engine_from_options(options)
            # Apply any kwargs as overrides
            if kwargs:
                options = self._override_options(options, **kwargs)

        # Get detector
        detector = self._get_detector(engine)

        # Run detection
        try:
            return detector.detect(image, options)
        except Exception as e:
            self.logger.error(f"Checkbox detection failed with {engine}: {e}", exc_info=True)
            raise

    def _get_engine_from_options(self, options: CheckboxOptions) -> str:
        """Determine engine from options type."""
        for engine_name, engine_info in self.ENGINE_REGISTRY.items():
            if isinstance(options, engine_info["options_class"]):
                return engine_name
        # Default if can't determine
        return "rtdetr"

    def _create_options(self, engine: str, **kwargs) -> CheckboxOptions:
        """Create options instance for the specified engine."""
        if engine not in self.ENGINE_REGISTRY:
            raise ValueError(
                f"Unknown checkbox detection engine: {engine}. "
                f"Available: {list(self.ENGINE_REGISTRY.keys())}"
            )

        options_class = self.ENGINE_REGISTRY[engine]["options_class"]
        return options_class(**kwargs)

    def _override_options(self, options: CheckboxOptions, **kwargs) -> CheckboxOptions:
        """Create a new options instance with overrides applied."""
        # Get current values as dict
        import dataclasses

        current_values = dataclasses.asdict(options)

        # Apply overrides
        current_values.update(kwargs)

        # Create new instance
        return type(options)(**current_values)

    def _get_detector(self, engine: str) -> CheckboxDetector:
        """Get or create a detector instance for the specified engine."""
        if engine not in self._detector_cache:
            if engine not in self.ENGINE_REGISTRY:
                raise ValueError(
                    f"Unknown checkbox detection engine: {engine}. "
                    f"Available: {list(self.ENGINE_REGISTRY.keys())}"
                )

            # Get detector class (lazy import)
            detector_class = self.ENGINE_REGISTRY[engine]["class"]
            if callable(detector_class):
                detector_class = detector_class()  # Call factory function

            # Check availability
            if not detector_class.is_available():
                raise RuntimeError(
                    f"Checkbox detection engine '{engine}' is not available. "
                    f"Please install required dependencies."
                )

            # Create instance
            self._detector_cache[engine] = detector_class()
            self.logger.info(f"Initialized checkbox detector: {engine}")

        return self._detector_cache[engine]

    def is_engine_available(self, engine: str) -> bool:
        """Check if a specific engine is available."""
        if engine not in self.ENGINE_REGISTRY:
            return False

        try:
            detector_class = self.ENGINE_REGISTRY[engine]["class"]
            if callable(detector_class):
                detector_class = detector_class()
            return detector_class.is_available()
        except Exception:
            return False

    def list_available_engines(self) -> List[str]:
        """List all available checkbox detection engines."""
        available = []
        for engine in self.ENGINE_REGISTRY:
            if self.is_engine_available(engine):
                available.append(engine)
        return available
