"""Manager for checkbox detection engines."""

import logging
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

from PIL import Image

from .base import CheckboxDetector
from .checkbox_options import CheckboxOptions, RTDETRCheckboxOptions

logger = logging.getLogger(__name__)


def _lazy_import_rtdetr_detector() -> Type[CheckboxDetector]:
    """Lazy import RT-DETR detector to avoid heavy dependencies at module load."""
    from .rtdetr import RTDETRCheckboxDetector

    return RTDETRCheckboxDetector


@dataclass(frozen=True)
class EngineRegistration:
    """Describes a registered checkbox detection engine."""

    detector_factory: Callable[[], Type[CheckboxDetector]]
    options_type: Type[CheckboxOptions]


class CheckboxManager:
    """Manages checkbox detection engines and provides a unified interface."""

    # Registry of available engines
    DEFAULT_ENGINE = "rtdetr"
    ENGINE_REGISTRY: Dict[str, EngineRegistration] = {
        "rtdetr": EngineRegistration(
            detector_factory=_lazy_import_rtdetr_detector,
            options_type=RTDETRCheckboxOptions,
        ),
        "wendys": EngineRegistration(
            detector_factory=_lazy_import_rtdetr_detector,
            options_type=RTDETRCheckboxOptions,
        ),
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
        engine_name, final_options = self.prepare_options(engine, options, overrides=kwargs)

        # Get detector
        detector = self._get_detector(engine_name)

        # Run detection
        try:
            return detector.detect(image, final_options)
        except Exception as e:
            self.logger.error(f"Checkbox detection failed with {engine_name}: {e}", exc_info=True)
            raise

    def prepare_options(
        self,
        engine: Optional[str],
        options: Optional[Union[CheckboxOptions, Mapping[str, Any]]],
        *,
        overrides: Optional[Mapping[str, Any]] = None,
    ) -> Tuple[str, CheckboxOptions]:
        """
        Resolve the engine name and return a concrete options instance.

        Args:
            engine: Requested engine name (optional).
            options: Existing CheckboxOptions instance or mapping of parameters.
            overrides: Additional option values to merge in.

        Returns:
            Tuple of (engine_name, CheckboxOptions instance).
        """
        override_kwargs: Dict[str, Any] = {}
        if overrides:
            override_kwargs.update(dict(overrides))

        if isinstance(options, CheckboxOptions):
            engine_name = engine or self._get_engine_from_options(options)
            final_options = (
                self._override_options(options, **override_kwargs) if override_kwargs else options
            )
            return engine_name, final_options

        engine_name = engine or self.DEFAULT_ENGINE
        base_kwargs: Dict[str, Any] = {}
        if isinstance(options, Mapping):
            base_kwargs.update(dict(options))
        base_kwargs.update(override_kwargs)

        final_options = self._create_options(engine_name, **base_kwargs)
        return engine_name, final_options

    def _get_engine_from_options(self, options: CheckboxOptions) -> str:
        """Determine engine from options type."""
        for engine_name, registration in self.ENGINE_REGISTRY.items():
            if isinstance(options, registration.options_type):
                return engine_name
        # Default if can't determine
        return self.DEFAULT_ENGINE

    def _create_options(self, engine: str, **kwargs) -> CheckboxOptions:
        """Create options instance for the specified engine."""
        if engine not in self.ENGINE_REGISTRY:
            raise ValueError(
                f"Unknown checkbox detection engine: {engine}. "
                f"Available: {list(self.ENGINE_REGISTRY.keys())}"
            )

        registration = self.ENGINE_REGISTRY[engine]
        return registration.options_type(**kwargs)

    def _override_options(self, options: CheckboxOptions, **kwargs) -> CheckboxOptions:
        """Create a new options instance with overrides applied."""
        return replace(options, **kwargs)

    def _get_detector(self, engine: str) -> CheckboxDetector:
        """Get or create a detector instance for the specified engine."""
        if engine not in self._detector_cache:
            if engine not in self.ENGINE_REGISTRY:
                raise ValueError(
                    f"Unknown checkbox detection engine: {engine}. "
                    f"Available: {list(self.ENGINE_REGISTRY.keys())}"
                )

            # Get detector class (lazy import)
            registration = self.ENGINE_REGISTRY[engine]
            detector_cls = registration.detector_factory()

            # Check availability
            if not detector_cls.is_available():
                raise RuntimeError(
                    f"Checkbox detection engine '{engine}' is not available. "
                    f"Please install required dependencies."
                )

            # Create instance
            self._detector_cache[engine] = detector_cls()
            self.logger.info(f"Initialized checkbox detector: {engine}")

        return self._detector_cache[engine]

    def is_engine_available(self, engine: str) -> bool:
        """Check if a specific engine is available."""
        if engine not in self.ENGINE_REGISTRY:
            return False

        try:
            detector_cls = self.ENGINE_REGISTRY[engine].detector_factory()
            return detector_cls.is_available()
        except Exception:
            return False

    def list_available_engines(self) -> List[str]:
        """List all available checkbox detection engines."""
        available = []
        for engine in self.ENGINE_REGISTRY:
            if self.is_engine_available(engine):
                available.append(engine)
        return available
