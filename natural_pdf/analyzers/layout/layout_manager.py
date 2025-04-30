# layout_manager.py
import copy
import logging
from typing import Any, Dict, List, Optional, Type, Union

from PIL import Image

# --- Import detector classes and options ---
# Use try-except blocks for robustness if some detectors might be missing dependencies
try:
    from .base import LayoutDetector
except ImportError:
    LayoutDetector = type("LayoutDetector", (), {})

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

try:
    from .gemini import GeminiLayoutDetector
except ImportError:
    GeminiLayoutDetector = None

from .layout_options import (
    BaseLayoutOptions,
    DoclingLayoutOptions,
    GeminiLayoutOptions,
    LayoutOptions,
    PaddleLayoutOptions,
    SuryaLayoutOptions,
    TATRLayoutOptions,
    YOLOLayoutOptions,
)

logger = logging.getLogger(__name__)


class LayoutManager:
    """Manages layout detector selection, configuration, and execution."""

    # Registry mapping engine names to classes and default options
    ENGINE_REGISTRY: Dict[str, Dict[str, Any]] = {}

    # Populate registry only with available detectors
    if YOLODocLayoutDetector:
        ENGINE_REGISTRY["yolo"] = {
            "class": YOLODocLayoutDetector,
            "options_class": YOLOLayoutOptions,
        }
    if TableTransformerDetector:
        ENGINE_REGISTRY["tatr"] = {
            "class": TableTransformerDetector,
            "options_class": TATRLayoutOptions,
        }
    if PaddleLayoutDetector:
        ENGINE_REGISTRY["paddle"] = {
            "class": PaddleLayoutDetector,
            "options_class": PaddleLayoutOptions,
        }
    if SuryaLayoutDetector:
        ENGINE_REGISTRY["surya"] = {
            "class": SuryaLayoutDetector,
            "options_class": SuryaLayoutOptions,
        }
    if DoclingLayoutDetector:
        ENGINE_REGISTRY["docling"] = {
            "class": DoclingLayoutDetector,
            "options_class": DoclingLayoutOptions,
        }

    # Add Gemini entry if available
    if GeminiLayoutDetector:
        ENGINE_REGISTRY["gemini"] = {
            "class": GeminiLayoutDetector,
            "options_class": GeminiLayoutOptions,
        }

    def __init__(self):
        """Initializes the Layout Manager."""
        # Cache for detector instances (different from model cache inside detector)
        self._detector_instances: Dict[str, LayoutDetector] = {}
        logger.info(
            f"LayoutManager initialized. Available engines: {list(self.ENGINE_REGISTRY.keys())}"
        )

    def _get_engine_instance(self, engine_name: str) -> LayoutDetector:
        """Retrieves or creates an instance of the specified layout detector."""
        engine_name = engine_name.lower()
        if engine_name not in self.ENGINE_REGISTRY:
            raise ValueError(
                f"Unknown layout engine: '{engine_name}'. Available: {list(self.ENGINE_REGISTRY.keys())}"
            )

        if engine_name not in self._detector_instances:
            logger.info(f"Creating instance of layout engine: {engine_name}")
            engine_class = self.ENGINE_REGISTRY[engine_name]["class"]
            detector_instance = engine_class()  # Instantiate
            if not detector_instance.is_available():
                # Check availability before storing
                # Construct helpful error message with install hint
                install_hint = ""
                if engine_name == "yolo":
                    install_hint = "pip install 'natural-pdf[layout_yolo]'"
                elif engine_name == "tatr":
                    install_hint = "pip install 'natural-pdf[core-ml]'"
                elif engine_name == "paddle":
                    install_hint = "pip install 'natural-pdf[paddle]'"
                elif engine_name == "surya":
                    install_hint = "pip install 'natural-pdf[surya]'"
                # Add other engines like docling if they become optional extras
                else:
                    install_hint = f"(Check installation requirements for {engine_name})"

                raise RuntimeError(
                    f"Layout engine '{engine_name}' is not available. Please install the required dependencies: {install_hint}"
                )
            self._detector_instances[engine_name] = detector_instance  # Store if available

        return self._detector_instances[engine_name]

    def analyze_layout(
        self,
        image: Image.Image,
        options: LayoutOptions,
    ) -> List[Dict[str, Any]]:
        """
        Analyzes layout of a single image using a specific options object.

        Args:
            image: The PIL Image to analyze.
            options: Specific LayoutOptions object containing configuration and context.
                     This object MUST be provided.

        Returns:
            A list of standardized detection dictionaries.
        """
        selected_engine_name: Optional[str] = None
        found_engine = False
        for name, registry_entry in self.ENGINE_REGISTRY.items():
            if isinstance(options, registry_entry["options_class"]):
                selected_engine_name = name
                found_engine = True
                break
        if not found_engine or selected_engine_name is None:
            available_options_types = [
                reg["options_class"].__name__ for reg in self.ENGINE_REGISTRY.values()
            ]
            raise TypeError(
                f"Provided options object type '{type(options).__name__}' does not match any registered layout engine options: {available_options_types}"
            )

        try:
            engine_instance = self._get_engine_instance(selected_engine_name)
            logger.info(f"Analyzing layout with engine '{selected_engine_name}'...")

            detections = engine_instance.detect(image, options)  # Pass options directly

            logger.info(f"Layout analysis complete. Found {len(detections)} regions.")
            return detections

        except (ImportError, RuntimeError, ValueError, TypeError) as e:
            # Add engine name to error message if possible
            engine_context = f" for engine '{selected_engine_name}'" if selected_engine_name else ""
            logger.error(f"Layout analysis failed{engine_context}: {e}", exc_info=True)
            raise  # Re-raise expected errors
        except Exception as e:
            engine_context = f" for engine '{selected_engine_name}'" if selected_engine_name else ""
            logger.error(
                f"An unexpected error occurred during layout analysis{engine_context}: {e}",
                exc_info=True,
            )
            raise  # Re-raise unexpected errors

    def get_available_engines(self) -> List[str]:
        """Returns a list of registered layout engine names that are currently available."""
        available = []
        for name, registry_entry in self.ENGINE_REGISTRY.items():
            try:
                engine_class = registry_entry["class"]
                # Check availability without full instantiation if possible
                if hasattr(engine_class, "is_available") and callable(engine_class.is_available):
                    # Create temporary instance only for check if needed, or use classmethod
                    if engine_class().is_available():  # Assumes instance needed for check
                        available.append(name)
                else:
                    # Assume available if class exists (less robust)
                    available.append(name)
            except Exception as e:
                logger.debug(f"Layout engine '{name}' check failed: {e}")
                pass
        return available
