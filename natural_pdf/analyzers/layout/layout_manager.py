# layout_manager.py
import copy
import logging
from typing import Any, Dict, List, Optional, Type, Union

from PIL import Image

# --- Import lightweight components only ---
# Heavy detector implementations (paddle, yolo, etc.) are **not** imported at module load.
# Instead, we provide tiny helper functions that import them lazily **only when needed**.

from .base import LayoutDetector  # Lightweight base class
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

# ------------------ Lazy import helpers ------------------ #


def _lazy_import_yolo_detector():
    """Import YOLO detector lazily to avoid heavy deps at import time."""
    from .yolo import YOLODocLayoutDetector  # Local import

    return YOLODocLayoutDetector


def _lazy_import_tatr_detector():
    from .tatr import TableTransformerDetector

    return TableTransformerDetector


def _lazy_import_paddle_detector():
    from .paddle import PaddleLayoutDetector

    return PaddleLayoutDetector


def _lazy_import_surya_detector():
    from .surya import SuryaLayoutDetector

    return SuryaLayoutDetector


def _lazy_import_docling_detector():
    from .docling import DoclingLayoutDetector

    return DoclingLayoutDetector


def _lazy_import_gemini_detector():
    from .gemini import GeminiLayoutDetector

    return GeminiLayoutDetector

# --------------------------------------------------------- #

logger = logging.getLogger(__name__)


class LayoutManager:
    """Manages layout detector selection, configuration, and execution."""

    # Registry mapping engine names to classes and default options
    ENGINE_REGISTRY: Dict[str, Dict[str, Any]] = {}

    # Populate registry with lazy import callables. The heavy imports are executed only
    # when the corresponding engine is first requested.
    ENGINE_REGISTRY = {
        "yolo": {
            "class": _lazy_import_yolo_detector,  # returns detector class when called
            "options_class": YOLOLayoutOptions,
        },
        "tatr": {
            "class": _lazy_import_tatr_detector,
            "options_class": TATRLayoutOptions,
        },
        "paddle": {
            "class": _lazy_import_paddle_detector,
            "options_class": PaddleLayoutOptions,
        },
        "surya": {
            "class": _lazy_import_surya_detector,
            "options_class": SuryaLayoutOptions,
        },
        "docling": {
            "class": _lazy_import_docling_detector,
            "options_class": DoclingLayoutOptions,
        },
        "gemini": {
            "class": _lazy_import_gemini_detector,
            "options_class": GeminiLayoutOptions,
        },
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
            engine_class_or_factory = self.ENGINE_REGISTRY[engine_name]["class"]
            # If the registry provides a callable (lazy import helper), call it to obtain the real class.
            if callable(engine_class_or_factory) and not isinstance(engine_class_or_factory, type):
                engine_class = engine_class_or_factory()
            else:
                engine_class = engine_class_or_factory

            detector_instance = engine_class()  # Instantiate
            if not detector_instance.is_available():
                # Check availability before storing
                # Construct helpful error message with install hint
                install_hint = ""
                if engine_name == "yolo":
                    install_hint = "pip install doclayout_yolo"
                elif engine_name == "tatr":
                    # This should now be installed with core dependencies
                    install_hint = "(should be installed with natural-pdf, check for import errors)"
                elif engine_name == "paddle":
                    install_hint = "pip install paddleocr paddlepaddle"
                elif engine_name == "surya":
                    install_hint = "pip install surya-ocr"
                elif engine_name == "docling":
                    install_hint = "pip install docling"
                elif engine_name == "gemini":
                    install_hint = "pip install openai"
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
