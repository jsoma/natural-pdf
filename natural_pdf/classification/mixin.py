import logging
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

# Assuming PIL is installed as it's needed for vision
try:
    from PIL import Image
except ImportError:
    Image = None  # type: ignore

# Import result classes
from .results import ClassificationResult  # Assuming results.py is in the same dir

if TYPE_CHECKING:
    # Avoid runtime import cycle
    from natural_pdf.core.page import Page
    from natural_pdf.elements.region import Region

    from .manager import ClassificationManager

logger = logging.getLogger(__name__)


class ClassificationMixin:
    """
    Mixin class providing classification capabilities to Page and Region objects.
    Relies on a ClassificationManager being accessible, typically via the parent PDF.
    """

    # --- Abstract methods/properties required by the host class --- #
    # These must be implemented by classes using this mixin (Page, Region)

    def _get_classification_manager(self) -> "ClassificationManager":
        """Should return the ClassificationManager instance."""
        raise NotImplementedError

    def _get_classification_content(self, model_type: str, **kwargs) -> Union[str, "Image"]:
        """Should return the text content (str) or image (PIL.Image) for classification."""
        raise NotImplementedError

    # Host class needs 'analyses' attribute initialized as Dict[str, Any]
    # analyses: Dict[str, Any]

    # --- End Abstract --- #

    def classify(
        self,
        labels: List[str],
        model: Optional[str] = None,
        using: Optional[str] = None,
        min_confidence: float = 0.0,
        analysis_key: str = "classification",  # Default key
        multi_label: bool = False,
        **kwargs,
    ) -> "ClassificationMixin":  # Return self for chaining
        """
        Classifies this item (Page or Region) using the configured manager.

        Stores the result in self.analyses[analysis_key]. If analysis_key is not
        provided, it defaults to 'classification' and overwrites any previous
        result under that key.

        Args:
            labels: A list of string category names.
            model: Model identifier (e.g., 'text', 'vision', HF ID). Defaults handled by manager.
            using: Optional processing mode ('text' or 'vision'). If None, inferred by manager.
            min_confidence: Minimum confidence threshold for results (0.0-1.0).
            analysis_key: Key under which to store the result in `self.analyses`.
                          Defaults to 'classification'.
            multi_label: Whether to allow multiple labels (passed to HF pipeline).
            **kwargs: Additional arguments passed to the ClassificationManager.

        Returns:
            Self for method chaining.
        """
        # Ensure analyses dict exists
        if not hasattr(self, "analyses") or self.analyses is None:
            logger.warning("'analyses' attribute not found or is None. Initializing as empty dict.")
            self.analyses = {}

        try:
            manager = self._get_classification_manager()

            # Determine the effective model ID and engine type
            effective_model_id = model
            inferred_using = manager.infer_using(
                model if model else manager.DEFAULT_TEXT_MODEL, using
            )

            # If model was not provided, use the manager's default for the inferred engine type
            if effective_model_id is None:
                effective_model_id = (
                    manager.DEFAULT_TEXT_MODEL
                    if inferred_using == "text"
                    else manager.DEFAULT_VISION_MODEL
                )
                logger.debug(
                    f"No model provided, using default for mode '{inferred_using}': '{effective_model_id}'"
                )

            # Get content based on the *final* determined engine type
            content = self._get_classification_content(model_type=inferred_using, **kwargs)

            # Manager now returns a ClassificationResult object
            result_obj: ClassificationResult = manager.classify_item(
                item_content=content,
                labels=labels,
                model_id=effective_model_id,
                using=inferred_using,
                min_confidence=min_confidence,
                multi_label=multi_label,
                **kwargs,
            )

            # Store the structured result object under the specified key
            self.analyses[analysis_key] = result_obj
            logger.debug(f"Stored classification result under key '{analysis_key}': {result_obj}")

        except NotImplementedError as nie:
            logger.error(f"Classification cannot proceed: {nie}")
            raise
        except Exception as e:
            logger.error(f"Classification failed: {e}", exc_info=True)
            # Optionally re-raise or just log and return self
            # raise

        return self

    @property
    def classification_results(self) -> Optional[ClassificationResult]:
        """Returns the ClassificationResult from the *default* ('classification') key, or None."""
        if not hasattr(self, "analyses") or self.analyses is None:
            return None
        # Return the result object directly from the default key
        return self.analyses.get("classification")

    @property
    def category(self) -> Optional[str]:
        """Returns the top category label from the *default* ('classification') key, or None."""
        result_obj = self.classification_results  # Uses the property above
        # Access the property on the result object
        return result_obj.top_category if result_obj else None

    @property
    def category_confidence(self) -> Optional[float]:
        """Returns the top category confidence from the *default* ('classification') key, or None."""
        result_obj = self.classification_results  # Uses the property above
        # Access the property on the result object
        return result_obj.top_confidence if result_obj else None

    # Maybe add a helper to get results by specific key?
    def get_classification_result(
        self, analysis_key: str = "classification"
    ) -> Optional[ClassificationResult]:
        """Gets a classification result object stored under a specific key."""
        if not hasattr(self, "analyses") or self.analyses is None:
            return None
        result = self.analyses.get(analysis_key)
        if result is not None and not isinstance(result, ClassificationResult):
            logger.warning(
                f"Item found under key '{analysis_key}' is not a ClassificationResult (type: {type(result)}). Returning None."
            )
            return None
        return result
