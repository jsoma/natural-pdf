import logging
import warnings
from typing import TYPE_CHECKING, List, Optional, Union

from PIL import Image

from .classification_provider import get_classification_engine, run_classification_item
from .results import ClassificationResult

if TYPE_CHECKING:

    from .manager import ClassificationManager

logger = logging.getLogger(__name__)


class ClassificationMixin:
    """Mixin class providing classification capabilities to Page and Region objects.

    This mixin adds AI-powered classification functionality to pages, regions, and
    elements, enabling document categorization and content analysis using both
    text-based and vision-based models. It integrates with the ClassificationManager
    to provide a consistent interface across different model types.

    The mixin supports both single-label and multi-label classification, confidence
    thresholding, and various analysis storage strategies for complex workflows.
    Results are stored in the host object's 'analyses' dictionary for retrieval
    and further processing.

    Classification modes:
    - Text-based: Uses extracted text content for classification
    - Vision-based: Uses rendered images for visual classification
    - Automatic: Manager selects best mode based on content availability

    Host class requirements:
    - Must implement _get_classification_content() -> str | Image
    - Must have 'analyses' attribute as Dict[str, Any]

    Example:
        ```python
        pdf = npdf.PDF("document.pdf")
        page = pdf.pages[0]

        # Document type classification
        page.classify(['invoice', 'contract', 'report'],
                     model='text', analysis_key='doc_type')

        # Multi-label content analysis
        region = page.find('text:contains("Summary")').below()
        region.classify(['technical', 'financial', 'legal'],
                       multi_label=True, min_confidence=0.8)

        # Access results
        doc_type = page.analyses['doc_type']
        content_labels = region.analyses['classification']
        ```

    Note:
        Classification requires appropriate models to be available through the
        ClassificationManager. Results include confidence scores and detailed
        metadata for analysis workflows.
    """

    # --- Abstract methods/properties required by the host class --- #
    # These must be implemented by classes using this mixin (Page, Region)

    def _get_classification_content(self, model_type: str, **kwargs) -> Union[str, "Image.Image"]:
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
            engine_obj = get_classification_engine(self, kwargs.pop("classification_engine", None))

            chosen_mode = using
            content = None

            candidate_model = model or engine_obj.default_model("text")
            inferred_mode = engine_obj.infer_using(candidate_model, chosen_mode)
            chosen_mode = inferred_mode

            if chosen_mode == "text":
                try:
                    tentative_text = self._get_classification_content("text", **kwargs)
                    if tentative_text and not (
                        isinstance(tentative_text, str) and tentative_text.isspace()
                    ):
                        content = tentative_text
                    else:
                        raise ValueError("Empty text")
                except Exception:
                    warnings.warn(
                        "No text found for classification; falling back to vision model. "
                        "Pass using='vision' explicitly to silence this message.",
                        UserWarning,
                    )
                    chosen_mode = "vision"

            if content is None:
                if chosen_mode is None:
                    chosen_mode = "vision"
                content = self._get_classification_content(model_type=chosen_mode, **kwargs)

            effective_model_id = model or engine_obj.default_model(chosen_mode)

            result_obj = run_classification_item(
                context=self,
                content=content,
                labels=labels,
                model_id=effective_model_id,
                using=chosen_mode,
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
