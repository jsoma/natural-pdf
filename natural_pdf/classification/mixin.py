import logging
import warnings
from typing import TYPE_CHECKING, Any, List, Optional, Union

from PIL import Image

from natural_pdf.classification.results import ClassificationResult
from natural_pdf.services.base import resolve_service

if TYPE_CHECKING:
    from .manager import ClassificationManager

logger = logging.getLogger(__name__)


class ClassificationMixin:
    """Mixin class providing classification capabilities to Page and Region objects.

    This mixin adds AI-powered classification functionality to pages, regions, and
    elements, enabling document categorization and content analysis using both
    text-based and vision-based models. It integrates with the ClassificationManager
    to provide a consistent interface across different model types.
    """

    def _get_classification_content(self, model_type: str, **kwargs) -> Union[str, "Image.Image"]:
        """Should return the text content (str) or image (PIL.Image) for classification."""
        raise NotImplementedError

    def classify(
        self,
        labels: List[str],
        model: Optional[str] = None,
        using: Optional[str] = None,
        min_confidence: float = 0.0,
        analysis_key: str = "classification",
        multi_label: bool = False,
        **kwargs,
    ) -> "ClassificationMixin":
        """Classify this item via the classification service."""
        resolve_service(self, "classification").classify(
            host=self,
            labels=labels,
            model=model,
            using=using,
            min_confidence=min_confidence,
            analysis_key=analysis_key,
            multi_label=multi_label,
            **kwargs,
        )
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
