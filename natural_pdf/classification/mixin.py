import logging
import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from PIL.Image import Image

if TYPE_CHECKING:
    from natural_pdf.classification.manager import ClassificationManager

logger = logging.getLogger(__name__)

class ClassificationError(Exception):
    """Custom exception for classification failures."""
    pass

class ClassificationMixin(ABC):
    """
    Mixin class providing classification capabilities to Page and Region objects.
    """

    @abstractmethod
    def _get_classification_manager(self) -> "ClassificationManager":
        """Return the ClassificationManager instance."""
        pass

    @abstractmethod
    def _get_classification_content(self, model_type: str) -> Union[str, Image]:
        """
        Get the content suitable for the specified model type ('text' or 'vision').

        Raises:
            ValueError: If content cannot be generated.
        """
        pass

    @abstractmethod
    def _get_metadata_storage(self) -> Dict[str, Any]:
        """Return the dictionary where classification results should be stored."""
        pass

    def classify(
        self,
        categories: List[str],
        model: str = "text",
        engine_type: Optional[str] = None,
        min_confidence: float = 0.0,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Classify this object (Page or Region) into one of the provided categories.

        Args:
            categories: A list of string category names.
            model: Model identifier ('text', 'vision', or specific Hugging Face model ID).
            engine_type: Explicitly specify 'text' or 'vision'. If None, it's inferred.
            min_confidence: Minimum confidence score for a category to be included.
            **kwargs: Additional keyword arguments passed to the ClassificationManager.

        Returns:
            A dictionary containing the classification results.

        Raises:
            ImportError: If required classification dependencies are not installed.
            ValueError: If categories list is empty, or content generation fails.
            ClassificationError: If model loading or inference fails.
        """
        if not categories:
            raise ValueError("Categories list cannot be empty.")

        try:
            manager = self._get_classification_manager()
        except AttributeError as e:
             raise ClassificationError("Could not access classification manager.") from e

        if not manager:
             try:
                 from natural_pdf.classification.manager import ClassificationManager
                 raise ClassificationError("ClassificationManager is not available.")
             except ImportError:
                  raise ImportError(
                     "Classification dependencies missing. "
                     "Install with: pip install \"natural-pdf[classification]\""
                  )

        if engine_type is None:
            engine_type = manager.infer_engine_type(model)
        elif engine_type not in ["text", "vision"]:
            raise ValueError("engine_type must be 'text' or 'vision' if specified.")

        logger.info(f"Classifying {self!r} using model '{model}' (engine: {engine_type}) with categories: {categories}")

        try:
            content = self._get_classification_content(engine_type)
        except ValueError as e:
            raise ClassificationError(f"Cannot classify: {e}") from e
        except Exception as e:
            raise ClassificationError(f"Error getting classification content: {e}") from e

        result_dict = manager.get_classification_result(
            item_content=content,
            categories=categories,
            model_id=model,
            engine_type=engine_type,
            min_confidence=min_confidence,
            **kwargs,
        )

        try:
            metadata = self._get_metadata_storage()
            metadata['classification'] = result_dict
            logger.debug(f"Stored classification results in metadata for {self!r}")
        except Exception as e:
            logger.warning(f"Failed to store classification results in metadata for {self!r}: {e}")

        return result_dict

    @property
    def category(self) -> Optional[str]:
        """Returns the top category label from the most recent classification, or None."""
        results = self.classification_results
        if results and results.get('scores'):
            return results['scores'][0].get('label')
        return None

    @property
    def category_confidence(self) -> Optional[float]:
        """Returns the top category confidence score from the most recent classification, or None."""
        results = self.classification_results
        if results and results.get('scores'):
            return results['scores'][0].get('confidence')
        return None

    @property
    def classification_results(self) -> Optional[Dict]:
        """Returns the full results dictionary from the most recent classification, or None."""
        try:
            metadata = self._get_metadata_storage()
            return metadata.get('classification')
        except AttributeError:
             logger.error(f"Object {self!r} missing expected metadata storage.")
             return None
        except Exception as e:
             logger.error(f"Error accessing classification results for {self!r}: {e}")
             return None 