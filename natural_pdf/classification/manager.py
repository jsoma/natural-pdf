import logging
import time
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

from PIL import Image

# Use try-except for robustness if dependencies are missing
try:
    import torch
    from transformers import (
        AutoModelForSequenceClassification,
        AutoModelForZeroShotImageClassification,
        AutoTokenizer,
        pipeline,
    )

    _CLASSIFICATION_AVAILABLE = True
except ImportError:
    _CLASSIFICATION_AVAILABLE = False
    # Define dummy types for type hinting if imports fail
    pipeline = object
    AutoTokenizer = object
    AutoModelForZeroShotImageClassification = object
    AutoModelForSequenceClassification = object
    torch = None

from natural_pdf.utils.tqdm_utils import get_tqdm

# Import result classes
from .results import CategoryScore, ClassificationResult

if TYPE_CHECKING:
    from transformers import Pipeline


logger = logging.getLogger(__name__)

# Global cache for models/pipelines
_PIPELINE_CACHE: Dict[str, "Pipeline"] = {}
_TOKENIZER_CACHE: Dict[str, Any] = {}
_MODEL_CACHE: Dict[str, Any] = {}


class ClassificationError(Exception):
    """Custom exception for classification errors."""

    pass


class ClassificationManager:
    """Manages classification models and execution."""

    DEFAULT_TEXT_MODEL = "facebook/bart-large-mnli"
    DEFAULT_VISION_MODEL = "openai/clip-vit-base-patch16"

    def __init__(
        self,
        model_mapping: Optional[Dict[str, str]] = None,
        default_device: Optional[str] = None,
    ):
        """
        Initialize the ClassificationManager.

        Args:
            model_mapping: Optional dictionary mapping aliases ('text', 'vision') to model IDs.
            default_device: Default device ('cpu', 'cuda') if not specified in classify calls.
        """
        if not _CLASSIFICATION_AVAILABLE:
            raise ImportError(
                "Classification dependencies missing. "
                'Install with: pip install "natural-pdf[classification]"'
            )

        self.pipelines: Dict[Tuple[str, str], "Pipeline"] = (
            {}
        )  # Cache: (model_id, device) -> pipeline

        self.device = default_device
        logger.info(f"ClassificationManager initialized on device: {self.device}")

    def is_available(self) -> bool:
        """Check if required dependencies are installed."""
        return _CLASSIFICATION_AVAILABLE

    def _get_pipeline(self, model_id: str, using: str) -> "Pipeline":
        """Get or create a classification pipeline."""
        cache_key = f"{model_id}_{using}_{self.device}"
        if cache_key not in _PIPELINE_CACHE:
            logger.info(
                f"Loading {using} classification pipeline for model '{model_id}' on device '{self.device}'..."
            )
            start_time = time.time()
            try:
                task = (
                    "zero-shot-classification"
                    if using == "text"
                    else "zero-shot-image-classification"
                )
                _PIPELINE_CACHE[cache_key] = pipeline(task, model=model_id, device=self.device)
                end_time = time.time()
                logger.info(
                    f"Pipeline for '{model_id}' loaded in {end_time - start_time:.2f} seconds."
                )
            except Exception as e:
                logger.error(
                    f"Failed to load pipeline for model '{model_id}' (using: {using}): {e}",
                    exc_info=True,
                )
                raise ClassificationError(
                    f"Failed to load pipeline for model '{model_id}'. Ensure the model ID is correct and supports the {task} task."
                ) from e
        return _PIPELINE_CACHE[cache_key]

    def infer_using(self, model_id: str, using: Optional[str] = None) -> str:
        """Infers processing mode ('text' or 'vision') if not provided."""
        if using in ["text", "vision"]:
            return using

        # Simple inference based on common model names
        normalized_model_id = model_id.lower()
        if (
            "clip" in normalized_model_id
            or "vit" in normalized_model_id
            or "siglip" in normalized_model_id
        ):
            logger.debug(f"Inferred using='vision' for model '{model_id}'")
            return "vision"
        if (
            "bart" in normalized_model_id
            or "bert" in normalized_model_id
            or "mnli" in normalized_model_id
            or "xnli" in normalized_model_id
            or "deberta" in normalized_model_id
        ):
            logger.debug(f"Inferred using='text' for model '{model_id}'")
            return "text"

        # Fallback or raise error? Let's try loading text first, then vision.
        logger.warning(
            f"Could not reliably infer mode for '{model_id}'. Trying text, then vision pipeline loading."
        )
        try:
            self._get_pipeline(model_id, "text")
            logger.info(f"Successfully loaded '{model_id}' as a text model.")
            return "text"
        except Exception:
            logger.warning(f"Failed to load '{model_id}' as text model. Trying vision.")
            try:
                self._get_pipeline(model_id, "vision")
                logger.info(f"Successfully loaded '{model_id}' as a vision model.")
                return "vision"
            except Exception as e_vision:
                logger.error(
                    f"Failed to load '{model_id}' as either text or vision model.", exc_info=True
                )
                raise ClassificationError(
                    f"Cannot determine mode for model '{model_id}'. Please specify `using='text'` or `using='vision'`. Error: {e_vision}"
                )

    def classify_item(
        self,
        item_content: Union[str, Image.Image],
        labels: List[str],
        model_id: Optional[str] = None,
        using: Optional[str] = None,
        min_confidence: float = 0.0,
        multi_label: bool = False,
        **kwargs,
    ) -> ClassificationResult:  # Return ClassificationResult
        """Classifies a single item (text or image)."""

        # Determine model and engine type
        effective_using = using
        if model_id is None:
            # Try inferring based on content type
            if isinstance(item_content, str):
                effective_using = "text"
                model_id = self.DEFAULT_TEXT_MODEL
            elif isinstance(item_content, Image.Image):
                effective_using = "vision"
                model_id = self.DEFAULT_VISION_MODEL
            else:
                raise TypeError(f"Unsupported item_content type: {type(item_content)}")
        else:
            # Infer engine type if not given
            effective_using = self.infer_using(model_id, using)
            # Set default model if needed (though should usually be provided if engine known)
            if model_id is None:
                model_id = (
                    self.DEFAULT_TEXT_MODEL
                    if effective_using == "text"
                    else self.DEFAULT_VISION_MODEL
                )

        if not labels:
            raise ValueError("Labels list cannot be empty.")

        pipeline_instance = self._get_pipeline(model_id, effective_using)
        timestamp = datetime.now()
        parameters = {  # Store parameters used for this run
            "labels": labels,
            "model_id": model_id,
            "using": effective_using,
            "min_confidence": min_confidence,
            "multi_label": multi_label,
            **kwargs,
        }

        logger.debug(
            f"Classifying content (type: {type(item_content).__name__}) with model '{model_id}'"
        )
        try:
            # Handle potential kwargs for specific pipelines if needed
            # The zero-shot pipelines expect `candidate_labels`
            result_raw = pipeline_instance(
                item_content, candidate_labels=labels, multi_label=multi_label, **kwargs
            )
            logger.debug(f"Raw pipeline result: {result_raw}")

            # --- Process raw result into ClassificationResult --- #
            scores_list: List[CategoryScore] = []

            # Handle text pipeline format (dict with 'labels' and 'scores')
            if isinstance(result_raw, dict) and "labels" in result_raw and "scores" in result_raw:
                for label, score_val in zip(result_raw["labels"], result_raw["scores"]):
                    if score_val >= min_confidence:
                        try:
                            scores_list.append(CategoryScore(label, score_val))
                        except (ValueError, TypeError) as score_err:
                            logger.warning(
                                f"Skipping invalid score from text pipeline: label='{label}', score={score_val}. Error: {score_err}"
                            )
            # Handle vision pipeline format (list of dicts with 'label' and 'score')
            elif isinstance(result_raw, list) and all(
                isinstance(item, dict) and "label" in item and "score" in item
                for item in result_raw
            ):
                for item in result_raw:
                    score_val = item["score"]
                    label = item["label"]
                    if score_val >= min_confidence:
                        try:
                            scores_list.append(CategoryScore(label, score_val))
                        except (ValueError, TypeError) as score_err:
                            logger.warning(
                                f"Skipping invalid score from vision pipeline: label='{label}', score={score_val}. Error: {score_err}"
                            )
            else:
                logger.warning(
                    f"Unexpected raw result format from pipeline for model '{model_id}': {type(result_raw)}. Cannot extract scores."
                )
                # Return empty result?
                # scores_list = []

            # ClassificationResult now calculates top score/category internally
            result_obj = ClassificationResult(
                scores=scores_list,  # Pass the filtered list
                model_id=model_id,
                using=effective_using,
                parameters=parameters,
                timestamp=timestamp,
            )
            return result_obj
            # --- End Processing --- #

        except Exception as e:
            logger.error(f"Classification failed for model '{model_id}': {e}", exc_info=True)
            # Return an empty result object on failure?
            # return ClassificationResult(model_id=model_id, engine_type=engine_type, timestamp=timestamp, parameters=parameters, scores=[])
            raise ClassificationError(
                f"Classification failed using model '{model_id}'. Error: {e}"
            ) from e

    def classify_batch(
        self,
        item_contents: List[Union[str, Image.Image]],
        labels: List[str],
        model_id: Optional[str] = None,
        using: Optional[str] = None,
        min_confidence: float = 0.0,
        multi_label: bool = False,
        batch_size: int = 8,
        progress_bar: bool = True,
        **kwargs,
    ) -> List[ClassificationResult]:  # Return list of ClassificationResult
        """Classifies a batch of items (text or image) using the pipeline's batching."""
        if not item_contents:
            return []

        # Determine model and engine type (assuming uniform type in batch)
        first_item = item_contents[0]
        effective_using = using
        if model_id is None:
            if isinstance(first_item, str):
                effective_using = "text"
                model_id = self.DEFAULT_TEXT_MODEL
            elif isinstance(first_item, Image.Image):
                effective_using = "vision"
                model_id = self.DEFAULT_VISION_MODEL
            else:
                raise TypeError(f"Unsupported item_content type in batch: {type(first_item)}")
        else:
            effective_using = self.infer_using(model_id, using)
            if model_id is None:
                model_id = (
                    self.DEFAULT_TEXT_MODEL
                    if effective_using == "text"
                    else self.DEFAULT_VISION_MODEL
                )

        if not labels:
            raise ValueError("Labels list cannot be empty.")

        pipeline_instance = self._get_pipeline(model_id, effective_using)
        timestamp = datetime.now()  # Single timestamp for the batch run
        parameters = {  # Parameters for the whole batch
            "labels": labels,
            "model_id": model_id,
            "using": effective_using,
            "min_confidence": min_confidence,
            "multi_label": multi_label,
            "batch_size": batch_size,
            **kwargs,
        }

        logger.info(
            f"Classifying batch of {len(item_contents)} items with model '{model_id}' (batch size: {batch_size})"
        )
        batch_results_list: List[ClassificationResult] = []

        try:
            # Use pipeline directly for batching
            results_iterator = pipeline_instance(
                item_contents,
                candidate_labels=labels,
                multi_label=multi_label,
                batch_size=batch_size,
                **kwargs,
            )

            # Wrap with tqdm for progress if requested
            total_items = len(item_contents)
            if progress_bar:
                # Get the appropriate tqdm class
                tqdm_class = get_tqdm()
                results_iterator = tqdm_class(
                    results_iterator,
                    total=total_items,
                    desc=f"Classifying batch ({model_id})",
                    leave=False,  # Don't leave progress bar hanging
                )

            for raw_result in results_iterator:
                # --- Process each raw result (which corresponds to ONE input item) --- #
                scores_list: List[CategoryScore] = []
                try:
                    # Check for text format (dict with 'labels' and 'scores')
                    if (
                        isinstance(raw_result, dict)
                        and "labels" in raw_result
                        and "scores" in raw_result
                    ):
                        for label, score_val in zip(raw_result["labels"], raw_result["scores"]):
                            if score_val >= min_confidence:
                                try:
                                    scores_list.append(CategoryScore(label, score_val))
                                except (ValueError, TypeError) as score_err:
                                    logger.warning(
                                        f"Skipping invalid score from text pipeline batch: label='{label}', score={score_val}. Error: {score_err}"
                                    )
                    # Check for vision format (list of dicts with 'label' and 'score')
                    elif isinstance(raw_result, list):
                        for item in raw_result:
                            try:
                                score_val = item["score"]
                                label = item["label"]
                                if score_val >= min_confidence:
                                    scores_list.append(CategoryScore(label, score_val))
                            except (KeyError, ValueError, TypeError) as item_err:
                                logger.warning(
                                    f"Skipping invalid item in vision result list from batch: {item}. Error: {item_err}"
                                )
                    else:
                        logger.warning(
                            f"Unexpected raw result format in batch item from model '{model_id}': {type(raw_result)}. Cannot extract scores."
                        )

                except Exception as proc_err:
                    logger.error(
                        f"Error processing result item in batch: {proc_err}", exc_info=True
                    )
                    # scores_list remains empty for this item

                # --- Determine top category and score ---
                scores_list.sort(key=lambda s: s.score, reverse=True)
                top_category = scores_list[0].label
                top_score = scores_list[0].score
                # --- End Determine top category ---

                # Append result object for this item
                batch_results_list.append(
                    ClassificationResult(
                        scores=scores_list,  # Pass the full list, init will sort/filter
                        model_id=model_id,
                        using=effective_using,
                        timestamp=timestamp,  # Use same timestamp for batch
                        parameters=parameters,  # Use same params for batch
                    )
                )
                # --- End Processing --- #

            if len(batch_results_list) != total_items:
                logger.warning(
                    f"Batch classification returned {len(batch_results_list)} results, but expected {total_items}. Results might be incomplete or misaligned."
                )

            return batch_results_list

        except Exception as e:
            logger.error(f"Batch classification failed for model '{model_id}': {e}", exc_info=True)
            # Return list of empty results?
            # return [ClassificationResult(model_id=model_id, s=engine_type, timestamp=timestamp, parameters=parameters, scores=[]) for _ in item_contents]
            raise ClassificationError(
                f"Batch classification failed using model '{model_id}'. Error: {e}"
            ) from e
