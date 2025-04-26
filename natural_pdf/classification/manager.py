import logging
import threading
from typing import Any, Dict, List, Optional, Union
import time

from PIL.Image import Image

# Try importing optional dependencies
_CLASSIFICATION_AVAILABLE = False

ClassificationError = ImportError # Default to ImportError if deps missing

try:
    import torch
    from transformers import pipeline, AutoConfig
    from transformers.pipelines.base import PipelineException
    _CLASSIFICATION_AVAILABLE = True

    # Redefine error if dependencies are available
    class ClassificationError(Exception):
        """Custom exception for classification failures."""
        pass

    class ModelNotFoundError(ClassificationError):
        """Raised when a specified model cannot be loaded."""
        pass

except ImportError:
    pass

logger = logging.getLogger(__name__)

# Default Model Aliases
DEFAULT_TEXT_MODEL = "facebook/bart-large-mnli"
DEFAULT_VISION_MODEL = "openai/clip-vit-base-patch32"

class ClassificationManager:
    """
    Manages loading and running text and vision classification models.
    Handles model caching and device management.
    """
    def __init__(self):
        if not _CLASSIFICATION_AVAILABLE:
             logger.warning(
                 "Classification dependencies (torch, transformers) not found. "
                 "Classification features will be unavailable. "
                 "Install with: pip install \"natural-pdf[classification]\""
             )
        self._loaded_models = {}
        self._model_locks = {}
        self._lock = threading.Lock()

    def _get_model_lock(self, model_id: str) -> threading.Lock:
        """Get or create a lock specific to a model ID."""
        with self._lock:
            if model_id not in self._model_locks:
                self._model_locks[model_id] = threading.Lock()
            return self._model_locks[model_id]

    def _get_pipeline(self, model_id: str, engine_type: str, device: Optional[Union[str, int]] = None, **kwargs):
        """Loads and returns a cached Hugging Face pipeline."""
        if not _CLASSIFICATION_AVAILABLE:
            raise ImportError(
                "Classification dependencies missing. Install with: pip install \"natural-pdf[classification]\""
            )

        # Resolve aliases
        if model_id == 'text':
            resolved_model_id = DEFAULT_TEXT_MODEL
            task = "zero-shot-classification"
        elif model_id == 'vision':
            resolved_model_id = DEFAULT_VISION_MODEL
            task = "zero-shot-image-classification"
        else:
            resolved_model_id = model_id
            # Determine task based on engine_type
            if engine_type == 'text':
                task = "zero-shot-classification"
            elif engine_type == 'vision':
                task = "zero-shot-image-classification"
            else:
                 raise ValueError(f"Invalid engine_type: {engine_type}")

        cache_key = (resolved_model_id, task, device)

        # Check cache first
        if cache_key in self._loaded_models:
            logger.debug(f"Using cached classification pipeline for key: {cache_key}")
            return self._loaded_models[cache_key]

        # Acquire model-specific lock before loading
        model_lock = self._get_model_lock(resolved_model_id)
        with model_lock:
            # Double-check cache after acquiring lock
            if cache_key in self._loaded_models:
                logger.debug(f"Using cached classification pipeline for key (post-lock): {cache_key}")
                return self._loaded_models[cache_key]

            logger.info(f"Loading classification pipeline: task='{task}', model='{resolved_model_id}', device='{device}'")
            try:
                pipeline_kwargs = {k: v for k, v in kwargs.items() if k in ['trust_remote_code']}
                loaded_pipeline = pipeline(
                    task,
                    model=resolved_model_id,
                    device=device,
                    **pipeline_kwargs
                )
                logger.info(f"Successfully loaded pipeline '{resolved_model_id}' on device '{loaded_pipeline.device}'")

                self._loaded_models[cache_key] = loaded_pipeline
                return loaded_pipeline

            except PipelineException as e:
                logger.error(f"Hugging Face pipeline error for model '{resolved_model_id}': {e}", exc_info=False)
                raise ClassificationError(f"Pipeline error for model '{resolved_model_id}': {e}") from e
            except Exception as e:
                 logger.error(f"Failed to load model '{resolved_model_id}': {e}", exc_info=False)
                 if "not found" in str(e) or isinstance(e, OSError):
                     raise ModelNotFoundError(f"Could not find or load model '{resolved_model_id}'. Ensure it exists and dependencies are installed.") from e
                 else:
                     raise ClassificationError(f"Failed to load model '{resolved_model_id}': {e}") from e

    def infer_engine_type(self, model_id: str) -> str:
        """Infers whether a model ID refers to a text or vision model."""
        if model_id == 'text': return 'text'
        if model_id == 'vision': return 'vision'

        if not _CLASSIFICATION_AVAILABLE:
             raise ImportError("Classification dependencies are required to infer model type.")

        logger.debug(f"Inferring engine type for model: {model_id}")
        try:
            config = AutoConfig.from_pretrained(model_id)
            model_type = getattr(config, "model_type", "").lower()
            architectures = [arch.lower() for arch in getattr(config, "architectures", [])]
            config_dict = config.to_dict()

            is_vision_type = any(vis_type in model_type for vis_type in ["clip", "vit", "beit", "swin", "resnet", "convnext", "siglip"])
            has_vision_architecture = any(
                any(arch.endswith(suffix) for suffix in ["visionmodel", "forimageclassification"])
                or arch == "vision-encoder-decoder"
                for arch in architectures
            )
            has_vision_config_key = "vision_config" in config_dict

            if is_vision_type or has_vision_architecture or has_vision_config_key:
                logger.debug(f"Inferred engine type for '{model_id}' as 'vision' (type: {is_vision_type}, arch: {has_vision_architecture}, key: {has_vision_config_key})")
                return "vision"
            else:
                logger.debug(f"Inferred engine type for '{model_id}' as 'text' (default). Type: '{model_type}', Arch: {architectures}")
                return "text"

        except Exception as e:
            logger.warning(f"Could not automatically determine engine type for model '{model_id}': {e}. Defaulting to 'text'. Specify 'engine_type' manually if this is incorrect.")
            return "text"

    def _run_inference(
        self,
        item_content: Union[str, Image],
        categories: List[str],
        model_id: str,
        engine_type: str,
        **kwargs,
    ) -> List[Dict[str, Union[str, float]]]:
        """Runs the inference using the appropriate pipeline, returns raw scores."""
        pipe = self._get_pipeline(model_id, engine_type, device=kwargs.get('device'), **kwargs)

        pipeline_input = item_content
    
        logger.debug(f"Running classification inference with pipeline: {pipe.model.name_or_path} on device {pipe.device}")
        if engine_type == 'text':
            logger.debug("Calling text pipeline for inference...")
            result = pipe(pipeline_input, candidate_labels=categories, multi_label=False)
            logger.debug("Text pipeline inference call completed.")
            output_scores = [{"label": label, "confidence": score} for label, score in zip(result['labels'], result['scores'])]
        elif engine_type == 'vision':
            logger.debug("Calling vision pipeline for inference...")
            result = pipe(pipeline_input, candidate_labels=categories)
            logger.debug("Vision pipeline inference call completed.")
            output_scores = [{"label": r["label"], "confidence": r["score"]} for r in result]
        else:
            raise ValueError(f"Unsupported engine_type: {engine_type}")

        # Sort scores
        output_scores.sort(key=lambda x: x['confidence'], reverse=True)
        return output_scores

    def get_classification_result(
        self,
        item_content: Union[str, Image],
        categories: List[str],
        model_id: str,
        engine_type: str,
        min_confidence: float = 0.0,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Performs classification, filters results, and returns the formatted result dictionary.

        Args:
            item_content: The text (str) or image (PIL.Image) to classify.
            categories: List of candidate category labels.
            model_id: Model identifier (alias or HF ID).
            engine_type: 'text' or 'vision'.
            min_confidence: Minimum confidence score for inclusion.
            **kwargs: Additional arguments for the pipeline (e.g., device).

        Returns:
            The structured classification result dictionary.
        """
        raw_scores = self._run_inference(
            item_content=item_content,
            categories=categories,
            model_id=model_id,
            engine_type=engine_type,
            **kwargs
        )

        # Filter scores based on threshold
        filtered_scores = [
            score for score in raw_scores
            if score['confidence'] >= min_confidence
        ]

        result_dict = {
            'model': model_id,
            'engine_type': engine_type,
            'categories_used': categories,
            'scores': filtered_scores,
            'timestamp': time.time()
        }
        return result_dict

    def classify_batch(
        self,
        item_contents: List[Union[str, Image]],
        categories: List[str],
        model_id: str,
        engine_type: str,
        min_confidence: float = 0.0,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Performs batch classification on homogeneous items (all text or all images).

        Args:
            item_contents: List of text (str) or images (PIL.Image) to classify.
            categories: List of candidate category labels.
            model_id: Model identifier (alias or HF ID).
            engine_type: 'text' or 'vision'.
            min_confidence: Minimum confidence score for inclusion in results.
            **kwargs: Additional arguments for the pipeline (e.g., device).

        Returns:
            A list of structured classification result dictionaries, one for each item.
        """
        if not isinstance(item_contents, list) or not item_contents:
             raise TypeError("item_contents must be a non-empty list.")
        
        # Basic type check
        if engine_type == 'text' and not all(isinstance(item, str) for item in item_contents):
             logger.warning("Batch contains non-string items but engine_type is 'text'.")
        elif engine_type == 'vision' and not all(isinstance(item, Image) for item in item_contents):
             logger.warning("Batch contains non-Image items but engine_type is 'vision'.")

        pipe = self._get_pipeline(model_id, engine_type, device=kwargs.get('device'), **kwargs)
        current_timestamp = time.time()

        logger.debug(f"Running BATCH classification inference ({len(item_contents)} items) with pipeline: {pipe.model.name_or_path} on device {pipe.device}")

        try:
            if engine_type == 'text':
                batch_results = pipe(item_contents, candidate_labels=categories, multi_label=False)
            elif engine_type == 'vision':
                batch_results = pipe(item_contents, candidate_labels=categories)
            else:
                raise ValueError(f"Unsupported engine_type: {engine_type}")
        except Exception as e:
             logger.error(f"Error during batch pipeline execution for model '{model_id}': {e}", exc_info=True)
             raise ClassificationError(f"Error during batch inference for model '{model_id}': {e}") from e

        logger.debug("Batch inference call completed.")

        final_results = []
        for i, item_result in enumerate(batch_results):
            if engine_type == 'text':
                raw_scores = [{"label": label, "confidence": score}
                              for label, score in zip(item_result['labels'], item_result['scores'])]
            elif engine_type == 'vision':
                 raw_scores = [{"label": r["label"], "confidence": r["score"]} for r in item_result]
            
            # Sort and filter individual item scores
            raw_scores.sort(key=lambda x: x['confidence'], reverse=True)
            filtered_scores = [
                score for score in raw_scores
                if score['confidence'] >= min_confidence
            ]

            result_dict = {
                'model': model_id,
                'engine_type': engine_type,
                'categories_used': categories,
                'scores': filtered_scores,
                'timestamp': current_timestamp
            }
            final_results.append(result_dict)

        return final_results

    # --- Batch classification (Placeholder - can be complex) ---
    # For now, the mixin/collection calls classify_item repeatedly.
    # True batching would require gathering all content first,
    # then calling the pipeline with a list, and mapping results back.
    # This adds complexity around handling mixed text/image batches, etc.
    # Let's defer optimizing batching within the manager itself.

    # def classify_batch(...):
    #     pass 