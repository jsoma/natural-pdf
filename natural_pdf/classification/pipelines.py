import logging
import threading
import time
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Sequence, Union, cast

from PIL import Image
from tqdm.auto import tqdm

from natural_pdf.utils.optional_imports import is_available, require

from .results import CategoryScore, ClassificationResult

if TYPE_CHECKING:  # pragma: no cover - optional dependency types
    from transformers.pipelines.base import Pipeline

logger = logging.getLogger(__name__)

DEFAULT_TEXT_MODEL = "facebook/bart-large-mnli"
DEFAULT_VISION_MODEL = "openai/clip-vit-base-patch16"

_PIPELINE_CACHE: Dict[str, "Pipeline"] = {}
_CACHE_LOCK = threading.RLock()


class ClassificationError(Exception):
    """Raised when classification cannot be performed."""


def _check_classification_dependencies() -> bool:
    return is_available("torch") and is_available("transformers")


def is_classification_available() -> bool:
    """Return True when torch/transformers are installed."""
    return _check_classification_dependencies()


def _get_torch():
    return require("torch")


def _get_transformers_components():
    transformers_mod = require("transformers")
    return {
        "AutoModelForSequenceClassification": getattr(
            transformers_mod, "AutoModelForSequenceClassification"
        ),
        "AutoModelForZeroShotImageClassification": getattr(
            transformers_mod, "AutoModelForZeroShotImageClassification"
        ),
        "AutoTokenizer": getattr(transformers_mod, "AutoTokenizer"),
        "pipeline": getattr(transformers_mod, "pipeline"),
    }


def _load_pipeline(model_id: str, using: str, *, device: Optional[str]) -> "Pipeline":
    cache_key = f"{model_id}_{using}_{device}"
    with _CACHE_LOCK:
        if cache_key not in _PIPELINE_CACHE:
            logger.info(
                "Loading %s classification pipeline for model '%s' on device '%s'...",
                using,
                model_id,
                device,
            )
            start = time.time()
            task = (
                "zero-shot-classification" if using == "text" else "zero-shot-image-classification"
            )
            components = _get_transformers_components()
            pipeline_factory = components["pipeline"]
            try:
                _PIPELINE_CACHE[cache_key] = pipeline_factory(task, model=model_id, device=device)
            except Exception as exc:  # pragma: no cover - delegated to caller
                logger.error(
                    "Failed to load pipeline for model '%s' (mode=%s): %s",
                    model_id,
                    using,
                    exc,
                    exc_info=True,
                )
                raise ClassificationError(
                    f"Failed to load pipeline for model '{model_id}'. "
                    f"Ensure it supports the {task} task."
                ) from exc
            finally:
                end = time.time()
                logger.info(
                    "Pipeline for '%s' loaded in %.2f seconds.",
                    model_id,
                    end - start,
                )
        return _PIPELINE_CACHE[cache_key]


def infer_using(
    model_id: Optional[str], using: Optional[str], *, device: Optional[str] = None
) -> str:
    """Infer whether a model should run in text or vision mode."""
    if using in {"text", "vision"}:
        return using

    candidate = (model_id or "").lower()
    if any(token in candidate for token in ("clip", "vit", "siglip")):
        return "vision"
    if any(token in candidate for token in ("bart", "bert", "mnli", "xnli", "deberta")):
        return "text"

    # Fallback: try loading as text, then vision
    if model_id:
        logger.warning(
            "Could not infer mode for model '%s'. Attempting to load text then vision pipelines.",
            model_id,
        )
        try:
            _load_pipeline(model_id, "text", device=device)
            return "text"
        except Exception:
            logger.warning("Failed to load '%s' as text model; trying vision.", model_id)
            _load_pipeline(model_id, "vision", device=device)
            return "vision"

    raise ClassificationError(
        "Model identifier required when 'using' cannot be inferred automatically."
    )


def classify_single(
    *,
    item_content: Union[str, Image.Image],
    labels: List[str],
    model_id: Optional[str] = None,
    using: Optional[str] = None,
    min_confidence: float = 0.0,
    multi_label: bool = False,
    device: Optional[str] = None,
    **kwargs,
) -> ClassificationResult:
    """Classify a single piece of content."""

    if not _check_classification_dependencies():
        raise ImportError(
            'Classification dependencies missing. Install with `pip install "natural-pdf[classification]"`.'
        )

    if not labels:
        raise ValueError("Labels list cannot be empty.")

    selected_model = model_id
    effective_using = using

    if selected_model is None:
        if isinstance(item_content, str):
            selected_model = DEFAULT_TEXT_MODEL
            effective_using = "text"
        elif isinstance(item_content, Image.Image):
            selected_model = DEFAULT_VISION_MODEL
            effective_using = "vision"
        else:
            raise TypeError(f"Unsupported item_content type: {type(item_content)}")
    else:
        effective_using = infer_using(selected_model, effective_using, device=device)
        if selected_model is None:
            selected_model = (
                DEFAULT_TEXT_MODEL if effective_using == "text" else DEFAULT_VISION_MODEL
            )

    pipeline_instance = _load_pipeline(selected_model, effective_using, device=device)
    timestamp = datetime.now()
    parameters = {
        "labels": labels,
        "model_id": selected_model,
        "using": effective_using,
        "min_confidence": min_confidence,
        "multi_label": multi_label,
        **kwargs,
    }

    logger.debug(
        "Classifying content (type=%s) with model '%s'",
        type(item_content).__name__,
        selected_model,
    )

    try:
        raw_result = pipeline_instance(
            item_content,
            candidate_labels=labels,
            multi_label=multi_label,
            **kwargs,
        )
    except Exception as exc:
        logger.error("Classification failed for model '%s': %s", selected_model, exc, exc_info=True)
        raise ClassificationError(
            f"Classification failed using model '{selected_model}'. Error: {exc}"
        ) from exc

    scores_list: List[CategoryScore] = []

    if isinstance(raw_result, dict) and "labels" in raw_result and "scores" in raw_result:
        for label, score_val in zip(raw_result["labels"], raw_result["scores"]):
            if score_val >= min_confidence:
                try:
                    scores_list.append(CategoryScore(label, score_val))
                except (ValueError, TypeError) as err:
                    logger.warning(
                        "Skipping invalid score from text pipeline (label=%s, score=%s): %s",
                        label,
                        score_val,
                        err,
                    )
    elif isinstance(raw_result, list) and all(
        isinstance(item, dict) and "label" in item and "score" in item for item in raw_result
    ):
        for item in raw_result:
            score_val = item["score"]
            label = item["label"]
            if score_val >= min_confidence:
                try:
                    scores_list.append(CategoryScore(label, score_val))
                except (ValueError, TypeError) as err:
                    logger.warning(
                        "Skipping invalid score from vision pipeline (label=%s, score=%s): %s",
                        label,
                        score_val,
                        err,
                    )
    else:
        logger.warning(
            "Unexpected raw result format from pipeline for model '%s': %s",
            selected_model,
            type(raw_result),
        )

    return ClassificationResult(
        scores=scores_list,
        model_id=selected_model,
        using=effective_using,
        parameters=parameters,
        timestamp=timestamp,
    )


def classify_batch_contents(
    *,
    contents: Sequence[Union[str, Image.Image]],
    labels: List[str],
    model_id: Optional[str] = None,
    using: Optional[str] = None,
    min_confidence: float = 0.0,
    multi_label: bool = False,
    batch_size: int = 8,
    progress_bar: bool = False,
    device: Optional[str] = None,
    **kwargs,
) -> List[ClassificationResult]:
    """Classify a batch of content items."""

    if not _check_classification_dependencies():
        raise ImportError(
            'Classification dependencies missing. Install with `pip install "natural-pdf[classification]"`.'
        )

    if not labels:
        raise ValueError("Labels list cannot be empty.")
    if not contents:
        return []

    selected_model = model_id or (DEFAULT_TEXT_MODEL if using == "text" else DEFAULT_VISION_MODEL)
    effective_using = infer_using(selected_model, using, device=device)
    pipeline_instance = _load_pipeline(selected_model, effective_using, device=device)
    timestamp = datetime.now()
    parameters = {
        "labels": labels,
        "model_id": selected_model,
        "using": effective_using,
        "min_confidence": min_confidence,
        "multi_label": multi_label,
        "batch_size": batch_size,
        **kwargs,
    }

    try:
        iterator = cast(
            Iterable[Any],
            pipeline_instance(
                contents,
                candidate_labels=labels,
                multi_label=multi_label,
                batch_size=batch_size,
                **kwargs,
            ),
        )
    except Exception as exc:
        logger.error(
            "Batch classification failed for model '%s': %s",
            selected_model,
            exc,
            exc_info=True,
        )
        raise ClassificationError(
            f"Batch classification failed using model '{selected_model}'. Error: {exc}"
        ) from exc

    if progress_bar:
        iterator = tqdm(
            iterator,
            total=len(contents),
            desc=f"Classifying batch ({selected_model})",
            leave=False,
        )

    batch_results: List[ClassificationResult] = []

    for raw_result in iterator:
        scores_list: List[CategoryScore] = []
        try:
            if isinstance(raw_result, dict) and "labels" in raw_result and "scores" in raw_result:
                for label, score_val in zip(raw_result["labels"], raw_result["scores"]):
                    if score_val >= min_confidence:
                        scores_list.append(CategoryScore(label, score_val))
            elif isinstance(raw_result, list):
                for item in raw_result:
                    score_val = item.get("score")
                    label = item.get("label")
                    if label is None or score_val is None:
                        continue
                    if score_val >= min_confidence:
                        scores_list.append(CategoryScore(label, score_val))
            else:
                logger.warning(
                    "Unexpected raw result format in batch for model '%s': %s",
                    selected_model,
                    type(raw_result),
                )
        except Exception as err:
            logger.error("Error processing batch result: %s", err, exc_info=True)

        scores_list.sort(key=lambda s: s.score, reverse=True)
        batch_results.append(
            ClassificationResult(
                scores=scores_list,
                model_id=selected_model,
                using=effective_using,
                timestamp=timestamp,
                parameters=parameters,
            )
        )

    return batch_results


def cleanup_models(model_id: Optional[str] = None) -> int:
    """Release cached transformers pipelines."""
    if not _PIPELINE_CACHE:
        return 0

    cleaned = 0
    torch = None

    with _CACHE_LOCK:
        items = list(_PIPELINE_CACHE.items())
        if model_id:
            items = [item for item in items if model_id in item[0]]

    for cache_key, pipeline in items:
        model = getattr(pipeline, "model", None)
        if model is not None:
            try:
                if torch is None:
                    torch = _get_torch()
                move_to = getattr(model, "to", None)
                if callable(move_to):
                    move_to("cpu")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as exc:
                logger.debug("GPU cleanup failed for pipeline %s: %s", cache_key, exc)
        with _CACHE_LOCK:
            if cache_key in _PIPELINE_CACHE:
                _PIPELINE_CACHE.pop(cache_key, None)
                cleaned += 1

    return cleaned
