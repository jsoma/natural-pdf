from __future__ import annotations

import logging
import warnings
from typing import Any, List, Optional

from PIL import Image

from natural_pdf.classification.classification_provider import (
    get_classification_engine,
    run_classification_item,
)
from natural_pdf.classification.results import ClassificationResult

logger = logging.getLogger(__name__)


class ClassificationService:
    """Shared classification helpers extracted from ClassificationMixin."""

    def __init__(self, context):
        self._context = context

    def classify(
        self,
        host,
        labels: List[str],
        *,
        model: Optional[str] = None,
        using: Optional[str] = None,
        min_confidence: float = 0.0,
        analysis_key: str = "classification",
        multi_label: bool = False,
        **kwargs,
    ) -> ClassificationResult:
        analyses = getattr(host, "analyses", None)
        if analyses is None:
            logger.warning("'analyses' attribute not found or is None. Initializing as empty dict.")
            host.analyses = {}
            analyses = host.analyses

        engine_obj = get_classification_engine(host, kwargs.pop("classification_engine", None))

        # Split the kwarg stream: content-extraction options go to the host's
        # content getter, everything else to the engine. Forwarding one stream
        # to both lets strays be silently swallowed by the pipeline while still
        # being recorded in the result's parameters.
        content_kwargs = {}
        if "resolution" in kwargs:
            content_kwargs["resolution"] = kwargs.pop("resolution")

        chosen_mode = using
        content = None

        candidate_model = model or engine_obj.default_model("text")
        inferred_mode = engine_obj.infer_using(candidate_model, chosen_mode)
        chosen_mode = inferred_mode

        if chosen_mode == "text":
            try:
                tentative_text = self._get_classification_content(host, "text", **content_kwargs)
                if tentative_text and not (
                    isinstance(tentative_text, str) and tentative_text.isspace()
                ):
                    content = tentative_text
                else:
                    raise ValueError("Empty text")
            except ValueError as exc:
                if not self._is_empty_text_error(exc):
                    raise RuntimeError(
                        "Failed to extract text content for classification while using='text'."
                    ) from exc
                warnings.warn(
                    "No text found for classification; falling back to vision model. "
                    "Pass using='vision' explicitly to silence this message.",
                    UserWarning,
                )
                chosen_mode = "vision"
            except Exception as exc:
                raise RuntimeError(
                    "Failed to extract text content for classification while using='text'."
                ) from exc

        if content is None:
            if chosen_mode is None:
                chosen_mode = "vision"
            content = self._get_classification_content(host, chosen_mode, **content_kwargs)

        effective_model_id = model or engine_obj.default_model(chosen_mode)

        result_obj = run_classification_item(
            context=host,
            content=content,
            labels=labels,
            model_id=effective_model_id,
            using=chosen_mode,
            min_confidence=min_confidence,
            multi_label=multi_label,
            **kwargs,
        )

        analyses[analysis_key] = result_obj
        logger.debug("Stored classification result under key '%s': %s", analysis_key, result_obj)
        return result_obj

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _is_empty_text_error(exc: ValueError) -> bool:
        """Return whether a host's ``ValueError`` denotes genuinely empty text.

        Hosts historically signal an absent text layer with ``ValueError``.
        Restricting the fallback to those known absence messages prevents a
        broken text extractor from being mistaken for a scanned document.
        """

        empty_messages = {
            "empty text",
            "cannot classify element with 'text' model: no text content found.",
            "cannot classify page with 'text' model: no text content found.",
            "cannot classify region with 'text' model: no text content found.",
            "pdf contains no extractable text for classification.",
        }
        current: Optional[BaseException] = exc
        while current is not None:
            message = str(current).strip().casefold()
            if message in empty_messages:
                return True
            current = current.__cause__
        return False

    @staticmethod
    def _get_classification_content(host, model_type: str, **kwargs) -> Any:
        getter = getattr(host, "_get_classification_content", None)
        if not callable(getter):
            raise NotImplementedError(
                f"{type(host).__name__} must implement _get_classification_content()."
            )
        content = getter(model_type=model_type, **kwargs)
        if model_type == "text" and isinstance(content, Image.Image):
            raise ValueError("Expected text content but received an image.")
        return content
