from __future__ import annotations

import logging
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, List, Optional

from PIL import Image

from natural_pdf.classification.classification_provider import (
    run_classification_item,
)
from natural_pdf.classification.pipelines import validate_classification_labels
from natural_pdf.classification.results import ClassificationResult
from natural_pdf.engine_provider import get_provider

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ClassificationCallOptions:
    """Options split at the host/content/engine service boundary.

    Classification accepts extension-engine keyword arguments, so the public
    API cannot enumerate every engine option.  It can still make ownership
    explicit: ``resolution`` belongs to content rendering and
    ``classification_engine`` belongs to engine selection; only the remaining
    options cross into the selected engine.
    """

    engine_name: Optional[str]
    content: dict[str, Any]
    engine: dict[str, Any]


def partition_classification_kwargs(kwargs: dict[str, Any]) -> ClassificationCallOptions:
    """Return independently owned option maps without mutating ``kwargs``."""

    engine_options = dict(kwargs)
    engine_name = engine_options.pop("classification_engine", None)
    content_options = {}
    if "resolution" in engine_options:
        content_options["resolution"] = engine_options.pop("resolution")
    return ClassificationCallOptions(engine_name, content_options, engine_options)


@contextmanager
def checkout_classification_engine(context: Any, engine_name: Optional[str] = None):
    """Yield a typed classification engine scoped to one classify call.

    Wraps :meth:`EngineProvider.checkout`, so transient-lifetime registrations
    have their ``cleanup()``/``close()`` hook invoked exactly once when the
    call finishes; cached lifetimes (context/singleton) are never cleaned here.
    """
    from natural_pdf.classification.classification_provider import ClassificationEngine

    name = (engine_name or "default").strip().lower()
    with get_provider().checkout("classification", context=context, name=name) as engine:
        if not isinstance(engine, ClassificationEngine):
            raise TypeError(
                f"Classification engine '{name}' does not implement the "
                "ClassificationEngine interface"
            )
        yield engine


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
        # Validate before touching host state or resolving an engine.  Besides
        # making single-item behavior consistent with the batch APIs, this
        # avoids constructing a potentially heavyweight transient engine for
        # a call that can never run.
        validate_classification_labels(labels)

        analyses = getattr(host, "analyses", None)
        if analyses is None:
            logger.warning("'analyses' attribute not found or is None. Initializing as empty dict.")
            host.analyses = {}
            analyses = host.analyses

        # Split the kwarg stream: content-extraction options go to the host's
        # content getter, everything else to the engine. Forwarding one stream
        # to both lets strays be silently swallowed by the pipeline while still
        # being recorded in the result's parameters.
        options = partition_classification_kwargs(kwargs)

        with checkout_classification_engine(host, options.engine_name) as engine_obj:
            chosen_mode = using
            content = None

            candidate_model = model or engine_obj.default_model("text")
            inferred_mode = engine_obj.infer_using(candidate_model, chosen_mode)
            chosen_mode = inferred_mode

            if chosen_mode == "text":
                try:
                    tentative_text = self._get_classification_content(
                        host, "text", **options.content
                    )
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
                content = self._get_classification_content(host, chosen_mode, **options.content)

            effective_model_id = model or engine_obj.default_model(chosen_mode)

            result_obj = run_classification_item(
                context=host,
                engine=engine_obj,
                content=content,
                labels=labels,
                model_id=effective_model_id,
                using=chosen_mode,
                min_confidence=min_confidence,
                multi_label=multi_label,
                **options.engine,
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
