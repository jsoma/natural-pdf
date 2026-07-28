"""Provider-backed helpers for document classification."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, List, Optional, Protocol, Sequence, Union, runtime_checkable

from PIL import Image

from natural_pdf.engine_provider import get_provider
from natural_pdf.engine_registry import register_builtin

from .pipelines import (
    DEFAULT_TEXT_MODEL,
    DEFAULT_VISION_MODEL,
    classify_batch_contents,
    classify_single,
    infer_using,
    is_classification_available,
    validate_classification_labels,
)
from .results import ClassificationResult

logger = logging.getLogger(__name__)


@runtime_checkable
class ClassificationEngine(Protocol):
    """Structural interface for classification engines."""

    def infer_using(self, model_id: Optional[str], using: Optional[str]) -> str: ...

    def default_model(self, using: str) -> str: ...

    def classify_item(
        self,
        *,
        item_content: Union[str, "Image.Image"],
        labels: List[str],
        model_id: Optional[str],
        using: Optional[str],
        min_confidence: float,
        multi_label: bool,
        **kwargs: Any,
    ) -> ClassificationResult: ...

    def classify_batch(
        self,
        *,
        contents: Sequence[Union[str, "Image.Image"]],
        labels: List[str],
        model_id: Optional[str],
        using: Optional[str],
        min_confidence: float,
        multi_label: bool,
        batch_size: int,
        progress_bar: bool,
        **kwargs: Any,
    ) -> List[ClassificationResult]: ...


class _DefaultClassificationEngine:
    def __init__(self) -> None:
        if not is_classification_available():
            raise ImportError(
                "Classification dependencies missing. "
                "Install with: pip install torch transformers"
            )
        self._device: Optional[str] = None

    def infer_using(self, model_id: Optional[str], using: Optional[str]) -> str:
        candidate = model_id or DEFAULT_TEXT_MODEL
        return infer_using(candidate, using, device=self._device)

    def default_model(self, using: str) -> str:
        return DEFAULT_TEXT_MODEL if using == "text" else DEFAULT_VISION_MODEL

    def classify_item(self, **kwargs):
        # An explicit device= from the caller wins over the engine default;
        # popping it also prevents a duplicate-keyword TypeError.
        device = kwargs.pop("device", self._device)
        return classify_single(device=device, **kwargs)

    def classify_batch(self, **kwargs):
        device = kwargs.pop("device", self._device)
        return classify_batch_contents(device=device, **kwargs)


def register_classification_engines(provider=None) -> None:
    def factory(**_opts):
        return _DefaultClassificationEngine()

    register_builtin(provider, "classification", "default", factory)


def get_classification_engine(context: Any, name: Optional[str] = None) -> ClassificationEngine:
    return _get_engine(context, name)


def _get_engine(context: Any, name: Optional[str] = None) -> ClassificationEngine:
    provider = get_provider()
    engine_name = (name or "default").strip().lower()
    engine = provider.get("classification", context=context, name=engine_name)
    return _validate_engine(engine, engine_name)


def _validate_engine(engine: Any, engine_name: str) -> ClassificationEngine:
    if not isinstance(engine, ClassificationEngine):
        raise TypeError(
            f"Classification engine '{engine_name}' does not implement the ClassificationEngine interface"
        )
    return engine


@contextmanager
def _checkout_engine(context: Any, name: Optional[str] = None):
    """Yield a provider-owned engine and release caller-owned instances."""

    engine_name = (name or "default").strip().lower()
    with get_provider().checkout("classification", context=context, name=engine_name) as engine:
        yield _validate_engine(engine, engine_name)


def run_classification_item(
    *,
    context: Any,
    content: Union[str, "Image.Image"],
    labels: List[str],
    model_id: Optional[str],
    using: Optional[str],
    min_confidence: float,
    multi_label: bool,
    engine_name: Optional[str] = None,
    engine: Optional[ClassificationEngine] = None,
    **kwargs,
) -> ClassificationResult:
    validate_classification_labels(labels)

    def invoke(resolved_engine: ClassificationEngine) -> ClassificationResult:
        return resolved_engine.classify_item(
            item_content=content,
            labels=labels,
            model_id=model_id,
            using=using,
            min_confidence=min_confidence,
            multi_label=multi_label,
            **kwargs,
        )

    # A pre-resolved engine remains owned by the caller.  High-level APIs use
    # this path after inspecting mode/default-model information, so they must
    # neither resolve a second transient instance nor close the one passed in.
    if engine is not None:
        return invoke(engine)

    # Direct public helper calls resolve through checkout so transient and
    # explicitly uncacheable engines are released after the invocation.
    with _checkout_engine(context, engine_name) as resolved_engine:
        return invoke(resolved_engine)


def run_classification_batch(
    *,
    context: Any,
    contents: Sequence[Union[str, "Image.Image"]],
    labels: List[str],
    model_id: Optional[str],
    using: Optional[str],
    min_confidence: float,
    multi_label: bool,
    batch_size: int,
    progress_bar: bool,
    engine_name: Optional[str] = None,
    engine: Optional[ClassificationEngine] = None,
    **kwargs,
) -> List[ClassificationResult]:
    validate_classification_labels(labels)

    def invoke(resolved_engine: ClassificationEngine) -> List[ClassificationResult]:
        return resolved_engine.classify_batch(
            contents=contents,
            labels=labels,
            model_id=model_id,
            using=using,
            min_confidence=min_confidence,
            multi_label=multi_label,
            batch_size=batch_size,
            progress_bar=progress_bar,
            **kwargs,
        )

    if engine is not None:
        return invoke(engine)

    with _checkout_engine(context, engine_name) as resolved_engine:
        return invoke(resolved_engine)


# Register built-in engine at import time. A failure here must surface
# immediately — swallowing it turns every later classify() call into an
# opaque LookupError.
register_classification_engines()


__all__ = [
    "register_classification_engines",
    "get_classification_engine",
    "run_classification_item",
    "run_classification_batch",
]
