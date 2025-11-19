"""Provider-backed helpers for document classification."""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Sequence, Union

from PIL import Image

from natural_pdf.engine_provider import get_provider
from natural_pdf.engine_registry import register_builtin, register_classification_engine

from .pipelines import (
    DEFAULT_TEXT_MODEL,
    DEFAULT_VISION_MODEL,
    classify_batch_contents,
    classify_single,
    infer_using,
    is_classification_available,
)
from .results import ClassificationResult

logger = logging.getLogger(__name__)


class ClassificationEngine:
    """Protocol-like wrapper for classification engines."""

    def infer_using(self, model_id: Optional[str], using: Optional[str]) -> str:
        raise NotImplementedError

    def default_model(self, using: str) -> str:
        raise NotImplementedError

    def classify_item(
        self,
        *,
        item_content: Union[str, "Image.Image"],
        labels: List[str],
        model_id: Optional[str],
        using: Optional[str],
        min_confidence: float,
        multi_label: bool,
        **kwargs,
    ) -> ClassificationResult:
        raise NotImplementedError

    def classify_batch(
        self,
        *,
        item_contents: Sequence[Union[str, "Image.Image"]],
        labels: List[str],
        model_id: Optional[str],
        using: Optional[str],
        min_confidence: float,
        multi_label: bool,
        batch_size: int,
        progress_bar: bool,
        **kwargs,
    ) -> List[ClassificationResult]:
        raise NotImplementedError


class _DefaultClassificationEngine(ClassificationEngine):
    def __init__(self) -> None:
        if not is_classification_available():
            raise ImportError(
                "Classification dependencies missing. "
                'Install with: pip install "natural-pdf[ai]"'
            )
        self._device: Optional[str] = None

    def infer_using(self, model_id: Optional[str], using: Optional[str]) -> str:
        candidate = model_id or DEFAULT_TEXT_MODEL
        return infer_using(candidate, using, device=self._device)

    def default_model(self, using: str) -> str:
        return DEFAULT_TEXT_MODEL if using == "text" else DEFAULT_VISION_MODEL

    def classify_item(self, **kwargs):
        return classify_single(device=self._device, **kwargs)

    def classify_batch(self, **kwargs):
        return classify_batch_contents(device=self._device, **kwargs)


def register_classification_engines(provider=None) -> None:
    def factory(**_opts):
        return _DefaultClassificationEngine()

    register_builtin(provider, "classification", "default", factory)


def get_classification_engine(context: Any, name: Optional[str] = None) -> ClassificationEngine:
    return _get_engine(context, name)


def _get_engine(context: Any, name: Optional[str] = None) -> ClassificationEngine:
    provider = get_provider()
    engine_name = (name or "default").strip().lower()
    return provider.get("classification", context=context, name=engine_name)


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
    **kwargs,
) -> ClassificationResult:
    engine = _get_engine(context, engine_name)
    return engine.classify_item(
        item_content=content,
        labels=labels,
        model_id=model_id,
        using=using,
        min_confidence=min_confidence,
        multi_label=multi_label,
        **kwargs,
    )


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
    **kwargs,
) -> List[ClassificationResult]:
    engine = _get_engine(context, engine_name)
    return engine.classify_batch(
        item_contents=contents,
        labels=labels,
        model_id=model_id,
        using=using,
        min_confidence=min_confidence,
        multi_label=multi_label,
        batch_size=batch_size,
        progress_bar=progress_bar,
        **kwargs,
    )


try:  # Register built-in engine immediately
    register_classification_engines()
except Exception:  # pragma: no cover
    logger.exception("Failed to register classification engines")


__all__ = [
    "register_classification_engines",
    "get_classification_engine",
    "run_classification_item",
    "run_classification_batch",
]
