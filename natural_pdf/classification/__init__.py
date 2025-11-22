"""Classification helpers and registrations."""

from natural_pdf.engine_registry import register_classification_engine

from .accessors import ClassificationResultAccessorMixin
from .classification_provider import (
    register_classification_engines,
    run_classification_batch,
    run_classification_item,
)
from .pipelines import (
    ClassificationError,
    classify_batch_contents,
    classify_single,
    cleanup_models,
    infer_using,
    is_classification_available,
)
from .results import ClassificationResult

__all__ = [
    "ClassificationError",
    "ClassificationResult",
    "ClassificationResultAccessorMixin",
    "classify_batch_contents",
    "classify_single",
    "cleanup_models",
    "infer_using",
    "is_classification_available",
    "register_classification_engine",
    "register_classification_engines",
    "run_classification_item",
    "run_classification_batch",
]
