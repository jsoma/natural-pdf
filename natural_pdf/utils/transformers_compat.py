"""Compatibility helpers for optional HuggingFace transformers APIs."""

from __future__ import annotations

from importlib import import_module
from typing import Any, Tuple


def get_pp_doclayout_v3_classes() -> Tuple[Any, Any]:
    """Return PP-DocLayout-V3 processor/model classes across transformers releases.

    Transformers has exposed the PP-DocLayout-V3 image processor as both
    ``PPDocLayoutV3ImageProcessor`` and ``PPDocLayoutV3ImageProcessorFast``.
    Prefer the unsuffixed class because the ``Fast`` name is deprecated in
    newer releases, while keeping the suffixed fallback for older releases.
    """
    try:
        transformers = import_module("transformers")
    except ImportError as exc:
        raise ImportError("transformers is required for PP-DocLayout-V3") from exc

    try:
        model_cls = getattr(transformers, "PPDocLayoutV3ForObjectDetection")
    except AttributeError as exc:
        raise ImportError("transformers does not provide PPDocLayoutV3ForObjectDetection") from exc

    processor_cls = getattr(transformers, "PPDocLayoutV3ImageProcessor", None)
    if processor_cls is None:
        processor_cls = getattr(transformers, "PPDocLayoutV3ImageProcessorFast", None)
    if processor_cls is None:
        raise ImportError(
            "transformers does not provide PPDocLayoutV3ImageProcessor "
            "or PPDocLayoutV3ImageProcessorFast"
        )

    return processor_cls, model_cls
