"""Fast-path simple page selectors with reusable page arrays."""

from contextlib import contextmanager

import numpy as np

from experiments.performance import vector_metrics as metrics
from experiments.performance.vector_helpers import PageArrayView

METADATA = {
    "track": "vector",
    "candidate": "simple_selector_page_store",
    "cache_only": False,
    "hypothesis": "Simple page selectors need reusable page arrays; rebuilding per selector is too expensive.",
}

_TYPE_TO_ARRAY = {
    "text": "words",
    "rect": "rects",
    "line": "lines",
    "image": "images",
    "region": "regions",
}

_BOOLEAN_PSEUDOS = {
    "highlight": "highlight",
    "highlighted": "highlight",
    "underline": "underline",
    "underlined": "underline",
    "strike": "strike",
    "strikethrough": "strike",
    "strikeout": "strike",
}


def _bool_value(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _can_fast_path(selector_obj):
    if selector_obj.get("filters"):
        return False
    if selector_obj.get("relational_pseudos") or selector_obj.get("post_pseudos"):
        return False
    for pseudo in selector_obj.get("pseudo_classes", []):
        if pseudo.get("name") not in _BOOLEAN_PSEUDOS or pseudo.get("args") is not None:
            return False
    for attr in selector_obj.get("attributes", []):
        if attr.get("name") not in {"strike", "underline", "highlight"}:
            return False
        if attr.get("op") not in {"=", "=="}:
            return False
    return True


def _mask(selector_obj, arr):
    mask = arr.valid.copy()
    for pseudo in selector_obj.get("pseudo_classes", []):
        mask &= getattr(arr, _BOOLEAN_PSEUDOS[pseudo["name"]])
    for attr in selector_obj.get("attributes", []):
        expected = _bool_value(attr.get("value"))
        values = getattr(arr, attr["name"])
        mask &= values if expected else ~values
    return mask


def _patched_execute_selector_branch(
    host,
    selector_obj,
    elements,
    *,
    selector_kwargs=None,
    selector_type=None,
    logger=None,
):
    branch_type = (selector_type or selector_obj.get("type", "any")).lower()
    array_name = _TYPE_TO_ARRAY.get(branch_type)
    if array_name is None or not _can_fast_path(selector_obj):
        metrics.count("vector.simple_selector_page_store.fallback")
        return _ORIGINAL(
            host,
            selector_obj,
            elements,
            selector_kwargs=selector_kwargs,
            selector_type=selector_type,
            logger=logger,
        )

    from natural_pdf.core.page import Page

    if not isinstance(host, Page):
        metrics.count("vector.simple_selector_page_store.fallback_non_page")
        return _ORIGINAL(
            host,
            selector_obj,
            elements,
            selector_kwargs=selector_kwargs,
            selector_type=selector_type,
            logger=logger,
        )

    arr = PageArrayView.for_page(host).elements(array_name)
    mask = _mask(selector_obj, arr)
    selected = np.flatnonzero(mask)
    if (selector_kwargs or {}).get("reading_order", True) and len(selected) > 1:
        order = np.lexsort((arr.x0[selected], arr.top[selected]))
        selected = selected[order]
        metrics.byte_count(
            "vector.simple_selector_page_store.sort_index_bytes",
            selected.nbytes + order.nbytes,
        )

    metrics.count("vector.simple_selector_page_store.fast_path")
    metrics.count("vector.simple_selector_page_store.output_elements", len(selected))
    metrics.byte_count("vector.simple_selector_page_store.mask_bytes", mask.nbytes)
    return [arr.items[int(index)] for index in selected]


@contextmanager
def install():
    import natural_pdf.core.selector_utils as selector_utils

    global _ORIGINAL
    _ORIGINAL = selector_utils.execute_selector_branch
    selector_utils.execute_selector_branch = _patched_execute_selector_branch
    try:
        yield
    finally:
        selector_utils.execute_selector_branch = _ORIGINAL
