"""Fast-path simple selector branches with element arrays."""

from contextlib import contextmanager

import numpy as np

from experiments.performance import vector_metrics as metrics
from experiments.performance.vector_helpers import element_array

METADATA = {
    "track": "vector",
    "candidate": "simple_selector_fast_path",
    "cache_only": False,
    "hypothesis": "Common simple selectors can filter/sort with arrays and fall back for complex selectors.",
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


def _simple_mask(selector_obj, arr):
    mask = arr.valid.copy()
    for pseudo in selector_obj.get("pseudo_classes", []):
        name = pseudo.get("name")
        if pseudo.get("args") is not None or name not in _BOOLEAN_PSEUDOS:
            return None
        mask &= getattr(arr, _BOOLEAN_PSEUDOS[name])

    for attr in selector_obj.get("attributes", []):
        name = attr.get("name")
        op = attr.get("op")
        if name not in _BOOLEAN_PSEUDOS.values() or op not in {"=", "=="}:
            return None
        expected = _bool_value(attr.get("value"))
        values = getattr(arr, name)
        mask &= values if expected else ~values

    return mask


def _can_fast_path(selector_obj):
    if selector_obj.get("filters"):
        return False
    if selector_obj.get("relational_pseudos") or selector_obj.get("post_pseudos"):
        return False
    for attr in selector_obj.get("attributes", []):
        value = attr.get("value")
        if isinstance(value, dict) and value.get("type") == "aggregate":
            return False
    for pseudo in selector_obj.get("pseudo_classes", []):
        if pseudo.get("name") not in _BOOLEAN_PSEUDOS or pseudo.get("args") is not None:
            return False
    return True


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
    if branch_type not in {"any", "text", "rect", "line", "image", "region", "form_cell"}:
        metrics.count("vector.simple_selector.fallback_type")
        return _ORIGINAL(
            host,
            selector_obj,
            elements,
            selector_kwargs=selector_kwargs,
            selector_type=selector_type,
            logger=logger,
        )
    if not _can_fast_path(selector_obj):
        metrics.count("vector.simple_selector.fallback_complex")
        return _ORIGINAL(
            host,
            selector_obj,
            elements,
            selector_kwargs=selector_kwargs,
            selector_type=selector_type,
            logger=logger,
        )

    element_list = list(elements)
    arr = element_array(element_list, label="vector.simple_selector.elements")
    mask = _simple_mask(selector_obj, arr)
    if mask is None:
        metrics.count("vector.simple_selector.fallback_mask")
        return _ORIGINAL(
            host,
            selector_obj,
            element_list,
            selector_kwargs=selector_kwargs,
            selector_type=selector_type,
            logger=logger,
        )

    selected = np.flatnonzero(mask)
    if (selector_kwargs or {}).get("reading_order", True) and len(selected) > 1:
        order = np.lexsort((arr.x0[selected], arr.top[selected]))
        selected = selected[order]
        metrics.byte_count(
            "vector.simple_selector.sort_index_bytes", selected.nbytes + order.nbytes
        )

    metrics.count("vector.simple_selector.fast_path")
    metrics.count("vector.simple_selector.input_elements", len(element_list))
    metrics.count("vector.simple_selector.output_elements", len(selected))
    metrics.byte_count("vector.simple_selector.mask_bytes", mask.nbytes)
    return [element_list[int(index)] for index in selected]


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
