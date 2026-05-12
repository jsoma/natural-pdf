"""Cache exact SelectorService.find_all calls for repeated-work comparison."""

from contextlib import contextmanager

METADATA = {
    "track": "selectors",
    "candidate": "exact_query_cache",
    "cache_only": True,
    "hypothesis": "Exact selector caching improves repeated-query workloads but not cold single-use flows.",
}


def _make_hashable(value):
    if isinstance(value, dict):
        return tuple(sorted((key, _make_hashable(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_make_hashable(item) for item in value)
    if isinstance(value, set):
        return tuple(sorted(_make_hashable(item) for item in value))
    try:
        hash(value)
    except Exception:
        return repr(value)
    return value


@contextmanager
def install():
    from natural_pdf.elements.element_collection import ElementCollection
    from natural_pdf.services.selector_service import SelectorService

    original_find_all = SelectorService.find_all
    cache = {}

    def patched_find_all(self, host, **kwargs):
        host_key = id(host)
        context = getattr(host, "_context", self._context)
        key = (host_key, _make_hashable(kwargs))
        if key in cache:
            return ElementCollection(list(cache[key]), context=context)
        result = original_find_all(self, host, **kwargs)
        elements = list(getattr(result, "elements", result or []))
        cache[key] = elements
        return result

    SelectorService.find_all = patched_find_all
    try:
        yield
    finally:
        SelectorService.find_all = original_find_all
        cache.clear()
