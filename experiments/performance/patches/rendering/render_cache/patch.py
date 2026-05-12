"""Cache identical RenderingService.render calls for repeated-render comparison."""

from contextlib import contextmanager

METADATA = {
    "track": "rendering",
    "candidate": "render_cache",
    "cache_only": True,
    "hypothesis": "Repeated render calls in extraction workflows can be isolated with a render cache.",
}


def _make_hashable(value):
    if isinstance(value, dict):
        return tuple(sorted((key, _make_hashable(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_make_hashable(item) for item in value)
    try:
        hash(value)
    except Exception:
        return repr(value)
    return value


@contextmanager
def install():
    from natural_pdf.services.rendering_service import RenderingService

    original_render = RenderingService.render
    cache = {}

    def patched_render(self, host, **kwargs):
        key = (id(host), _make_hashable(kwargs))
        if key not in cache:
            cache[key] = original_render(self, host, **kwargs)
        return cache[key]

    RenderingService.render = patched_render
    try:
        yield
    finally:
        RenderingService.render = original_render
        cache.clear()
