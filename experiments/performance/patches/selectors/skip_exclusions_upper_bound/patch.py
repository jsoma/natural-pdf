"""Disable selector exclusion filtering to estimate its upper-bound cost."""

from contextlib import contextmanager

METADATA = {
    "track": "selectors",
    "candidate": "skip_exclusions_upper_bound",
    "cache_only": False,
    "hypothesis": "A structural exclusion-mask optimization has bounded value close to the no-exclusion timing.",
}


@contextmanager
def install():
    from natural_pdf.services.selector_service import SelectorService

    original_find_all_page = SelectorService._find_all_page

    def patched_find_all_page(self, page, **kwargs):
        kwargs = dict(kwargs)
        kwargs["apply_exclusions"] = False
        return original_find_all_page(self, page, **kwargs)

    SelectorService._find_all_page = patched_find_all_page
    try:
        yield
    finally:
        SelectorService._find_all_page = original_find_all_page
