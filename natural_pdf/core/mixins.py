from __future__ import annotations

from typing import Any, Iterable, List, Optional, Set


class ContextResolverMixin:
    """Shared helpers for objects that resolve config/manager values via a graph of contexts."""

    _context_parent_attrs = (
        "parent_region",
        "parent",
        "flow",
        "page",
        "pdf",
        "_parent",
    )

    def _context_resolution_roots(self) -> Iterable[Any]:
        """Return the initial objects used for resolution (self by default)."""
        return (self,)

    def _context_resolution_chain(self) -> Iterable[Any]:
        seen: Set[int] = set()
        queue: List[Any] = list(self._context_resolution_roots())

        while queue:
            obj = queue.pop(0)
            if obj is None:
                continue
            marker = id(obj)
            if marker in seen:
                continue
            seen.add(marker)
            yield obj

            parent = self._resolve_parent_context(obj)
            if parent is not None:
                queue.append(parent)

    @classmethod
    def _resolve_parent_context(cls, obj: Any) -> Optional[Any]:
        for attr in cls._context_parent_attrs:
            if hasattr(obj, attr):
                parent = getattr(obj, attr)
                if parent is not None:
                    return parent
        return None

    def get_config(self, key: str, default: Any = None, *, scope: str = "region") -> Any:
        sentinel = object()
        for context in self._context_resolution_chain():
            if context is self:
                continue
            getter = getattr(context, "get_config", None)
            if callable(getter):
                try:
                    value = getter(key, sentinel, scope=scope)
                except ValueError:
                    continue
                if value is not sentinel:
                    return value
        return default

    def get_manager(self, name: str) -> Any:
        for context in self._context_resolution_chain():
            if context is self:
                continue
            manager_getter = getattr(context, "get_manager", None)
            if callable(manager_getter):
                return manager_getter(name)
        raise AttributeError(f"Cannot resolve manager '{name}' for {self.__class__.__name__}")


class SinglePageContextMixin(ContextResolverMixin):
    """Shared helpers for objects tied to a single PDF page."""

    def _context_page(self):
        """Return the owning Page instance."""
        raise NotImplementedError

    def _context_pdf(self):
        page = self._context_page()
        return getattr(page, "pdf", getattr(page, "_parent", None))

    def _context_region_config(self, key: str, sentinel: object) -> Any:
        return sentinel

    def _context_resolution_roots(self) -> Iterable[Any]:
        page = self._context_page()
        roots: List[Any] = [page]
        pdf_obj = self._context_pdf()
        if pdf_obj is not None:
            roots.append(pdf_obj)
        return roots

    @property
    def pages(self):
        return (self._context_page(),)

    def get_highlighter(self):
        from natural_pdf.core.highlighter_utils import resolve_highlighter

        return resolve_highlighter(self._context_page())

    def get_config(self, key: str, default: Any = None, *, scope: str = "region") -> Any:
        if scope not in {"region", "page", "pdf"}:
            raise ValueError(f"Unsupported configuration scope: {scope}")

        sentinel = object()
        if scope == "region":
            region_value = self._context_region_config(key, sentinel)
            if region_value is not sentinel:
                return region_value

        page = self._context_page()
        page_cfg = getattr(page, "_config", None)
        if page_cfg is not None and scope in {"region", "page"} and key in page_cfg:
            return page_cfg[key]

        pdf_obj = self._context_pdf()
        if pdf_obj is not None and hasattr(pdf_obj, "_config"):
            pdf_cfg = pdf_obj._config
            if scope in {"region", "page", "pdf"} and key in pdf_cfg:
                return pdf_cfg[key]

        return super().get_config(key, default, scope=scope)

    def get_manager(self, name: str):
        pdf_obj = self._context_pdf()
        if pdf_obj is None:
            raise AttributeError(f"Cannot resolve manager '{name}' without PDF context")
        return pdf_obj.get_manager(name)
