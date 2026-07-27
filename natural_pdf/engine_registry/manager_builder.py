"""Data-driven builder shared by per-capability engine manager modules.

``natural_pdf/layout/layout_manager.py`` and
``natural_pdf/checkbox/checkbox_manager.py`` used to be copy-paste twins:
engine-definition tuples, alias handling, options-class lookups, instance
creation with ``is_available()`` checks plus install hints, and a
``register_builtin`` loop. :class:`EngineManagerSpec` captures that shape
once so each manager module is reduced to its data (engine defs, aliases,
install hints) plus thin module-level wrappers.
"""

from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Set, Tuple, Type

from natural_pdf.engine_provider import get_provider
from natural_pdf.engine_registry import register_builtin

# (engine_name, lazy class factory OR class, options_class)
EngineDef = Tuple[str, Any, type]


@dataclass
class EngineManagerSpec:
    """One capability's engine roster and the behavior derived from it.

    Parameters
    ----------
    capability:
        EngineProvider capability string (e.g. ``"layout"``).
    label:
        Human-readable capitalized name used in error messages
        (e.g. ``"Layout"`` -> "Layout engine 'x' is not available").
    engine_defs:
        Sequence of ``(name, class_factory, options_class)`` tuples.
        ``class_factory`` may be the class itself or a zero-arg callable
        returning it (for lazy imports).
    deprecated_aliases:
        Mapping of deprecated engine name -> canonical name. Aliases are
        registered with the provider (sharing the canonical options class),
        skipped by :meth:`engine_name_for_options`, and warn once on use.
    install_hints:
        Mapping of engine name -> pip install hint appended to the
        "not available" error.
    fallback_options_class:
        Options class used for an alias whose canonical engine is missing
        from ``engine_defs``.
    logger:
        Logger for registration/creation messages; defaults to this module's.
    """

    capability: str
    label: str
    engine_defs: Sequence[EngineDef]
    deprecated_aliases: Mapping[str, str] = field(default_factory=dict)
    install_hints: Mapping[str, str] = field(default_factory=dict)
    fallback_options_class: type = object
    logger: logging.Logger = field(default_factory=lambda: logging.getLogger(__name__))
    _deprecation_warned: Set[str] = field(default_factory=set)

    # -- Options-class <-> engine-name mapping ------------------------------

    def engine_name_for_options(self, options: Any) -> Optional[str]:
        """Return the engine name whose options_class matches *options*, or None.

        Deprecated aliases are skipped so the canonical name is always returned.
        """
        provider = get_provider()
        engines = provider.list(self.capability).get(self.capability, ())
        for name in engines:
            if name in self.deprecated_aliases:
                continue
            meta = provider.get_metadata(self.capability, name)
            if meta and isinstance(options, meta.get("options_class", type(None))):
                return name
        return None

    def get_options_class_for_engine(self, name: str) -> Optional[type]:
        """Return the options class registered for *name*, or None."""
        provider = get_provider()
        meta = provider.get_metadata(self.capability, name)
        if meta:
            return meta.get("options_class")
        return None

    # -- Instance creation ---------------------------------------------------

    def _resolve_engine_class(self, engine_name: str) -> type:
        """Resolve the engine class for a name by querying registered metadata."""
        provider = get_provider()
        meta = provider.get_metadata(self.capability, engine_name)
        if not meta or "class_factory" not in meta:
            available = list(provider.list(self.capability).get(self.capability, ()))
            raise RuntimeError(
                f"Unknown {self.capability} engine '{engine_name}'. Available: {available}"
            )
        entry = meta["class_factory"]
        if inspect.isclass(entry):
            return entry
        return entry()

    def create_engine_instance(self, engine_name: str) -> Any:
        """Create a new engine instance. EngineProvider handles caching."""
        engine_name = engine_name.lower()

        if engine_name in self.deprecated_aliases:
            canonical = self.deprecated_aliases[engine_name]
            if engine_name not in self._deprecation_warned:
                self.logger.warning(
                    "%s engine name '%s' is deprecated; use '%s' instead.",
                    self.label,
                    engine_name,
                    canonical,
                )
                self._deprecation_warned.add(engine_name)
            engine_name = canonical

        self.logger.info("Creating %s engine instance: %s", self.capability, engine_name)
        engine_class = self._resolve_engine_class(engine_name)
        instance = engine_class()

        try:
            available = instance.is_available()
        except Exception as exc:  # pragma: no cover - defensive
            self.logger.exception("Failed to check availability for %s", engine_name)
            raise RuntimeError(
                f"{self.label} engine '{engine_name}' availability check failed: {exc}"
            )

        if not available:
            hint = self.install_hints.get(engine_name, "")
            raise RuntimeError(
                f"{self.label} engine '{engine_name}' is not available. {hint}".strip()
            )

        return instance

    # -- Provider registration -----------------------------------------------

    def register_engines(self, provider=None) -> None:
        """Register all built-in engines (and aliases) with the EngineProvider."""
        for engine_name, class_factory, options_class in self.engine_defs:

            def factory(*, context=None, _engine_name=engine_name, **opts):
                return self.create_engine_instance(_engine_name)

            register_builtin(
                provider,
                self.capability,
                engine_name,
                factory,
                metadata={"options_class": options_class, "class_factory": class_factory},
            )

        # Register deprecated aliases so they are discoverable via EngineProvider
        for alias, canonical in self.deprecated_aliases.items():

            def alias_factory(*, context=None, _alias=alias, **opts):
                return self.create_engine_instance(_alias)

            # Alias shares the canonical engine's options class
            canonical_options = next(
                (oc for name, _, oc in self.engine_defs if name == canonical),
                self.fallback_options_class,
            )
            register_builtin(
                provider,
                self.capability,
                alias,
                alias_factory,
                metadata={"options_class": canonical_options},
            )

    def register_at_import(self) -> None:
        """Register engines, logging (not raising) on failure.

        Manager modules call this at import time so engines are discoverable
        immediately without import errors breaking the package.
        """
        try:
            self.register_engines()
        except Exception:  # pragma: no cover - defensive
            self.logger.exception("Failed to register built-in %s engines", self.capability)


__all__ = ["EngineDef", "EngineManagerSpec"]
