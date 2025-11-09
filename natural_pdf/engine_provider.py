"""Central registry/factory for pluggable engines."""

from __future__ import annotations

import logging
import threading
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple, cast

try:  # Python 3.10+
    from importlib.metadata import entry_points
except ImportError:  # pragma: no cover - fallback for older versions
    from importlib_metadata import entry_points  # type: ignore


logger = logging.getLogger(__name__)


EngineFactory = Callable[[Any], Any]


@dataclass(frozen=True)
class _EngineRegistration:
    name: str
    factory: Callable[..., Any]
    metadata: Optional[Dict[str, Any]] = None


class EngineProvider:
    """Singleton registry that manages engine registration and retrieval."""

    ENTRY_POINT_GROUP = "natural_pdf.engines"

    def __init__(self) -> None:
        self._registry: Dict[str, Dict[str, _EngineRegistration]] = defaultdict(dict)
        self._instances: Dict[tuple[str, str], Any] = {}
        self._lock = threading.RLock()
        self._entry_points_loaded = False

    # ------------------------------------------------------------------
    # Registration & listing
    # ------------------------------------------------------------------
    def register(
        self,
        capability: str,
        name: str,
        factory: Callable[..., Any],
        metadata: Optional[Dict[str, Any]] = None,
        *,
        replace: bool = False,
    ) -> None:
        """Register a factory for a capability/name pair."""

        capability = capability.strip().lower()
        name = name.strip().lower()

        if not capability or not name:
            raise ValueError("capability and name must be non-empty strings")

        with self._lock:
            if name in self._registry[capability] and not replace:
                logger.warning(
                    "Engine for capability '%s' with name '%s' already registered; skipping",
                    capability,
                    name,
                )
                return

            self._registry[capability][name] = _EngineRegistration(
                name=name,
                factory=factory,
                metadata=metadata or {},
            )

            # Drop cached instance if we are replacing it
            self._instances.pop((capability, name), None)

    def list(self, capability: Optional[str] = None) -> Dict[str, Iterable[str]]:
        """Return registered engine names per capability."""

        if capability is None:
            return {cap: tuple(regs.keys()) for cap, regs in self._registry.items()}

        cap = capability.strip().lower()
        return {cap: tuple(self._registry.get(cap, {}).keys())}

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------
    def get(
        self,
        capability: str,
        *,
        context: Any,
        name: Optional[str] = None,
        **options: Any,
    ) -> Any:
        """Return an engine instance for the requested capability/name."""

        self._ensure_entry_points_loaded()

        cap = capability.strip().lower()
        if not cap:
            raise ValueError("capability must be provided")

        engine_name = (name or "").strip().lower()
        if not engine_name:
            raise ValueError(
                f"Engine name must be provided for capability '{cap}'. "
                "(Default resolution via options not implemented yet.)"
            )

        key = (cap, engine_name)

        with self._lock:
            registration = self._registry.get(cap, {}).get(engine_name)
            if registration is None:
                raise LookupError(
                    f"Engine '{engine_name}' is not registered for capability '{cap}'."
                )

            if key not in self._instances:
                self._instances[key] = registration.factory(
                    context=context,
                    **options,
                )

            return self._instances[key]

    # ------------------------------------------------------------------
    # Entry points
    # ------------------------------------------------------------------
    def _ensure_entry_points_loaded(self) -> None:
        if self._entry_points_loaded:
            return
        with self._lock:
            if self._entry_points_loaded:
                return

            try:
                groups = entry_points()
                if hasattr(groups, "select"):
                    candidates: Iterable[Any] = groups.select(group=self.ENTRY_POINT_GROUP)  # type: ignore[attr-defined]
                else:  # pragma: no cover - older importlib_metadata API
                    mapping = cast(Mapping[str, Sequence[Any]], groups)
                    candidates = mapping.get(self.ENTRY_POINT_GROUP, ())

                for ep in candidates:
                    try:
                        logger.debug("Loading natural-pdf engine entry point %s", ep)
                        register_fn = ep.load()
                        register_fn(self)
                    except Exception:  # pragma: no cover - defensive
                        logger.exception("Failed to load engine entry point '%s'", ep.name)
            finally:
                self._entry_points_loaded = True


_PROVIDER: Optional[EngineProvider] = None


def get_provider() -> EngineProvider:
    global _PROVIDER
    if _PROVIDER is None:
        _PROVIDER = EngineProvider()
    return _PROVIDER


# Ensure entry points are loaded when the module is imported.
get_provider()._ensure_entry_points_loaded()
