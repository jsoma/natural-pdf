"""Central registry/factory for pluggable engines."""

from __future__ import annotations

import logging
import threading
import weakref
from collections import defaultdict
from collections.abc import Hashable
from contextlib import contextmanager
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    cast,
)

try:  # Python 3.10+
    from importlib.metadata import entry_points
except ImportError:  # pragma: no cover - fallback for older versions
    from importlib_metadata import entry_points  # type: ignore


logger = logging.getLogger(__name__)


EngineFactory = Callable[..., Any]
EngineLifetime = Literal["context", "singleton", "transient"]
EngineCacheKey = Callable[[Any, Mapping[str, Any]], Hashable]


@dataclass(frozen=True)
class _EngineRegistration:
    name: str
    factory: EngineFactory
    metadata: Optional[Dict[str, Any]] = None
    lifetime: EngineLifetime = "context"
    cache_key: Optional[EngineCacheKey] = None


@dataclass(frozen=True)
class _CachedEngine:
    """An engine and opaque options whose identities participate in its key.

    Weak-referenceable contexts are evicted automatically. Retaining option
    values prevents Python from recycling the ids of opaque, identity-keyed
    clients and model handles while their engine is cached.
    """

    engine: Any
    retained_options: Mapping[str, Any]


class _WeakContextKey:
    __slots__ = ("identity", "reference", "token")

    def __init__(
        self,
        identity: int,
        reference: weakref.ReferenceType[Any],
        token: object,
    ) -> None:
        self.identity = identity
        self.reference = reference
        self.token = token

    def __hash__(self) -> int:
        return id(self.token)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _WeakContextKey) and self.token is other.token


class _StrongContextKey:
    """Identity key for contexts which do not support weak references."""

    __slots__ = ("context", "identity")

    def __init__(self, context: Any) -> None:
        self.context = context
        self.identity = id(context)

    def __hash__(self) -> int:
        return self.identity

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _StrongContextKey) and self.context is other.context


class EngineProvider:
    """Thread-safe registry and lifecycle owner for pluggable engines.

    Registrations choose one of three lifetimes:

    ``context`` (the default)
        Cache by context identity and constructor options. This is the safe
        default for extension factories because inputs from one document or
        client cannot silently initialize an engine reused by another.
        Default-keyed, weak-referenceable contexts are evicted when collected,
        provided the cached engine/options do not themselves retain the context.
        Non-weak-referenceable contexts, context-retaining engines, and custom
        cache keys require explicit ``evict()`` or ``clear()`` lifecycle control.
    ``singleton``
        Cache one instance for the capability/name pair. Use this explicitly
        only when the factory is independent of ``context`` and ``options``.
    ``transient``
        Never cache. The caller owns cleanup of every returned instance.

    A context registration may supply ``cache_key`` to intentionally share
    instances according to a stable, hashable application key instead of the
    default context/options key.
    """

    ENTRY_POINT_GROUP = "natural_pdf.engines"

    def __init__(self) -> None:
        self._registry: Dict[str, Dict[str, _EngineRegistration]] = defaultdict(dict)
        self._instances: Dict[tuple[str, str, Hashable], _CachedEngine] = {}
        self._context_keys: Dict[int, _WeakContextKey] = {}
        self._lock = threading.RLock()
        self._entry_points_loaded = False

    # ------------------------------------------------------------------
    # Registration & listing
    # ------------------------------------------------------------------
    def register(
        self,
        capability: str,
        name: str,
        factory: EngineFactory,
        metadata: Optional[Dict[str, Any]] = None,
        *,
        replace: bool = False,
        lifetime: EngineLifetime = "context",
        cache_key: Optional[EngineCacheKey] = None,
    ) -> None:
        """Register a factory for a capability/name pair.

        Args:
            capability: Capability implemented by the engine.
            name: Name used to resolve the engine.
            factory: Callable receiving ``context=...`` and retrieval options.
            metadata: Optional discovery metadata.
            replace: Replace an existing registration and evict its instances.
            lifetime: ``"context"``, ``"singleton"``, or ``"transient"``.
            cache_key: Optional function called as ``cache_key(context, options)``
                for context-lifetime registrations. It must return a hashable
                value and makes any cross-context sharing explicit.

        Replacing a registration detaches old instances while holding the
        provider lock, then invokes their cleanup hooks after releasing it.
        Cleanup failures are logged and do not undo the new registration.
        """

        capability = capability.strip().lower()
        name = name.strip().lower()

        if not capability or not name:
            raise ValueError("capability and name must be non-empty strings")
        if lifetime not in {"context", "singleton", "transient"}:
            raise ValueError("lifetime must be 'context', 'singleton', or 'transient'")
        if cache_key is not None and lifetime != "context":
            raise ValueError("cache_key is supported only for context-lifetime engines")

        cleanup: list[Any] = []
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
                metadata=dict(metadata or {}),
                lifetime=lifetime,
                cache_key=cache_key,
            )

            # Detach all variants initialized by the old registration. The
            # replacement is already visible before potentially slow cleanup.
            _, cleanup = self._detach_instances_locked(capability, name)

        self._cleanup_engines(cleanup)

    def list(self, capability: Optional[str] = None) -> Dict[str, Iterable[str]]:
        """Return registered engine names per capability."""

        with self._lock:
            if capability is None:
                return {cap: tuple(regs.keys()) for cap, regs in self._registry.items()}

            cap = capability.strip().lower()
            return {cap: tuple(self._registry.get(cap, {}).keys())}

    def get_metadata(
        self,
        capability: str,
        name: str,
    ) -> Optional[Dict[str, Any]]:
        """Return the metadata dict for a registered engine, or None if not found."""
        cap = capability.strip().lower()
        engine_name = name.strip().lower()
        with self._lock:
            reg = self._registry.get(cap, {}).get(engine_name)
            return dict(reg.metadata) if reg and reg.metadata else None

    # ------------------------------------------------------------------
    # Retrieval and lifecycle
    # ------------------------------------------------------------------
    def get(
        self,
        capability: str,
        *,
        context: Any,
        name: Optional[str] = None,
        **options: Any,
    ) -> Any:
        """Return an engine instance for the requested capability/name.

        Context-lifetime engines reuse an instance only when both the context
        and constructor options resolve to the same cache key. Singleton
        registrations deliberately ignore those inputs. Transient instances
        are returned directly and are never owned or cleaned by the provider.
        """

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

        with self._lock:
            registration = self._registry.get(cap, {}).get(engine_name)
            if registration is None:
                available = sorted(self._registry.get(cap, {}).keys())
                hint = (
                    f" Available engines for '{cap}': {', '.join(available)}."
                    if available
                    else f" No engines are registered for capability '{cap}'."
                )
                raise LookupError(
                    f"Engine '{engine_name}' is not registered for capability '{cap}'." + hint
                )

            if registration.lifetime == "transient":
                return registration.factory(context=context, **options)

            variant_key, retained_options = self._instance_variant_key(
                registration, context, options
            )
            key = (cap, engine_name, variant_key)
            cached = self._instances.get(key)
            if cached is None:
                engine = registration.factory(context=context, **options)
                cached = _CachedEngine(engine=engine, retained_options=retained_options)
                self._instances[key] = cached

            return cached.engine

    @contextmanager
    def checkout(
        self,
        capability: str,
        *,
        context: Any,
        name: Optional[str] = None,
        **options: Any,
    ):
        """Scoped :meth:`get`: yield an engine and clean up transient instances.

        Transient registrations return a fresh instance from every ``get``
        and assign cleanup to the caller; ``checkout`` discharges that duty by
        invoking the engine's ``cleanup()`` (or ``close()``) hook on exit —
        the same convention used for evicted cached engines. Engines with
        ``context``/``singleton`` lifetimes are yielded untouched: the
        provider still owns those instances and cleans them via
        :meth:`evict`/:meth:`clear` or context collection.
        """

        engine = self.get(capability, context=context, name=name, **options)

        cap = capability.strip().lower()
        engine_name = (name or "").strip().lower()
        with self._lock:
            registration = self._registry.get(cap, {}).get(engine_name)
        transient = registration is not None and registration.lifetime == "transient"

        try:
            yield engine
        finally:
            if transient:
                self._cleanup_engines([engine])

    def evict(self, capability: str, name: str) -> int:
        """Evict every cached variant for one registration.

        Instances are detached atomically and cleaned after releasing the
        provider lock. The return value is the number of cache entries removed;
        shared object identities are cleaned at most once, and only after no
        other provider cache entry references them. Callers must coordinate
        eviction with engine use; returned engines are not leased or reference
        counted once :meth:`get` returns.
        """

        cap = capability.strip().lower()
        engine_name = name.strip().lower()
        if not cap or not engine_name:
            raise ValueError("capability and name must be non-empty strings")

        with self._lock:
            count, cleanup = self._detach_instances_locked(cap, engine_name)
        self._cleanup_engines(cleanup)
        return count

    def clear(self, capability: Optional[str] = None) -> int:
        """Evict cached instances, optionally restricted to one capability.

        Registrations remain intact. Instances are detached under the provider
        lock and their cleanup hooks run after the lock has been released.
        Callers must ensure matching engines are not concurrently in use.
        """

        cap = capability.strip().lower() if capability is not None else None
        if capability is not None and not cap:
            raise ValueError("capability must be a non-empty string")

        with self._lock:
            count, cleanup = self._detach_instances_locked(capability=cap)
        self._cleanup_engines(cleanup)
        return count

    def _instance_variant_key(
        self,
        registration: _EngineRegistration,
        context: Any,
        options: Mapping[str, Any],
    ) -> tuple[Hashable, Mapping[str, Any]]:
        if registration.lifetime == "singleton":
            return "singleton", {}

        if registration.cache_key is not None:
            key = registration.cache_key(context, options)
            try:
                hash(key)
            except TypeError as exc:
                raise TypeError("Engine cache_key() must return a hashable value") from exc
            return key, {}

        return (self._context_key_locked(context), _freeze_options(options)), dict(options)

    def _context_key_locked(self, context: Any) -> Hashable:
        if context is None:
            return None

        identity = id(context)
        cached = self._context_keys.get(identity)
        if cached is not None and cached.reference() is context:
            return cached

        token = object()
        provider_ref = weakref.ref(self)

        def context_collected(_reference: weakref.ReferenceType[Any]) -> None:
            provider = provider_ref()
            if provider is not None:
                provider._evict_collected_context(identity, token)

        try:
            reference = weakref.ref(context, context_collected)
        except TypeError:
            # Lists, dicts, and some extension types cannot be weak-referenced.
            # Retaining them in the key is the only safe way to prevent id reuse.
            return _StrongContextKey(context)

        key = _WeakContextKey(identity, reference, token)
        self._context_keys[identity] = key
        return key

    def _evict_collected_context(self, identity: int, token: object) -> None:
        """Evict default-keyed engines when a weak context is collected."""

        with self._lock:
            context_key = self._context_keys.get(identity)
            if context_key is None or context_key.token is not token:
                return
            self._context_keys.pop(identity, None)
            count, cleanup = self._detach_context_instances_locked(context_key)

        if count:
            self._cleanup_engines(cleanup)

    def _detach_context_instances_locked(
        self, context_key: _WeakContextKey
    ) -> tuple[int, List[Any]]:
        matching = [
            key
            for key in self._instances
            if isinstance(key[2], tuple) and key[2] and key[2][0] is context_key
        ]
        return self._detach_keys_locked(matching)

    def _detach_instances_locked(
        self,
        capability: Optional[str] = None,
        name: Optional[str] = None,
    ) -> tuple[int, List[Any]]:
        """Detach matching records and return engines safe to clean. Lock required."""

        matching = [
            key
            for key in self._instances
            if (capability is None or key[0] == capability) and (name is None or key[1] == name)
        ]
        return self._detach_keys_locked(matching)

    def _detach_keys_locked(
        self, matching: Iterable[tuple[str, str, Hashable]]
    ) -> tuple[int, List[Any]]:
        keys = list(matching)
        detached = [self._instances.pop(key) for key in keys]
        live_ids = {id(record.engine) for record in self._instances.values()}

        cleanup: list[Any] = []
        seen: set[int] = set()
        for record in detached:
            identity = id(record.engine)
            if identity not in live_ids and identity not in seen:
                seen.add(identity)
                cleanup.append(record.engine)
        return len(keys), cleanup

    @staticmethod
    def _cleanup_engines(engines: Iterable[Any]) -> None:
        for engine in engines:
            cleanup_fn = getattr(engine, "cleanup", None)
            if not callable(cleanup_fn):
                cleanup_fn = getattr(engine, "close", None)
            if not callable(cleanup_fn):
                continue
            try:
                cleanup_fn()
            except Exception:
                logger.debug("Cleanup failed for evicted engine", exc_info=True)

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


def _freeze_options(options: Mapping[str, Any]) -> Hashable:
    """Build a stable key for ordinary option containers and opaque identities."""

    active: set[int] = set()
    return tuple(
        sorted((str(key), _freeze_option_value(value, active)) for key, value in options.items())
    )


def _freeze_option_value(value: Any, active: set[int]) -> Hashable:
    if value is None or isinstance(value, (bool, int, float, str, bytes)):
        return (type(value).__qualname__, value)

    if isinstance(value, (Mapping, tuple, list, set, frozenset)):
        identity = id(value)
        if identity in active:
            return ("cycle", type(value).__module__, type(value).__qualname__, identity)
        active.add(identity)
        try:
            if isinstance(value, Mapping):
                items = [
                    (_freeze_option_value(key, active), _freeze_option_value(item, active))
                    for key, item in value.items()
                ]
                return ("mapping", tuple(sorted(items, key=repr)))
            if isinstance(value, tuple):
                return ("tuple", tuple(_freeze_option_value(item, active) for item in value))
            if isinstance(value, list):
                return ("list", tuple(_freeze_option_value(item, active) for item in value))
            return (
                "set",
                tuple(sorted((_freeze_option_value(item, active) for item in value), key=repr)),
            )
        finally:
            active.remove(identity)

    # Arbitrary clients, sessions, model handles, and other opaque values use
    # identity rather than potentially surprising custom equality semantics.
    return ("identity", type(value).__module__, type(value).__qualname__, id(value))


_PROVIDER: Optional[EngineProvider] = None
_PROVIDER_LOCK = threading.Lock()


def get_provider() -> EngineProvider:
    global _PROVIDER
    if _PROVIDER is None:
        with _PROVIDER_LOCK:
            if _PROVIDER is None:
                _PROVIDER = EngineProvider()
    return _PROVIDER


# Ensure entry points are loaded when the module is imported.
get_provider()._ensure_entry_points_loaded()
