"""Makes search functionality easily importable and provides factory functions."""

import logging
from typing import Optional, Type, Union

# Import constants
from .search_options import (
    BaseSearchOptions,
    MultiModalSearchOptions,
    SearchOptions,
    TextSearchOptions,
)
from .search_service_protocol import Indexable, IndexConfigurationError, SearchServiceProtocol

# Check search extras availability
LANCEDB_AVAILABLE = False
SEARCH_DEPENDENCIES_AVAILABLE = False
LanceDBSearchService: Optional[Type[SearchServiceProtocol]] = None
NumpySearchService: Optional[Type[SearchServiceProtocol]] = None
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_LANCEDB_PERSIST_PATH = "./lancedb_data"

try:
    import numpy as np  # noqa: F401

    SEARCH_DEPENDENCIES_AVAILABLE = True

    try:
        from .lancedb_search_service import (
            DEFAULT_EMBEDDING_MODEL as LANCEDB_DEFAULT_EMBEDDING_MODEL,
        )
        from .lancedb_search_service import DEFAULT_LANCEDB_PERSIST_PATH as LANCEDB_PERSIST_PATH
        from .lancedb_search_service import LanceDBSearchService as _LanceDBSearchService

        LanceDBSearchService = _LanceDBSearchService
        DEFAULT_EMBEDDING_MODEL = LANCEDB_DEFAULT_EMBEDDING_MODEL
        DEFAULT_LANCEDB_PERSIST_PATH = LANCEDB_PERSIST_PATH
        LANCEDB_AVAILABLE = True
    except ImportError:
        LANCEDB_AVAILABLE = False
        try:
            from .numpy_search_service import (
                DEFAULT_EMBEDDING_MODEL as NUMPY_DEFAULT_EMBEDDING_MODEL,
            )
            from .numpy_search_service import NumpySearchService as _NumpySearchService

            NumpySearchService = _NumpySearchService
            DEFAULT_EMBEDDING_MODEL = NUMPY_DEFAULT_EMBEDDING_MODEL
        except ImportError:
            NumpySearchService = None
except ImportError:
    SEARCH_DEPENDENCIES_AVAILABLE = False
    LANCEDB_AVAILABLE = False

logger = logging.getLogger(__name__)


def _check_sentence_transformers():
    """Lazy check for sentence_transformers availability."""
    try:
        import sentence_transformers

        return True
    except ImportError:
        return False


def check_search_availability():
    """Check if required search dependencies are available."""
    if not SEARCH_DEPENDENCIES_AVAILABLE:
        raise ImportError(
            "Search functionality requires 'lancedb' and pyarrow. "
            "Install with: pip install natural-pdf[search] (or pip install lancedb pyarrow)"
        )

    # Lazy check for sentence_transformers when actually needed
    if not _check_sentence_transformers():
        raise ImportError(
            "Search functionality requires 'sentence-transformers'. "
            "Install with: pip install sentence-transformers"
        )


def get_search_service(
    collection_name: str,
    persist: bool = False,
    uri: Optional[str] = None,
    default_embedding_model: Optional[str] = None,
) -> SearchServiceProtocol:
    """
    Factory function to get an instance of the configured search service.

    Automatically selects the best available implementation:
    - LanceDB if installed (recommended for both in-memory and persistent)
    - Numpy fallback for in-memory only

    Args:
        collection_name: The logical name for the index/table this service instance manages.
        persist: If True, creates a service instance configured for persistent
                 storage. If False (default), uses InMemory (via temp dir for LanceDB).
        uri: Override the default path for persistent storage.
        default_embedding_model: Override the default embedding model used by the service.

    Returns:
        An instance conforming to the SearchServiceProtocol.
    """
    logger.debug(
        f"Calling get_search_service factory for collection '{collection_name}' (persist={persist}, uri={uri})..."
    )
    check_search_availability()

    service_args = {
        "collection_name": collection_name,
        "persist": persist,
    }
    if uri is not None:
        service_args["uri"] = uri

    if default_embedding_model is not None:
        service_args["embedding_model_name"] = default_embedding_model

    # If persistence is requested, LanceDB is required
    if persist and not LANCEDB_AVAILABLE:
        raise RuntimeError(
            "Persistent vector search requires LanceDB. " "Please install: pip install lancedb"
        )

    # Select the appropriate implementation
    if LANCEDB_AVAILABLE and LanceDBSearchService is not None:
        logger.info(f"Using LanceDB for vector search (collection: {collection_name})")
        service_instance = LanceDBSearchService(**service_args)
    elif NumpySearchService is not None:
        logger.info(
            f"Using NumPy fallback for in-memory vector search (collection: {collection_name})"
        )
        service_instance = NumpySearchService(**service_args)
    else:
        raise RuntimeError(
            "Vector search dependencies are not available. "
            "Install with: pip install natural-pdf[search]"
        )

    return service_instance
