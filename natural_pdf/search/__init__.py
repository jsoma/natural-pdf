"""Makes search functionality easily importable and provides factory functions."""

import logging
from typing import Optional

# --- Service Implementation Import ---
# Import the concrete implementation
from .haystack_search_service import HaystackSearchService

# --- Utils Import ---
from .haystack_utils import (  # Re-export flag and helper
    HAS_HAYSTACK_EXTRAS,
    check_haystack_availability,
)

# --- Option Imports (for convenience) ---
# Make options easily available via `from natural_pdf.search import ...`
from .search_options import SearchOptions  # Alias for TextSearchOptions for simplicity?
from .search_options import BaseSearchOptions, MultiModalSearchOptions, TextSearchOptions

# --- Protocol Import ---
# Import the protocol for type hinting
from .search_service_protocol import Indexable, IndexConfigurationError, SearchServiceProtocol

logger = logging.getLogger(__name__)


# Factory Function
def get_search_service(
    collection_name: str,
    persist: bool = False,
    uri: Optional[str] = None,
    default_embedding_model: Optional[str] = None,
) -> SearchServiceProtocol:
    """
    Factory function to get an instance of the configured search service.

    A service instance is tied to a specific index name (collection/table).

    Currently, only returns HaystackSearchService but is structured for future extension.

    Args:
        collection_name: The logical name for the index this service instance manages
                         (used as table_name for LanceDB).
        persist: If True, creates a service instance configured for persistent
                 storage (currently LanceDB). If False (default), uses InMemory.
        uri: Override the default path/URI for persistent storage.
        default_embedding_model: Override the default embedding model used by the service.
        **kwargs: Reserved for future configuration options.

    Returns:
        An instance conforming to the SearchServiceProtocol for the specified collection/table.
    """
    logger.debug(
        f"Calling get_search_service factory for index '{collection_name}' (persist={persist}, uri={uri})..."
    )

    # Collect arguments relevant to HaystackSearchService.__init__
    service_args = {}
    service_args["table_name"] = collection_name
    service_args["persist"] = persist
    if uri is not None:
        service_args["uri"] = uri
    if default_embedding_model is not None:
        service_args["embedding_model"] = default_embedding_model

    # Cache logic commented out as before

    try:
        service_instance = HaystackSearchService(**service_args)
        logger.info(f"Created new HaystackSearchService instance for index '{collection_name}'.")
        return service_instance
    except ImportError as e:
        # Error message remains valid
        logger.error(
            f"Failed to instantiate Search Service due to missing dependencies: {e}", exc_info=True
        )
        raise ImportError(
            "Search Service could not be created. Ensure Haystack extras are installed: pip install natural-pdf[haystack]"
        ) from e
    except Exception as e:
        logger.error(f"Failed to instantiate Search Service: {e}", exc_info=True)
        raise RuntimeError("Could not create Search Service instance.") from e


# Default instance commented out as before
