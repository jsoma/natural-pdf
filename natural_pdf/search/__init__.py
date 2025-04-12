"""Makes search functionality easily importable and provides factory functions."""

import logging
from typing import Optional

# --- Service Implementation Import ---
# Import the concrete implementation
from .haystack_search_service import HaystackSearchService

# --- Protocol Import ---
# Import the protocol for type hinting
from .search_service_protocol import (
    SearchServiceProtocol, 
    IndexConfigurationError, 
    Indexable
)

# --- Option Imports (for convenience) ---
# Make options easily available via `from natural_pdf.search import ...`
from .search_options import (
    BaseSearchOptions,
    SearchOptions, # Alias for TextSearchOptions for simplicity?
    TextSearchOptions,
    MultiModalSearchOptions
)
# --- Utils Import ---
from .haystack_utils import HAS_HAYSTACK_EXTRAS, check_haystack_availability # Re-export flag and helper

logger = logging.getLogger(__name__)

# --- Factory Function ---

def get_search_service(
    collection_name: str, # Add collection_name as a required argument
    persist: bool = False, # Default to In-Memory
    # Configuration for the service itself
    default_persist_path: Optional[str] = None,
    default_embedding_model: Optional[str] = None,
    # Potential future args: cache_services=True? service_type='haystack'?
) -> SearchServiceProtocol:
    """
    Factory function to get an instance of the configured search service.

    A service instance is tied to a specific collection name.

    Currently, only returns HaystackSearchService but is structured for future extension.

    Args:
        collection_name: The name of the collection this service instance will manage.
        persist: If True, creates a service instance configured for persistent
                 storage (ChromaDB). If False (default), uses In-Memory.
        default_persist_path: Override the default path for persistent storage.
        default_embedding_model: Override the default embedding model used by the service.
        **kwargs: Reserved for future configuration options.

    Returns:
        An instance conforming to the SearchServiceProtocol for the specified collection.
    """
    logger.debug(f"Calling get_search_service factory for collection '{collection_name}' (persist={persist})...")
    
    # For now, we only have one implementation
    # Collect arguments relevant to HaystackSearchService.__init__
    service_args = {}
    service_args['collection_name'] = collection_name # Pass collection_name
    service_args['persist'] = persist # Pass persist flag to service constructor
    if default_persist_path is not None:
        service_args['default_persist_path'] = default_persist_path
    if default_embedding_model is not None:
        service_args['default_embedding_model'] = default_embedding_model
        
    # TODO: Implement caching/registry if needed to return the same instance
    # for the same configuration instead of always creating a new one.
    # cache_key = tuple(sorted(service_args.items()))
    # if cache_key in _service_instance_cache:
    #    return _service_instance_cache[cache_key]
        
    try:
        service_instance = HaystackSearchService(**service_args)
        # _service_instance_cache[cache_key] = service_instance
        logger.info(f"Created new HaystackSearchService instance for collection '{collection_name}'.")
        return service_instance
    except ImportError as e:
         logger.error(f"Failed to instantiate Search Service due to missing dependencies: {e}", exc_info=True)
         raise ImportError("Search Service could not be created. Ensure Haystack extras are installed: pip install natural-pdf[haystack]") from e
    except Exception as e:
         logger.error(f"Failed to instantiate Search Service: {e}", exc_info=True)
         raise RuntimeError("Could not create Search Service instance.") from e

# --- Optional: Define a default instance for extreme ease of use? ---
# try:
#     default_search_service = get_search_service()
# except Exception:
#     default_search_service = None 
#     logger.warning("Could not create default search service instance on import.") 