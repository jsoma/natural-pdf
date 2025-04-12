import logging
from typing import Optional, List, Dict, Any, Iterable, TYPE_CHECKING, Union, Type, Generator
from abc import ABC, abstractmethod
import hashlib # For hashing content

# Now import the flag from the canonical source - this import should always work
from .haystack_utils import HAS_HAYSTACK_EXTRAS
DEFAULT_SEARCH_COLLECTION_NAME = "default_collection"

# Avoid runtime import errors if extras not installed
try:
    # Import protocols and options first
    from .search_service_protocol import (
        SearchServiceProtocol, Indexable, 
        IndexConfigurationError, IndexExistsError
    )
    from .search_options import SearchOptions, TextSearchOptions
    from . import get_search_service
    
    if TYPE_CHECKING: # Keep type hints working
        from natural_pdf.elements.region import Region # Example indexable type
except ImportError:
    # Define dummies if extras missing
    SearchServiceProtocol, Indexable, IndexConfigurationError, IndexExistsError = object, object, RuntimeError, RuntimeError
    SearchOptions, TextSearchOptions = object, object
    DEFAULT_SEARCH_COLLECTION_NAME = "default_collection"
    def get_search_service(**kwargs): raise ImportError("Search dependencies missing.")
    class Region: pass # Dummy for type hint

logger = logging.getLogger(__name__)

class SearchableMixin(ABC):
    """
    Mixin class providing search functionality (initialization, indexing, searching, syncing).

    Requires the inheriting class to implement `get_indexable_items`.
    Assumes the inheriting class has a `_search_service` attribute initialized to None.
    """
    # Ensure inheriting class initializes this
    _search_service: Optional[SearchServiceProtocol] = None

    @abstractmethod
    def get_indexable_items(self) -> Iterable[Indexable]:
        """
        Abstract method that must be implemented by the inheriting class.
        Should yield or return an iterable of objects conforming to the Indexable protocol.
        """
        pass

    def init_search(
        self,
        service: Optional[SearchServiceProtocol] = None,
        *,
        persist: Optional[bool] = None,
        collection_name: Optional[str] = None,
        embedding_model: Optional[str] = None, # Allow overriding embedding model
        index: bool = False, # Changed from index_now
        force_reindex: bool = False,
        embedder_device: Optional[str] = None,
        **kwargs # Pass other args to get_search_service
    ) -> 'SearchableMixin': # Return self for chaining
        """
        Initializes and configures the search service for this instance.

        Call this explicitly before `index_for_search`, `sync_index`, or `find_relevant`
        if using non-default settings (e.g., persistence) or attaching an
        existing service instance.

        Args:
            service: An optional pre-configured SearchServiceProtocol instance.
                     If provided, attaches this service directly, ignoring other
                     configuration arguments (persist, collection_name, etc.).
            persist: If creating a new service (service=None), determines if it should
                     use persistent storage (True) or be in-memory (False/None).
                     Defaults to False.
            collection_name: If creating a new service, the name for the index/collection.
                             Required if persist=True. Defaults to 'default_collection'
                             if persist=False.
            embedding_model: If creating a new service, override the default embedding model.
            index: If True, immediately indexes the collection's documents using the
                   configured service after setup. Calls `_perform_indexing`. Defaults to False.
            force_reindex: If index=True, instructs the service to delete any existing
                           index before indexing. Defaults to False.
            embedder_device: If index=True, optional device override for the embedder.
            **kwargs: Additional keyword arguments passed to get_search_service when creating
                      a new service instance.

        Returns:
            Self for method chaining.
        """
        if service:
            # Attach provided service
            logger.info(f"Attaching provided SearchService instance (Collection: '{getattr(service, 'collection_name', '<Unknown>')}').")
            # TODO: Add stricter type check? isinstance(service, SearchServiceProtocol) requires runtime_checkable
            self._search_service = service
        else:
            # Create new service
            effective_persist = persist if persist is not None else False
            effective_collection_name = collection_name
            if effective_persist and not effective_collection_name:
                raise ValueError("A collection_name must be provided when persist=True.")
            elif not effective_persist and not effective_collection_name:
                effective_collection_name = DEFAULT_SEARCH_COLLECTION_NAME
                logger.info(f"Using default collection name '{DEFAULT_SEARCH_COLLECTION_NAME}' for in-memory service.")

            logger.info(f"Creating new SearchService: name='{effective_collection_name}', persist={effective_persist}, model={embedding_model or 'default'}")
            try:
                service_args = {"collection_name": effective_collection_name, "persist": effective_persist, **kwargs}
                if embedding_model: service_args['embedding_model'] = embedding_model
                self._search_service = get_search_service(**service_args)
            except Exception as e:
                logger.error(f"Failed to create SearchService: {e}", exc_info=True)
                raise RuntimeError("Could not create SearchService instance.") from e

        # --- Optional Immediate Indexing (with safety check for persistent) --- 
        if index:
            if not self._search_service: # Should not happen if logic above is correct
                raise RuntimeError("Cannot index: Search service not available after initialization attempt.")

            is_persistent = getattr(self._search_service, '_persist', False) # Check if service is persistent
            collection_name = getattr(self._search_service, 'collection_name', '<Unknown>')

            if is_persistent and not force_reindex:
                # Check existence only if persistent and not forcing reindex
                if self._search_service.index_exists():
                    # Raise safety error if index exists and force_reindex is not True
                    raise IndexExistsError(
                        f"Persistent index '{collection_name}' already exists. "
                        f"To overwrite/re-index via init_search(index=True), explicitly set force_reindex=True. "
                        f"Alternatively, use index_for_search() or sync_index() for more granular control."
                    )
                else:
                    # Index doesn't exist, safe to proceed
                    logger.info(f"Persistent index '{collection_name}' does not exist. Proceeding with initial indexing.")
            elif is_persistent and force_reindex:
                logger.warning(f"Proceeding with index=True and force_reindex=True for persistent index '{collection_name}'. Existing data will be deleted.")
            # else: # Not persistent, safe to proceed without existence check
            #     logger.debug("Proceeding with index=True for non-persistent index.")

            # Proceed with indexing if checks passed or not applicable
            logger.info(f"index=True: Proceeding to index collection immediately after search initialization.")
            self._perform_indexing(force_reindex=force_reindex, embedder_device=embedder_device)

        return self

    def _perform_indexing(self, force_reindex: bool, embedder_device: Optional[str]):
        """Internal helper containing the core indexing logic."""
        if not self._search_service:
            raise RuntimeError("Search service not initialized. Call init_search first.")

        collection_name = getattr(self._search_service, 'collection_name', '<Unknown>')
        logger.info(f"Starting internal indexing process into SearchService collection '{collection_name}'...")

        # Use the abstract method to get items
        try:
            indexable_items = list(self.get_indexable_items()) # Consume iterator
        except Exception as e:
             logger.error(f"Error calling get_indexable_items: {e}", exc_info=True)
             raise RuntimeError("Failed to retrieve indexable items for indexing.") from e

        if not indexable_items:
            logger.warning("No indexable items provided by get_indexable_items(). Skipping index call.")
            return

        logger.info(f"Prepared {len(indexable_items)} indexable items for indexing.")
        try:
             logger.debug(f"Calling index() on SearchService for collection '{collection_name}' (force_reindex={force_reindex}).")
             self._search_service.index(
                 documents=indexable_items,
                 embedder_device=embedder_device,
                 force_reindex=force_reindex,
             )
             logger.info(f"Successfully completed indexing into SearchService collection '{collection_name}'.")
        except IndexConfigurationError as ice:
             logger.error(f"Indexing failed due to configuration error in collection '{collection_name}': {ice}", exc_info=True)
             raise # Re-raise specific error
        except Exception as e: # Catch other indexing errors from the service
             logger.error(f"Indexing failed for collection '{collection_name}': {e}", exc_info=True)
             raise RuntimeError(f"Indexing failed for collection '{collection_name}'.") from e


    def index_for_search(
        self, *, # Make args keyword-only
        embedder_device: Optional[str] = None,
        force_reindex: bool = False,
    ) -> 'SearchableMixin':
        """
        Ensures the search service is initialized (using default if needed)
        and indexes the items provided by `get_indexable_items`.

        If the search service hasn't been configured via `init_search`, this
        method will initialize the default in-memory service.

        Args:
            embedder_device: Optional device override for the embedder.
            force_reindex: If True, instructs the service to delete any existing
                           index before indexing.

        Returns:
            Self for method chaining.
        """
        # --- Ensure Service is Initialized (Use Default if Needed) ---
        if not self._search_service:
            logger.info("Search service not initialized prior to index_for_search. Initializing default in-memory service.")
            self.init_search() # Call init with defaults

        # --- Perform Indexing ---
        self._perform_indexing(force_reindex=force_reindex, embedder_device=embedder_device)
        return self

    def find_relevant(
        self,
        query: Any, # Query type depends on service capabilities
        *, # Make options/service keyword-only
        options: Optional[SearchOptions] = None,
        search_service: Optional[SearchServiceProtocol] = None # Allow override
    ) -> List[Dict[str, Any]]:
        """
        Finds relevant items using the configured or provided search service.

        Args:
            query: The search query (text, image path, PIL Image, Region, etc.).
                   The SearchService implementation handles the specific query type.
            options: Optional SearchOptions to configure the query (top_k, filters, etc.).
            search_service: Optional specific SearchService instance to use for this query,
                           overriding the collection's configured service.

        Returns:
            A list of result dictionaries, sorted by relevance.

        Raises:
            RuntimeError: If no search service is configured or provided, or if search fails.
            FileNotFoundError: If the collection managed by the service does not exist.
        """
        # --- Determine which Search Service to use ---
        effective_service = search_service or self._search_service
        if not effective_service:
            raise RuntimeError(
                "Search service not configured. Call init_search(...) or index_for_search() first, "
                "or provide an explicit 'search_service' instance to find_relevant()."
            )

        collection_name = getattr(effective_service, 'collection_name', '<Unknown>')
        logger.info(f"Searching collection '{collection_name}' via {type(effective_service).__name__}...")

        # --- Prepare Query and Options ---
        query_input = query
        # Example: Handle Region query - maybe move this logic into HaystackSearchService.search?
        # If we keep it here, it makes the mixin less generic.
        # Let's assume the SearchService handles the query type appropriately for now.
        # if isinstance(query, Region):
        #     logger.debug("Query is a Region object. Extracting text.")
        #     query_input = query.extract_text()
        #     if not query_input or query_input.isspace():
        #         logger.warning("Region provided for query has no extractable text.")
        #         return []

        effective_options = options if options is not None else TextSearchOptions()

        # --- Call SearchService Search Method ---
        try:
            results = effective_service.search(
                query=query_input,
                options=effective_options,
            )
            logger.info(f"SearchService returned {len(results)} results from collection '{collection_name}'.")
            return results
        except FileNotFoundError as fnf:
            logger.error(f"Search failed: Collection '{collection_name}' not found by service. Error: {fnf}")
            raise # Re-raise specific error
        except Exception as e:
            logger.error(f"Search failed for collection '{collection_name}': {e}", exc_info=True)
            # Consider wrapping in a SearchError?
            raise RuntimeError(f"Search failed in collection '{collection_name}'.") from e

    # --- NEW Sync Method ---
    def sync_index(
        self,
        strategy: str = 'full', # 'full' (add/update/delete) or 'upsert_only'
        dry_run: bool = False,
        batch_size: int = 100, # For batching deletes/updates if needed
        embedder_device: Optional[str] = None, # Pass embedder device if needed for updates
        **kwargs: Any # Allow passing extra args to get_search_service
    ) -> Dict[str, int]:
        """
        Synchronizes the search index with the current state of indexable items.
        Requires the configured search service to implement `list_documents`
        and `delete_documents` for the 'full' strategy.
        Requires `Indexable` items to implement `get_content_hash` for 'full' strategy
        change detection (falls back to ID-based update if hash is missing).

        Args:
            strategy: 'full' (Default): Adds new, updates changed (based on hash),
                      and deletes items no longer present.
                      'upsert_only': Adds new items and updates existing ones (based on ID),
                      but does not delete missing items. (Effectively like force_reindex=False index)
            dry_run: If True, calculates changes but does not modify the index.
            batch_size: Hint for batching delete/update operations (service implementation specific).
            embedder_device: Optional device for embedding during updates if needed by service.
            **kwargs: Additional keyword arguments passed to get_search_service when creating
                      a new service instance.

        Returns:
            A dictionary summarizing the changes (e.g., {'added': N, 'updated': M, 'deleted': K, 'skipped': S}).

        Raises:
            RuntimeError: For backend errors during synchronization.
        """
        if not self._search_service:
             raise RuntimeError("Search service not configured. Call init_search first.")

        collection_name = getattr(self._search_service, 'collection_name', '<Unknown>')
        logger.info(f"Starting index synchronization for collection '{collection_name}' (Strategy: {strategy}, Dry run: {dry_run})...")
        summary = {'added': 0, 'updated': 0, 'deleted': 0, 'skipped': 0}

        # --- Check Service Capabilities for 'full' sync ---
        if strategy == 'full':
             required_methods = ['list_documents', 'delete_documents']
             missing_methods = [m for m in required_methods if not hasattr(self._search_service, m)]
             if missing_methods:
                 raise NotImplementedError(
                     f"The configured search service ({type(self._search_service).__name__}) "
                     f"is missing required methods for 'full' sync strategy: {', '.join(missing_methods)}"
                 )

        # --- 1. Get Desired State (from current collection) ---
        desired_state: Dict[str, Indexable] = {} # {id: item}
        desired_hashes: Dict[str, Optional[str]] = {} # {id: hash or None}
        try:
             for item in self.get_indexable_items():
                 item_id = item.get_id()
                 if not item_id:
                     logger.warning(f"Skipping item with no ID: {item}")
                     summary['skipped'] += 1
                     continue
                 if item_id in desired_state:
                      logger.warning(f"Duplicate ID '{item_id}' found in get_indexable_items(). Skipping subsequent item.")
                      summary['skipped'] += 1
                      continue
                 desired_state[item_id] = item
                 # Try to get hash, store None if unavailable or fails
                 try:
                      desired_hashes[item_id] = item.get_content_hash()
                 except (AttributeError, NotImplementedError):
                      logger.debug(f"get_content_hash not available for item ID '{item_id}' ({type(item).__name__}). Sync update check will be ID-based.")
                      desired_hashes[item_id] = None
                 except Exception as e:
                      logger.warning(f"Error getting content hash for item ID '{item_id}': {e}. Sync update check will be ID-based.", exc_info=False)
                      desired_hashes[item_id] = None

        except Exception as e:
             logger.error(f"Error iterating through get_indexable_items: {e}", exc_info=True)
             raise RuntimeError("Failed to get current indexable items.") from e

        logger.info(f"Desired state contains {len(desired_state)} indexable items.")

        # --- 2. Handle Different Strategies ---
        if strategy == 'upsert_only':
            # Simple case: just index everything, let the service handle upserts
            items_to_index = list(desired_state.values())
            summary['added'] = len(items_to_index) # Approximate count
            logger.info(f"Strategy 'upsert_only': Prepared {len(items_to_index)} items for indexing/upserting.")
            if not dry_run and items_to_index:
                 logger.debug("Calling service.index for upsert...")
                 # Call index directly, force_reindex=False implies upsert
                 self._search_service.index(
                     documents=items_to_index,
                     force_reindex=False,
                     embedder_device=embedder_device
                 )
            elif dry_run:
                logger.info("[Dry Run] Would index/upsert %d items.", len(items_to_index))

        elif strategy == 'full':
            # Complex case: Add/Update/Delete
            # 2a. Get Current Index State
            try:
                 logger.debug("Listing documents currently in the index...")
                 # Assumes list_documents takes filters and include_metadata
                 # Fetch all documents with metadata
                 current_docs = self._search_service.list_documents(include_metadata=True)
                 current_state: Dict[str, Dict] = {} # {id: {'meta': {...}, ...}}
                 duplicates = 0
                 for doc in current_docs:
                     doc_id = doc.get('id')
                     if not doc_id: continue # Skip docs without ID from service
                     if doc_id in current_state: duplicates +=1
                     current_state[doc_id] = doc
                 logger.info(f"Found {len(current_state)} documents currently in the index (encountered {duplicates} duplicate IDs).")
                 if duplicates > 0: logger.warning(f"Found {duplicates} duplicate IDs in the index. Using the last encountered version for comparison.")

            except Exception as e:
                 logger.error(f"Failed to list documents from search service: {e}", exc_info=True)
                 raise RuntimeError("Could not retrieve current index state for sync.") from e

            # 2b. Compare States and Plan Actions
            ids_in_desired = set(desired_state.keys())
            ids_in_current = set(current_state.keys())

            ids_to_add = ids_in_desired - ids_in_current
            ids_to_delete = ids_in_current - ids_in_desired
            ids_to_check_update = ids_in_desired.intersection(ids_in_current)

            items_to_update = []
            for item_id in ids_to_check_update:
                 desired_hash = desired_hashes.get(item_id)
                 current_meta = current_state[item_id].get('meta', {})
                 current_hash = current_meta.get('content_hash') # Assuming hash stored in meta

                 # Check if hash exists and differs, or if hash is missing (force update)
                 if desired_hash is None or current_hash is None or desired_hash != current_hash:
                      if desired_hash != current_hash:
                          logger.debug(f"Content hash changed for ID {item_id}. Scheduling for update.")
                      else:
                          logger.debug(f"Hash missing for ID {item_id}. Scheduling for update.")
                      items_to_update.append(desired_state[item_id])
                 # Else: hashes match, no update needed

            items_to_add = [desired_state[id_] for id_ in ids_to_add]
            items_to_index = items_to_add + items_to_update # Combine adds and updates for single index call

            summary['added'] = len(items_to_add)
            summary['updated'] = len(items_to_update)
            summary['deleted'] = len(ids_to_delete)

            logger.info(f"Sync Plan: Add={summary['added']}, Update={summary['updated']}, Delete={summary['deleted']}")

            # 2c. Execute Actions (if not dry_run)
            if not dry_run:
                 # Execute Deletes
                 if ids_to_delete:
                      logger.info(f"Deleting {len(ids_to_delete)} items from index...")
                      try:
                           # Assuming delete_documents takes list of IDs
                           # Implement batching if needed
                           self._search_service.delete_documents(ids=list(ids_to_delete))
                           logger.info("Deletion successful.")
                      except Exception as e:
                           logger.error(f"Failed to delete documents: {e}", exc_info=True)
                           # Decide whether to continue or raise
                           raise RuntimeError("Failed during deletion phase of sync.") from e

                 # Execute Adds/Updates
                 if items_to_index:
                      logger.info(f"Indexing/Updating {len(items_to_index)} items...")
                      try:
                           # Upsert logic handled by service's index method with force_reindex=False
                           self._search_service.index(
                               documents=items_to_index,
                               force_reindex=False,
                               embedder_device=embedder_device
                           )
                           logger.info("Add/Update successful.")
                      except Exception as e:
                           logger.error(f"Failed to index/update documents: {e}", exc_info=True)
                           raise RuntimeError("Failed during add/update phase of sync.") from e
                 logger.info("Sync actions completed.")
            else:
                 logger.info("[Dry Run] No changes applied to the index.")

        else:
            raise ValueError(f"Unknown sync strategy: '{strategy}'. Use 'full' or 'upsert_only'.")

        return summary 