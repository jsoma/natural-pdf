"""Implementation of the SearchServiceProtocol using Haystack components."""

import logging
import os
from typing import List, Dict, Any, Optional, Union, Iterable
from pathlib import Path
from PIL import Image
import copy

# --- Haystack Imports ---
try:
    import haystack
    from haystack import Pipeline
    from haystack.dataclasses import Document as HaystackDocument
    from haystack.document_stores.types import DocumentStore, DuplicatePolicy
    from haystack_integrations.document_stores.chroma import ChromaDocumentStore
    from haystack.document_stores.in_memory import InMemoryDocumentStore
    from haystack.components.embedders import (
        SentenceTransformersTextEmbedder,
        SentenceTransformersDocumentEmbedder
    )
    # Import necessary retrievers, rankers etc. as needed for search()
    from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever # For InMem
    from haystack_integrations.components.retrievers.chroma import ChromaEmbeddingRetriever # Use embedding retriever
    # Need Ranker if used
    try:
        from haystack.components.rankers import CohereRanker
    except ImportError:
        CohereRanker = None

    # Don't define here, it's imported later
except ImportError:
    # Set flags/placeholders if Haystack isn't installed
    # Don't define here, it's imported later
    DocumentStore = object
    HaystackDocument = Dict
    ChromaDocumentStore = None
    InMemoryDocumentStore = None
    SentenceTransformersDocumentEmbedder = None
    SentenceTransformersTextEmbedder = None
    InMemoryEmbeddingRetriever = None
    ChromaEmbeddingRetriever = None # Fallback definition
    CohereRanker = None
    Pipeline = None
    DuplicatePolicy = None

# --- ChromaDB Client Import (for management) ---
try:
    import chromadb
    CHROMADB_AVAILABLE = True
except ImportError:
    chromadb = None
    CHROMADB_AVAILABLE = False

# --- Local Imports ---
from .search_service_protocol import SearchServiceProtocol, IndexConfigurationError, IndexExistsError
from .search_options import SearchOptions, TextSearchOptions, MultiModalSearchOptions, BaseSearchOptions
from .search_service_protocol import Indexable
from .haystack_utils import HAS_HAYSTACK_EXTRAS # <-- This is the canonical import

# --- Logging ---
logger = logging.getLogger(__name__)

# --- Default Configuration Values ---
DEFAULT_PERSIST_PATH = "./natural_pdf_index"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

class HaystackSearchService(SearchServiceProtocol):
    """
    Haystack-based implementation of the search service protocol.

    Manages ChromaDB (persistent) or InMemory (non-persistent) DocumentStores
    and uses Haystack components for embedding and retrieval.
    A single instance of this service is tied to a specific collection name.
    """

    def __init__(
        self,
        collection_name: str,
        persist: bool = False, # Store type configuration
        default_persist_path: str = DEFAULT_PERSIST_PATH,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL # Renamed for clarity
    ):
        """
        Initialize the service for a specific collection.

        Args:
            collection_name: The name of the index/collection this service instance manages.
            persist: If True, this service instance manages persistent ChromaDB stores.
                    If False, it manages transient InMemory stores.
            default_persist_path: Default path for persistent ChromaDB storage.
            embedding_model: The embedding model this service instance will use.
        """
        if not HAS_HAYSTACK_EXTRAS:
            raise ImportError("HaystackSearchService requires Haystack extras. Install with: pip install natural-pdf[haystack]")

        self.collection_name = collection_name # Store the collection name
        self._persist = persist # Store the persistence type for this instance
        self._default_persist_path = default_persist_path
        self._embedding_model = embedding_model # Store the configured model
        
        # Dictionary to hold InMemoryDocumentStore instances if not persisting
        self._in_memory_store: Optional[InMemoryDocumentStore] = None if persist else InMemoryDocumentStore()
        self._chroma_store: Optional[ChromaDocumentStore] = None # Lazy load

        logger.info(f"HaystackSearchService initialized for collection='{self.collection_name}' (persist={self._persist}, model='{self._embedding_model}'). Default path: '{self._default_persist_path}'")

    # --- Internal Helper Methods --- #

    def _get_store(
        self,
    ) -> DocumentStore:
        """Gets or creates the appropriate Haystack DocumentStore instance for this service's collection."""
        # Use the instance's configured persistence type and collection name
        if self._persist:
            if self._chroma_store is None:
                 # Lazy load Chroma store
                 logger.debug(f"Initializing ChromaDocumentStore for collection '{self.collection_name}'.")
                 self._chroma_store = ChromaDocumentStore(
                     persist_path=self._default_persist_path,
                     collection_name=self.collection_name # Use instance name
                 )
            return self._chroma_store
        else:
            # Return the instance's InMemory store
            if self._in_memory_store is None: # Should have been created in __init__ if persist=False
                 logger.warning(f"In-memory store for collection '{self.collection_name}' was not initialized. Creating now.")
                 self._in_memory_store = InMemoryDocumentStore()
            return self._in_memory_store

    def _get_document_embedder(
        self,
        device: Optional[str] = None
    ) -> SentenceTransformersDocumentEmbedder:
        """Creates the Haystack document embedder component."""
        model_name = self._embedding_model # Use instance model
        logger.debug(f"Creating SentenceTransformersDocumentEmbedder. Model: {model_name}, Device: {device or 'auto'}")
        if not SentenceTransformersDocumentEmbedder:
            raise ImportError("SentenceTransformersDocumentEmbedder is required but not available.")
        try:
            embedder = SentenceTransformersDocumentEmbedder(
                model=model_name,
                device=device,
            )
            embedder.warm_up()
            logger.info(f"Created SentenceTransformersDocumentEmbedder. Model: {model_name}, Device: {getattr(embedder, 'device', 'unknown')}")
            return embedder
        except Exception as e:
            logger.error(f"Failed to initialize SentenceTransformersDocumentEmbedder: {e}", exc_info=True)
            raise RuntimeError(f"Failed to initialize SentenceTransformersDocumentEmbedder with model '{model_name}'.") from e

    def _get_text_embedder(
        self,
        device: Optional[str] = None
    ) -> SentenceTransformersTextEmbedder:
        """Creates the Haystack text embedder component (for queries)."""
        model_name = self._embedding_model # Use instance model
        logger.debug(f"Creating SentenceTransformersTextEmbedder. Model: {model_name}, Device: {device or 'auto'}")
        if not SentenceTransformersTextEmbedder:
            raise ImportError("SentenceTransformersTextEmbedder is required but not available.")
        try:
            embedder = SentenceTransformersTextEmbedder(model=model_name, device=device)
            embedder.warm_up() 
            logger.info(f"Created SentenceTransformersTextEmbedder. Model: {model_name}, Device: {getattr(embedder, 'device', 'unknown')}")
            return embedder
        except Exception as e:
            logger.error(f"Failed to initialize SentenceTransformersTextEmbedder: {e}", exc_info=True)
            raise RuntimeError(f"Could not create SentenceTransformersTextEmbedder with model '{model_name}'") from e

    def _delete_chroma_collection(self) -> bool:
        """Internal helper to delete the ChromaDB collection managed by this service."""
        if not CHROMADB_AVAILABLE:
            logger.error("Cannot delete ChromaDB collection because 'chromadb' library is not installed.")
            raise ImportError("'chromadb' library required for collection deletion.")
        if not self._persist:
             logger.warning("Attempted to delete ChromaDB collection for a non-persistent service instance. Ignoring.")
             return False # Cannot delete if not persistent
        try:
            collection_name_to_delete = self.collection_name # Use instance collection name
            logger.warning(f"Attempting to delete existing ChromaDB collection '{collection_name_to_delete}' at path '{self._default_persist_path}'.")
            chroma_client = chromadb.PersistentClient(path=self._default_persist_path)
            try:
                chroma_client.delete_collection(name=collection_name_to_delete)
                logger.info(f"Successfully deleted existing ChromaDB collection '{collection_name_to_delete}'.")
                self._chroma_store = None # Reset lazy-loaded store
                return True
            except chromadb.errors.InvalidCollectionException:
                logger.info(f"ChromaDB collection '{collection_name_to_delete}' did not exist. No deletion needed.")
                return True # Deletion is effectively successful
            finally:
                pass # Cleanup if needed
        except ImportError as ie:
            raise ie
        except Exception as e:
            logger.error(f"Error during ChromaDB collection deletion '{self.collection_name}': {e}", exc_info=True)
            # Don't raise here, let index() decide based on force_reindex
            return False

    # --- Protocol Methods Implementation --- #

    def index(
        self,
        documents: Iterable[Indexable], # Accept Indexable objects
        embedder_device: Optional[str] = None,
        force_reindex: bool = False,
    ) -> None:
        # Need to consume the iterable to log count, or log differently
        # Let's convert to list for now, assuming size isn't prohibitive
        indexable_list = list(documents)
        logger.info(f"Index request for collection='{self.collection_name}', docs={len(indexable_list)}, model='{self._embedding_model}', force={force_reindex}, persist={self._persist}")
        
        if not indexable_list:
             logger.warning("No documents provided for indexing. Skipping.")
             return

        # --- 1. Handle Reindexing (Deletion before store/embedder init) ---
        if force_reindex:
             logger.info(f"Force reindex requested for collection '{self.collection_name}'.")
             if self._persist:
                 # Attempt deletion, raises ImportError if chromadb missing
                 deleted = self._delete_chroma_collection() # Uses self.collection_name
                 if not deleted:
                      # If deletion failed for other reasons, log and continue cautiously
                      logger.warning("Collection deletion failed, but force_reindex=True. Proceeding with indexing, but existing data/config may interfere.")
             else:
                 # For InMemory, force_reindex means we want a fresh store instance.
                 # Re-initialize the instance's in-memory store
                 logger.info(f"force_reindex=True: Re-initializing InMemory store for collection '{self.collection_name}'.")
                 self._in_memory_store = InMemoryDocumentStore() # Create a new one

        # REMOVED try...except around store retrieval
        # Let store initialization errors propagate directly
        store = self._get_store() # No argument needed

        # --- 3. Create Embedder ---
        # Errors during embedder creation will propagate from the helper
        embedder = self._get_document_embedder(embedder_device)

        # --- 4. Convert Indexable to Haystack Docs & Embed ---
        haystack_docs_to_embed: List[HaystackDocument] = []
        logger.info(f"Preparing Haystack Documents from {len(indexable_list)} indexable items...")
        # Consume Indexable items using the protocol methods
        for item in indexable_list:
            doc_id = item.get_id()
            metadata = item.get_metadata()
            content_obj = item.get_content() # This might be Page, Region, etc.

            # Determine content based on embedder type and content object
            # For now, assume text content is needed and try to extract it
            content_text = ""
            if isinstance(content_obj, str):
                 # If get_content() already returned text
                 content_text = content_obj
            elif hasattr(content_obj, 'extract_text') and callable(getattr(content_obj, 'extract_text')):
                 # If content object has extract_text (like Page or Region)
                 try:
                     content_text = content_obj.extract_text()
                     if not isinstance(content_text, str):
                         logger.warning(f"extract_text() on {type(content_obj)} did not return a string for doc '{doc_id}'. Using str().")
                         content_text = str(content_obj)
                 except Exception as extraction_error:
                     logger.error(f"Error calling extract_text() on {type(content_obj)} for doc '{doc_id}': {extraction_error}. Using str().", exc_info=False)
                     content_text = str(content_obj)
            else:
                 # Attempt to convert to string as fallback if no obvious text method
                 logger.warning(f"Could not extract text from content type {type(content_obj)} obtained via get_content() for doc '{doc_id}'. Using str().")
                 content_text = str(content_obj)

            # Construct HaystackDocument using data from Indexable protocol methods
            haystack_doc = HaystackDocument(
                id=doc_id, # Use ID from get_id()
                content=content_text,
                meta=metadata # Use metadata from get_metadata()
            )
            haystack_docs_to_embed.append(haystack_doc)

        if not haystack_docs_to_embed:
             logger.warning("No Haystack documents were prepared. Check conversion logic and input data.")
             return

        logger.info(f"Embedding {len(haystack_docs_to_embed)} documents using '{self._embedding_model}'...")
        try:
            # Embed the documents
            embedding_results = embedder.run(documents=haystack_docs_to_embed)
            embedded_docs = embedding_results["documents"]
            logger.info(f"Successfully embedded {len(embedded_docs)} documents.")

        except haystack.errors.dimensionality_mismatch.InvalidDimensionError as dim_error:
            # Keep specific catch for dimension mismatch - provides useful context
            error_msg = f"Indexing failed for collection '{self.collection_name}'. Dimension mismatch: {dim_error}. "
            error_msg += f"Ensure the embedding model ('{self._embedding_model}') matches the expected dimension of the store. "
            if self._persist:
                error_msg += f"If the collection already exists at '{self._default_persist_path}', it might have been created with a different model. "
                error_msg += "Try deleting the persistent storage directory or using force_reindex=True."
            else:
                 error_msg += "This usually indicates an issue with the embedder setup or Haystack compatibility."
            logger.error(error_msg, exc_info=True)
            raise IndexConfigurationError(error_msg) from dim_error
        # REMOVED broad except Exception for embedding errors. Let them propagate.

        # --- 5. Write Embedded Documents to Store ---
        logger.info(f"Writing {len(embedded_docs)} embedded documents to store '{self.collection_name}'...")
        # REMOVED try...except around store writing. Let errors propagate.
        write_result = store.write_documents(
             documents=embedded_docs,
             policy=DuplicatePolicy.OVERWRITE # Or configure as needed
        )
        logger.info(f"Successfully wrote {write_result} documents to store '{self.collection_name}'.")
        # --- Add explicit count check after writing ---
        logger.info(f"Store '{self.collection_name}' document count after write: {store.count_documents()}")
        # --- End count check ---

    def search(
        self,
        query: Any, # Changed from Union[str, Path, Image.Image] to Any
        options: BaseSearchOptions,
    ) -> List[Dict[str, Any]]:
        logger.info(f"Search request for collection='{self.collection_name}', query_type={type(query).__name__}, options={options}")

        store = self._get_store() # Let errors propagate

        # --- 1. Handle Query Type and Embedding ---
        # This implementation currently only supports text query embedding.
        # TODO: Refactor or extend for multimodal queries based on service capabilities/options.
        query_embedding = None
        query_text = ""
        if isinstance(query, (str, os.PathLike)):
             if isinstance(query, os.PathLike):
                 logger.warning("Image path query received, but multimodal search not fully implemented. Treating as text path string.")
                 query_text = str(query)
             else:
                 query_text = query

             text_embedder = self._get_text_embedder()
             embedding_result = text_embedder.run(text=query_text)
             query_embedding = embedding_result["embedding"]
             if not query_embedding:
                 raise ValueError("Text embedder did not return an embedding for the query.")
             logger.debug(f"Successfully generated query text embedding (dim: {len(query_embedding)}).")

        elif isinstance(query, Image.Image):
             logger.error("Multimodal query (PIL Image) is not yet supported by this service implementation.")
             raise NotImplementedError("Search with PIL Image queries is not implemented in HaystackSearchService.")
        # Check if query is Indexable and try extracting text?
        elif hasattr(query, 'extract_text') and callable(getattr(query, 'extract_text')):
             logger.debug(f"Query type {type(query).__name__} has extract_text. Extracting text for search.")
             try:
                 query_text = query.extract_text()
                 if not query_text or not query_text.strip():
                     logger.warning(f"Query object {type(query).__name__} provided empty text. Returning no results.")
                     return []
                 # Embed the extracted text
                 text_embedder = self._get_text_embedder()
                 embedding_result = text_embedder.run(text=query_text)
                 query_embedding = embedding_result["embedding"]
                 if not query_embedding:
                     raise ValueError(f"Text embedder did not return an embedding for text extracted from {type(query).__name__}.")
                 logger.debug(f"Successfully generated query embedding from extracted text (dim: {len(query_embedding)}).")
             except Exception as e:
                 logger.error(f"Failed to extract or embed text from query object {type(query).__name__}: {e}", exc_info=True)
                 raise RuntimeError("Query text extraction or embedding failed.") from e

        else:
             # Raise specific error for unsupported types by this implementation
             raise TypeError(f"Unsupported query type for HaystackSearchService: {type(query)}")

        # --- 2. Select Retriever based on Store Type ---
        retriever = None
        if isinstance(store, ChromaDocumentStore):
            if not ChromaEmbeddingRetriever:
                raise ImportError("ChromaEmbeddingRetriever is required but not available.")
            retriever = ChromaEmbeddingRetriever(document_store=store)
        elif isinstance(store, InMemoryDocumentStore):
             retriever = InMemoryEmbeddingRetriever(document_store=store)
        else:
             # Raise specific error for unsupported store
             raise TypeError(f"Cannot perform search with store type {type(store)}.")

        # --- 3. Build Retrieval Pipeline ---
        pipeline = Pipeline()
        pipeline.add_component("retriever", retriever)
        # Add Ranker logic (remains the same)
        # ... (ranker setup if needed)

        # --- 4. Prepare Filters (remains the same) ---
        haystack_filters = options.filters
        if haystack_filters:
             logger.debug(f"Applying filters: {haystack_filters}")

        # --- 5. Prepare Retriever Input Data (Dynamically) ---
        retriever_input_data = {
            "filters": haystack_filters,
            "top_k": options.top_k
        }
        # Both InMemoryEmbeddingRetriever and ChromaEmbeddingRetriever expect 'query_embedding'
        retriever_input_data["query_embedding"] = query_embedding
        logger.debug(f"Providing 'query_embedding' to {type(retriever).__name__}.")

        # --- 6. Run Retrieval ---
        try:
            logger.info(f"Running retrieval pipeline for collection '{self.collection_name}'...")
            result = pipeline.run(
                data={"retriever": retriever_input_data}
                # ... (ranker data if needed)
            )

            # --- 7. Format Results ---
            if "retriever" in result and "documents" in result["retriever"]:
                retrieved_docs: List[HaystackDocument] = result["retriever"]["documents"]
                logger.info(f"Retrieved {len(retrieved_docs)} documents.")
                # Format results (remains the same)
                final_results = []
                for doc in retrieved_docs:
                    # Include content_hash in returned metadata if present
                    meta_with_hash = doc.meta
                    # No need to explicitly add hash here if Haystack store preserves it
                    result_dict = {
                         "content_snippet": doc.content[:200] if doc.content else "",
                         "score": doc.score if doc.score is not None else 0.0,
                         "page_number": meta_with_hash.get("page_number", None),
                         "pdf_path": meta_with_hash.get("pdf_path", None),
                         "metadata": meta_with_hash, # Pass full metadata
                         # "_haystack_document": doc # Optionally include full object
                    }
                    final_results.append(result_dict)
                return final_results
            else:
                 logger.warning("Pipeline result did not contain expected retriever output.")
                 return []

        except FileNotFoundError:
             # Keep specific catch for collection not found during retrieval
             logger.error(f"Search failed: Collection '{self.collection_name}' not found at path '{self._default_persist_path}'.")
             raise # Re-raise the specific FileNotFoundError
        # REMOVED broad except Exception for pipeline execution. Let errors propagate.

    def delete_index(
        self,
    ) -> bool:
        """
        Deletes the entire index/collection managed by this service instance.

        Returns:
            True if deletion was successful or collection didn't exist, False otherwise.
        """
        logger.warning(f"Request to delete index for collection '{self.collection_name}'.")
        if self._persist:
            # Delegate to internal ChromaDB deletion helper
            return self._delete_chroma_collection()
        else:
            # For InMemory, "deleting" means re-initializing the store
            logger.info(f"Re-initializing InMemory store for '{self.collection_name}' as deletion request.")
            self._in_memory_store = InMemoryDocumentStore()
            return True # Considered successful

    def index_exists(
        self,
    ) -> bool:
        """
        Checks if the index/collection managed by this service instance exists.
        NOTE: For ChromaDB, this may involve trying to connect.
        For InMemory, it checks if the internal store object exists and has documents.
        """
        logger.debug(f"Checking existence of index for collection '{self.collection_name}'.")
        store = self._get_store() # Get the store instance
        try:
            count = store.count_documents()
            exists = count > 0
            logger.debug(f"Store type {type(store).__name__} for '{self.collection_name}' exists and has {count} documents: {exists}")
            return exists
        except Exception as e:
             # Catch errors during count_documents (e.g., connection error for persistent stores)
             logger.warning(f"Could not count documents in store for collection '{self.collection_name}' to check existence: {e}", exc_info=False)
             # Special handling for ChromaDB trying to connect to non-existent path? Check Haystack behavior.
             # Assume not exists if count fails
             return False

    # --- Sync Methods Implementation ---

    def list_documents(self, include_metadata: bool = False, **kwargs) -> List[Dict]:
        """Retrieves documents, required for sync.
           NOTE: Haystack's filter_documents is the closest match.
                 Fetches all docs if filters=None.
        """
        logger.debug(f"Listing documents for collection '{self.collection_name}' (include_metadata={include_metadata})...")
        store = self._get_store()
        try:
            # Use filter_documents with no filters to get all
            # This might be inefficient for very large stores.
            haystack_docs = store.filter_documents(filters=kwargs.get('filters')) # Pass filters if provided via kwargs
            logger.info(f"Retrieved {len(haystack_docs)} documents from store.")
            # Convert to simple dicts
            results = []
            for doc in haystack_docs:
                 doc_dict = {"id": doc.id} # ID is essential
                 if include_metadata:
                     # Ensure content_hash is included if it exists in meta
                     doc_dict["meta"] = doc.meta
                 # Optionally include content? Protocol doesn't require it.
                 # doc_dict["content"] = doc.content
                 results.append(doc_dict)
            return results
        except Exception as e:
            logger.error(f"Failed to list documents from store '{self.collection_name}': {e}", exc_info=True)
            raise RuntimeError(f"Failed to list documents from store '{self.collection_name}'.") from e


    def delete_documents(self, ids: List[str]) -> None:
        """Deletes documents by ID, required for sync."""
        if not ids:
            logger.debug("No document IDs provided for deletion. Skipping.")
            return
        logger.warning(f"Request to delete {len(ids)} documents from collection '{self.collection_name}'.")
        store = self._get_store()
        try:
            store.delete_documents(ids=ids)
            logger.info(f"Successfully deleted {len(ids)} documents (if they existed). Store count now: {store.count_documents()}")
        except Exception as e:
            logger.error(f"Failed to delete documents with IDs {ids} from store '{self.collection_name}': {e}", exc_info=True)
            raise RuntimeError(f"Failed to delete documents from store '{self.collection_name}'.") from e 