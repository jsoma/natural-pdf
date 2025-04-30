"""Implementation of the SearchServiceProtocol using Haystack components."""

import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

from PIL import Image

# Import sentence-transformers for dimension calculation
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

# --- Haystack Imports ---
try:
    import haystack
    from haystack import Pipeline
    from haystack.components.embedders import (
        SentenceTransformersDocumentEmbedder,
        SentenceTransformersTextEmbedder,
    )

    # Import InMemory Store & Retriever unconditionally
    from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
    from haystack.dataclasses import Document as HaystackDocument
    from haystack.document_stores.in_memory import InMemoryDocumentStore
    from haystack.document_stores.types import DocumentStore, DuplicatePolicy

    # Conditional LanceDB Imports
    try:
        from lancedb_haystack import LanceDBDocumentStore, LanceDBEmbeddingRetriever

        LANCEDB_HAYSTACK_AVAILABLE = True
    except ImportError:
        LanceDBDocumentStore = None
        LanceDBEmbeddingRetriever = None
        LANCEDB_HAYSTACK_AVAILABLE = False

    # Removed Chroma Imports

    # Need Ranker if used
    try:
        from haystack.components.rankers import CohereRanker
    except ImportError:
        CohereRanker = None

except ImportError:
    # Set flags/placeholders if Haystack isn't installed
    DocumentStore = object
    HaystackDocument = Dict
    InMemoryDocumentStore = None
    LanceDBDocumentStore = None
    SentenceTransformersDocumentEmbedder = None
    SentenceTransformersTextEmbedder = None
    InMemoryEmbeddingRetriever = None
    LanceDBEmbeddingRetriever = None
    CohereRanker = None
    Pipeline = None
    DuplicatePolicy = None
    LANCEDB_HAYSTACK_AVAILABLE = False

# LanceDB Client Import (for management)
try:
    import lancedb

    LANCEDB_CLIENT_AVAILABLE = True
except ImportError:
    lancedb = None
    LANCEDB_CLIENT_AVAILABLE = False

# Removed ChromaDB Client Import

from .haystack_utils import HAS_HAYSTACK_EXTRAS
from .search_options import (
    BaseSearchOptions,
    TextSearchOptions,
)

# --- Local Imports ---
from .search_service_protocol import (
    Indexable,
    IndexConfigurationError,
    SearchServiceProtocol,
)

logger = logging.getLogger(__name__)

# --- Default Configuration Values ---
DEFAULT_PERSIST_PATH = "./natural_pdf_index"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class HaystackSearchService(SearchServiceProtocol):
    """
    Haystack-based implementation of the search service protocol.

    Manages LanceDB (persistent) or InMemory (non-persistent) DocumentStores
    and uses Haystack components for embedding and retrieval.
    A single instance of this service is tied to a specific table name (LanceDB)
    or implicitly managed (InMemory).
    """

    def __init__(
        self,
        table_name: str,
        persist: bool = False,
        uri: str = DEFAULT_PERSIST_PATH,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    ):
        """
        Initialize the service for a specific LanceDB table or an InMemory store.

        Args:
            table_name: The name of the LanceDB table (if persist=True).
            persist: If True, this service instance manages a persistent LanceDB store.
                    If False, it manages a transient InMemory store.
            uri: Path/URI for the LanceDB database directory (if persist=True).
            embedding_model: The embedding model this service instance will use.
                               Required for LanceDB to know embedding dimensions.
        """
        if not HAS_HAYSTACK_EXTRAS:
            raise ImportError(
                "HaystackSearchService requires Haystack extras. Install with: pip install natural-pdf[haystack]"
            )

        self.table_name = table_name
        self._persist = persist
        self._uri = uri
        self._embedding_model = embedding_model
        self._embedding_dims: Optional[int] = None

        # Store instances (lazy loaded)
        self._in_memory_store: Optional[InMemoryDocumentStore] = None
        self._lancedb_store: Optional[LanceDBDocumentStore] = None

        # Eagerly create InMemoryStore if not persisting
        if not self._persist:
            if not InMemoryDocumentStore:
                raise ImportError(
                    "InMemoryDocumentStore not available. Cannot create non-persistent service."
                )
            self._in_memory_store = InMemoryDocumentStore()
            logger.info(
                f"HaystackSearchService initialized for InMemory store (table_name '{self.table_name}' ignored). Model: '{self._embedding_model}'"
            )
        else:
            # Check LanceDB availability if persisting
            if not LANCEDB_HAYSTACK_AVAILABLE:
                raise ImportError(
                    "LanceDB persistent store requires lancedb-haystack. Install with: pip install lancedb-haystack"
                )
            if not SentenceTransformer:
                raise ImportError(
                    "LanceDB persistent store requires sentence-transformers to determine embedding dimensions. Install with: pip install sentence-transformers"
                )
            # Calculate embedding dimensions needed for LanceDB initialization
            self._calculate_embedding_dims()
            logger.info(
                f"HaystackSearchService initialized for LanceDB table='{self.table_name}' at uri='{self._uri}'. Model: '{self._embedding_model}', Dims: {self._embedding_dims}"
            )

    # --- Internal Helper Methods ---

    def _calculate_embedding_dims(self) -> None:
        """Calculates and stores embedding dimensions from the model name."""
        if self._embedding_dims is None:
            if not SentenceTransformer:
                raise ImportError(
                    "sentence-transformers library is required to determine embedding dimensions."
                )
            try:
                model = SentenceTransformer(self._embedding_model)
                dims = model.get_sentence_embedding_dimension()
                if not dims:
                    raise ValueError(
                        f"Could not determine embedding dimension for model: {self._embedding_model}"
                    )
                self._embedding_dims = dims
                logger.debug(
                    f"Determined embedding dimension: {self._embedding_dims} for model '{self._embedding_model}'"
                )
            except Exception as e:
                logger.error(
                    f"Failed to load SentenceTransformer model '{self._embedding_model}' to get dimensions: {e}",
                    exc_info=True,
                )
                raise RuntimeError(
                    f"Failed to determine embedding dimension for model '{self._embedding_model}'."
                ) from e

    def _get_store(self) -> DocumentStore:
        """Gets or creates the appropriate Haystack DocumentStore instance."""
        if self._persist:
            if not LanceDBDocumentStore:
                raise ImportError("LanceDBDocumentStore not available.")
            if self._lancedb_store is None:
                logger.debug(
                    f"Initializing LanceDBDocumentStore for table '{self.table_name}' at uri '{self._uri}'."
                )
                if self._embedding_dims is None:
                    logger.warning(
                        "Embedding dimensions not calculated before getting store. Calculating now."
                    )
                    self._calculate_embedding_dims()

                self._lancedb_store = LanceDBDocumentStore(
                    database=self._uri,
                    table_name=self.table_name,
                    embedding_dims=self._embedding_dims,
                )
                logger.info(
                    f"Initialized LanceDBDocumentStore for table '{self.table_name}' (Dims: {self._embedding_dims})"
                )
            return self._lancedb_store
        else:
            if self._in_memory_store is None:
                logger.warning("In-memory store was not initialized. Creating now.")
                if not InMemoryDocumentStore:
                    raise ImportError("InMemoryDocumentStore not available.")
                self._in_memory_store = InMemoryDocumentStore()
            return self._in_memory_store

    def _get_document_embedder(
        self, device: Optional[str] = None
    ) -> SentenceTransformersDocumentEmbedder:
        """Creates the Haystack document embedder component."""
        model_name = self._embedding_model
        logger.debug(
            f"Creating SentenceTransformersDocumentEmbedder. Model: {model_name}, Device: {device or 'auto'}"
        )
        if not SentenceTransformersDocumentEmbedder:
            raise ImportError("SentenceTransformersDocumentEmbedder is required but not available.")
        try:
            embedder = SentenceTransformersDocumentEmbedder(
                model=model_name,
                device=device,
            )
            embedder.warm_up()
            logger.info(
                f"Created SentenceTransformersDocumentEmbedder. Model: {model_name}, Device: {getattr(embedder, 'device', 'unknown')}"
            )
            return embedder
        except Exception as e:
            logger.error(
                f"Failed to initialize SentenceTransformersDocumentEmbedder: {e}", exc_info=True
            )
            raise RuntimeError(
                f"Failed to initialize SentenceTransformersDocumentEmbedder with model '{model_name}'."
            ) from e

    def _get_text_embedder(self, device: Optional[str] = None) -> SentenceTransformersTextEmbedder:
        """Creates the Haystack text embedder component (for queries)."""
        model_name = self._embedding_model
        logger.debug(
            f"Creating SentenceTransformersTextEmbedder. Model: {model_name}, Device: {device or 'auto'}"
        )
        if not SentenceTransformersTextEmbedder:
            raise ImportError("SentenceTransformersTextEmbedder is required but not available.")
        try:
            embedder = SentenceTransformersTextEmbedder(model=model_name, device=device)
            embedder.warm_up()
            logger.info(
                f"Created SentenceTransformersTextEmbedder. Model: {model_name}, Device: {getattr(embedder, 'device', 'unknown')}"
            )
            return embedder
        except Exception as e:
            logger.error(
                f"Failed to initialize SentenceTransformersTextEmbedder: {e}", exc_info=True
            )
            raise RuntimeError(
                f"Could not create SentenceTransformersTextEmbedder with model '{model_name}'"
            ) from e

    def _delete_lancedb_table(self) -> bool:
        """Internal helper to delete the LanceDB table managed by this service."""
        if not self._persist:
            logger.warning(
                "Attempted to delete LanceDB table for a non-persistent service instance. Ignoring."
            )
            return False

        if not LANCEDB_CLIENT_AVAILABLE:
            logger.error("Cannot delete LanceDB table because 'lancedb' library is not installed.")
            raise ImportError("'lancedb' library required for table deletion.")

        table_name_to_delete = self.table_name
        db_uri = self._uri
        logger.warning(
            f"Attempting to delete existing LanceDB table '{table_name_to_delete}' at uri '{db_uri}'."
        )
        try:
            db = lancedb.connect(db_uri)
            table_names = db.table_names()
            if table_name_to_delete in table_names:
                db.drop_table(table_name_to_delete)
                logger.info(
                    f"Successfully deleted existing LanceDB table '{table_name_to_delete}'."
                )
            else:
                logger.info(
                    f"LanceDB table '{table_name_to_delete}' did not exist. No deletion needed."
                )

            self._lancedb_store = None
            return True
        except Exception as e:
            logger.error(
                f"Error during LanceDB table deletion '{table_name_to_delete}' at '{db_uri}': {e}",
                exc_info=True,
            )
            return False

    # --- Protocol Methods Implementation ---

    def index(
        self,
        documents: Iterable[Indexable],
        embedder_device: Optional[str] = None,
        force_reindex: bool = False,
    ) -> None:
        indexable_list = list(documents)
        logger.info(
            f"Index request for table='{self.table_name}', docs={len(indexable_list)}, model='{self._embedding_model}', force={force_reindex}, persist={self._persist}"
        )

        if not indexable_list:
            logger.warning("No documents provided for indexing. Skipping.")
            return

        # Handle Reindexing
        if force_reindex:
            logger.info(f"Force reindex requested for table '{self.table_name}'.")
            if self._persist:
                deleted = self._delete_lancedb_table()
                if not deleted:
                    logger.warning(
                        "LanceDB table deletion failed, but force_reindex=True. Proceeding with indexing, but existing data/config may interfere."
                    )
            else:
                # For InMemory, re-initialize the instance's store
                logger.info(f"force_reindex=True: Re-initializing InMemory store.")
                if not InMemoryDocumentStore:
                    raise ImportError("InMemoryDocumentStore not available.")
                self._in_memory_store = InMemoryDocumentStore()

        # Get Store
        store = self._get_store()

        # Create Embedder
        embedder = self._get_document_embedder(embedder_device)

        # Convert Indexable to Haystack Docs & Embed
        haystack_docs_to_embed: List[HaystackDocument] = []
        logger.info(f"Preparing Haystack Documents from {len(indexable_list)} indexable items...")
        for item in indexable_list:
            doc_id = item.get_id()
            metadata = item.get_metadata()
            content_obj = item.get_content()
            content_text = ""
            if isinstance(content_obj, str):
                content_text = content_obj
            elif hasattr(content_obj, "extract_text") and callable(
                getattr(content_obj, "extract_text")
            ):
                try:
                    content_text = content_obj.extract_text()
                    if not isinstance(content_text, str):
                        logger.warning(
                            f"extract_text() on {type(content_obj)} did not return a string for doc '{doc_id}'. Using str()."
                        )
                        content_text = str(content_obj)
                except Exception as extraction_error:
                    logger.error(
                        f"Error calling extract_text() on {type(content_obj)} for doc '{doc_id}': {extraction_error}. Using str().",
                        exc_info=False,
                    )
                    content_text = str(content_obj)
            else:
                logger.warning(
                    f"Could not extract text from content type {type(content_obj)} obtained via get_content() for doc '{doc_id}'. Using str()."
                )
                content_text = str(content_obj)

            haystack_doc = HaystackDocument(id=doc_id, content=content_text, meta=metadata)
            haystack_docs_to_embed.append(haystack_doc)

        if not haystack_docs_to_embed:
            logger.warning(
                "No Haystack documents were prepared. Check conversion logic and input data."
            )
            return

        logger.info(
            f"Embedding {len(haystack_docs_to_embed)} documents using '{self._embedding_model}'..."
        )
        try:
            embedding_results = embedder.run(documents=haystack_docs_to_embed)
            embedded_docs = embedding_results["documents"]
            logger.info(f"Successfully embedded {len(embedded_docs)} documents.")

        except haystack.errors.dimensionality_mismatch.InvalidDimensionError as dim_error:
            error_msg = (
                f"Indexing failed for table '{self.table_name}'. Dimension mismatch: {dim_error}. "
            )
            error_msg += f"Ensure the embedding model ('{self._embedding_model}', Dim: {self._embedding_dims}) matches the expected dimension of the store. "
            if self._persist:
                error_msg += f"If the table already exists at '{self._uri}', it might have been created with a different model/dimension. "
                error_msg += f"Try deleting the LanceDB table directory ('{os.path.join(self._uri, self.table_name + '.lance')}') or using force_reindex=True."
            else:
                error_msg += "This usually indicates an issue with the embedder setup or Haystack compatibility."
            logger.error(error_msg, exc_info=True)
            raise IndexConfigurationError(error_msg) from dim_error

        # Write Embedded Documents to Store
        logger.info(
            f"Writing {len(embedded_docs)} embedded documents to store (Table/Type: '{self.table_name if self._persist else 'InMemory'}')..."
        )
        write_result = store.write_documents(
            documents=embedded_docs, policy=DuplicatePolicy.OVERWRITE
        )
        logger.info(f"Successfully wrote {write_result} documents to store.")
        try:
            count = store.count_documents()
            logger.info(f"Store document count after write: {count}")
        except Exception as count_error:
            logger.warning(f"Could not get document count after write: {count_error}")

    def search(
        self,
        query: Any,
        options: BaseSearchOptions,
    ) -> List[Dict[str, Any]]:
        logger.info(
            f"Search request for table/store='{self.table_name if self._persist else 'InMemory'}', query_type={type(query).__name__}, options={options}"
        )

        store = self._get_store()

        # Handle Query Type and Embedding
        query_embedding = None
        query_text = ""
        if isinstance(query, (str, os.PathLike)):
            if isinstance(query, os.PathLike):
                logger.warning("Image path query received, treating as text path string.")
                query_text = str(query)
            else:
                query_text = query
            text_embedder = self._get_text_embedder()
            embedding_result = text_embedder.run(text=query_text)
            query_embedding = embedding_result["embedding"]
            if not query_embedding:
                raise ValueError("Text embedder did not return an embedding for the query.")
            logger.debug(
                f"Successfully generated query text embedding (dim: {len(query_embedding)})."
            )
        elif isinstance(query, Image.Image):
            logger.error("Multimodal query (PIL Image) is not yet supported.")
            raise NotImplementedError("Search with PIL Image queries is not implemented.")
        elif hasattr(query, "extract_text") and callable(getattr(query, "extract_text")):
            logger.debug(f"Query type {type(query).__name__} has extract_text. Extracting text.")
            try:
                query_text = query.extract_text()
                if not query_text or not query_text.strip():
                    logger.warning(
                        f"Query object {type(query).__name__} provided empty text. Returning no results."
                    )
                    return []
                text_embedder = self._get_text_embedder()
                embedding_result = text_embedder.run(text=query_text)
                query_embedding = embedding_result["embedding"]
                if not query_embedding:
                    raise ValueError(
                        f"Text embedder did not return embedding for text from {type(query).__name__}."
                    )
                logger.debug(
                    f"Generated query embedding from extracted text (dim: {len(query_embedding)})."
                )
            except Exception as e:
                logger.error(
                    f"Failed to extract/embed text from query object {type(query).__name__}: {e}",
                    exc_info=True,
                )
                raise RuntimeError("Query text extraction or embedding failed.") from e
        else:
            raise TypeError(f"Unsupported query type for HaystackSearchService: {type(query)}")

        # Select Retriever based on Store Type
        retriever = None
        # Check if LanceDB is available *before* checking isinstance
        if (
            LANCEDB_HAYSTACK_AVAILABLE
            and LanceDBDocumentStore
            and isinstance(store, LanceDBDocumentStore)
        ):
            if not LanceDBEmbeddingRetriever:
                raise ImportError("LanceDBEmbeddingRetriever is required but not available.")
            retriever = LanceDBEmbeddingRetriever(document_store=store)
        # Check if InMemory is available *before* checking isinstance
        elif (
            InMemoryDocumentStore
            and InMemoryEmbeddingRetriever
            and isinstance(store, InMemoryDocumentStore)
        ):
            # No separate HAS_INMEMORY flag, check if classes are not None
            retriever = InMemoryEmbeddingRetriever(document_store=store)
        else:
            # Improved error message if store type is unexpected
            store_type_name = type(store).__name__
            available_integrations = []
            if LANCEDB_HAYSTACK_AVAILABLE and LanceDBDocumentStore:
                available_integrations.append("LanceDB")
            if InMemoryDocumentStore:
                available_integrations.append("InMemory")

            if not available_integrations:
                raise TypeError(
                    f"Cannot perform search: No supported document store integrations (LanceDB, InMemory) seem to be available. "
                    f"Check Haystack installation."
                )
            # Check if the store type matches one of the available integrations' expected types
            elif (
                LANCEDB_HAYSTACK_AVAILABLE
                and LanceDBDocumentStore
                and isinstance(store, LanceDBDocumentStore)
            ) or (InMemoryDocumentStore and isinstance(store, InMemoryDocumentStore)):
                # This case implies the retriever class (e.g., LanceDBEmbeddingRetriever) might be missing
                missing_retriever = ""
                if isinstance(store, LanceDBDocumentStore):
                    missing_retriever = "LanceDBEmbeddingRetriever"
                if isinstance(store, InMemoryDocumentStore):
                    missing_retriever = "InMemoryEmbeddingRetriever"
                raise ImportError(
                    f"Store type '{store_type_name}' is supported, but its retriever component '{missing_retriever}' failed to import or is unavailable."
                )
            else:  # Store type doesn't match any known/available store type
                raise TypeError(
                    f"Cannot perform search with unexpected store type '{store_type_name}'. "
                    f"Available integrations: {', '.join(available_integrations)}."
                )

        # This check remains as a final safeguard, though the logic above should catch most issues
        if not retriever:
            raise RuntimeError(
                f"Failed to select a suitable retriever for store type {type(store).__name__}. Please check dependencies and integration availability."
            )

        logger.debug(f"Selected retriever: {type(retriever).__name__}")

        # Build Retrieval Pipeline
        pipeline = Pipeline()
        pipeline.add_component("retriever", retriever)

        # Prepare Filters
        haystack_filters = options.filters
        if haystack_filters:
            logger.debug(f"Applying filters: {haystack_filters}")

        # Prepare Retriever Input Data
        retriever_input_data = {"filters": haystack_filters, "top_k": options.top_k}
        retriever_input_data["query_embedding"] = query_embedding
        logger.debug(f"Providing 'query_embedding' to {type(retriever).__name__}.")

        # Run Retrieval
        try:
            logger.info(
                f"Running retrieval pipeline for table/store '{self.table_name if self._persist else 'InMemory'}'..."
            )
            result = pipeline.run(data={"retriever": retriever_input_data})

            # Format Results
            if "retriever" in result and "documents" in result["retriever"]:
                retrieved_docs: List[HaystackDocument] = result["retriever"]["documents"]
                logger.info(f"Retrieved {len(retrieved_docs)} documents.")
                final_results = []
                for doc in retrieved_docs:
                    meta_with_hash = doc.meta
                    result_dict = {
                        "content_snippet": doc.content[:200] if doc.content else "",
                        "score": doc.score if doc.score is not None else 0.0,
                        "page_number": meta_with_hash.get("page_number", None),
                        "pdf_path": meta_with_hash.get("pdf_path", None),
                        "metadata": meta_with_hash,
                    }
                    final_results.append(result_dict)
                return final_results
            else:
                logger.warning("Pipeline result did not contain expected retriever output.")
                return []

        except FileNotFoundError:
            logger.error(
                f"Search failed: Could not access path for table/store '{self.table_name if self._persist else 'InMemory'}' (URI: '{self._uri if self._persist else 'N/A'}')."
            )
            raise

    def delete_index(self) -> bool:
        """
        Deletes the entire LanceDB table or resets the InMemory store.

        Returns:
            True if deletion was successful or table/store didn't exist, False otherwise.
        """
        if self._persist:
            logger.warning(
                f"Request to delete LanceDB table '{self.table_name}' at uri '{self._uri}'."
            )
            return self._delete_lancedb_table()
        else:
            logger.info("Request to delete InMemory store (re-initializing).)")
            if not InMemoryDocumentStore:
                raise ImportError("InMemoryDocumentStore not available.")
            self._in_memory_store = InMemoryDocumentStore()
            return True

    def index_exists(self) -> bool:
        """
        Checks if the LanceDB table or InMemory store exists and has documents.
        NOTE: For LanceDB, this tries to count documents, implicitly checking connection/table existence.
              For InMemory, it checks if the internal store object exists and has documents.
        """
        store_name = self.table_name if self._persist else "InMemory"
        logger.debug(
            f"Checking existence of index for '{store_name}'. URI: '{self._uri if self._persist else 'N/A'}'"
        )
        try:
            store = self._get_store()
            count = store.count_documents()
            exists = count > 0
            logger.debug(
                f"Store type {type(store).__name__} for '{store_name}' exists and has {count} documents: {exists}"
            )
            return exists
        except ImportError as ie:
            logger.error(f"Import error checking index existence for '{store_name}': {ie}")
            return False
        except Exception as e:
            logger.warning(
                f"Could not confirm existence or count documents in store for '{store_name}': {e}",
                exc_info=False,
            )
            return False

    # --- Sync Methods Implementation ---

    def list_documents(self, include_metadata: bool = False, **kwargs) -> List[Dict]:
        """Retrieves documents, required for sync."""
        store_name = self.table_name if self._persist else "InMemory"
        logger.debug(
            f"Listing documents for '{store_name}' (include_metadata={include_metadata})..."
        )
        store = self._get_store()
        try:
            haystack_docs = store.filter_documents(filters=kwargs.get("filters"))
            logger.info(f"Retrieved {len(haystack_docs)} documents from store '{store_name}'.")
            results = []
            for doc in haystack_docs:
                doc_dict = {"id": doc.id}
                if include_metadata:
                    doc_dict["meta"] = doc.meta
                results.append(doc_dict)
            return results
        except Exception as e:
            logger.error(f"Failed to list documents from store '{store_name}': {e}", exc_info=True)
            raise RuntimeError(f"Failed to list documents from store '{store_name}'.") from e

    def delete_documents(self, ids: List[str]) -> None:
        """Deletes documents by ID, required for sync."""
        store_name = self.table_name if self._persist else "InMemory"
        if not ids:
            logger.debug(f"No document IDs provided for deletion from '{store_name}'. Skipping.")
            return
        logger.warning(f"Request to delete {len(ids)} documents from '{store_name}'.")
        store = self._get_store()
        try:
            store.delete_documents(ids=ids)
            logger.info(
                f"Successfully requested deletion of {len(ids)} documents from '{store_name}'. Store count now: {store.count_documents()}"
            )
        except Exception as e:
            logger.error(
                f"Failed to delete documents with IDs {ids} from store '{store_name}': {e}",
                exc_info=True,
            )
            raise RuntimeError(f"Failed to delete documents from store '{store_name}'.") from e
