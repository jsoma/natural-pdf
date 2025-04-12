"""Defines the protocol for a search service."""
from typing import Protocol, List, Dict, Any, Optional, Union, Iterable
from pathlib import Path
from PIL import Image

# Use typing_extensions for Python < 3.8 compatibility if needed,
# otherwise, typing.Protocol is fine for >= 3.8
# from typing_extensions import Protocol

# Forward declare SearchOptions to avoid circular import if needed,
# or import if structure allows (assuming it's safe here)
from natural_pdf.search.search_options import SearchOptions, BaseSearchOptions

# Use Dict as placeholder for external Haystack Document type
HaystackDocument = Dict[str, Any]


class IndexConfigurationError(RuntimeError):
    """Custom exception for configuration mismatches during indexing."""
    pass


# Add new exception for sync/init safety
class IndexExistsError(RuntimeError):
    """Raised when attempting to index implicitly to an existing persistent index without force_reindex=True."""
    pass


class Indexable(Protocol):
    """
    Protocol defining the minimal interface for an object that can be indexed.
    Objects passed to SearchServiceProtocol.index must conform to this.
    """

    def get_id(self) -> str:
        """Return a unique identifier for this indexable item."""
        ...

    def get_metadata(self) -> Dict[str, Any]:
        """Return a dictionary of metadata associated with this item."""
        ...

    def get_content(self) -> Any:
        """
        Return the primary content of this item.
        The SearchService implementation will determine how to process this content
        (e.g., call .extract_text(), .to_image(), or handle directly).
        """
        ...

    # Added for syncing
    def get_content_hash(self) -> str:
        """
        Return a hash representing the indexable content.
        Used by SearchableMixin.sync_index to detect changes efficiently.
        Implementations should hash the same content used for generating embeddings.
        """
        ...


class SearchServiceProtocol(Protocol):
    """
    Protocol defining the interface for indexing and searching documents.

    Implementations of this protocol handle the specifics of interacting
    with a chosen search backend (e.g., Haystack with ChromaDB, Haystack In-Memory).
    An instance of a service implementing this protocol is tied to a specific collection name.
    """
    collection_name: str
    # Removed internal state hints (_persist, _embedding_model) - implementation detail

    def index(
        self,
        documents: Iterable[Indexable],
        embedder_device: Optional[str] = None,
        force_reindex: bool = False,
    ) -> None:
        """
        Indexes the provided documents into the collection managed by this service instance.

        Handles store and embedder creation/retrieval, configuration checks,
        re-indexing logic (including potential deletion), embedding, and writing.

        Args:
            documents: An iterable of objects conforming to the Indexable protocol.
            embedder_device: The device ('cpu', 'cuda', etc.) for the embedder.
                             Defaults defined by the implementation.
            force_reindex: If True, delete the entire existing collection
                           (if configuration permits) before indexing.

        Raises:
            IndexConfigurationError: If `force_reindex` is False and the existing
                                     collection has incompatible settings.
            ImportError: If required backend libraries are missing.
            RuntimeError: For other failures during indexing.
        """
        ...

    def search(
        self,
        query: Any, # Allow any query type, service implementation handles it
        options: BaseSearchOptions,
    ) -> List[Dict[str, Any]]:
        """
        Performs a search within the collection managed by this service instance.

        Args:
            query: The search query (type depends on service capabilities).
            options: SearchOptions object containing configuration like top_k, filters, etc.

        Returns:
            A list of result dictionaries, typically containing document content,
            metadata, and relevance scores.

        Raises:
            FileNotFoundError: If the collection managed by this service does not exist.
            RuntimeError: For other failures during search.
            TypeError: If the query type is incompatible with the backend/options.
        """
        ...

    def delete_index(
        self,
    ) -> bool:
        """
        Deletes the entire index/collection managed by this service instance.

        Returns:
            True if deletion was successful or collection didn't exist,
            False if deletion failed.

        Raises:
            ImportError: If required backend libraries are missing.
            RuntimeError: For backend errors during deletion.
        """
        ...

    def index_exists(
        self,
    ) -> bool:
        """
        Checks if the index/collection managed by this service instance exists.

        Returns:
            True if the index exists, False otherwise.

        Raises:
            ImportError: If required backend libraries are missing.
            RuntimeError: For backend errors during the check.
        """
        ...

    # --- Methods required for sync_index (full strategy) ---

    def list_documents(self, include_metadata: bool = False, **kwargs) -> List[Dict]:
        """
        Retrieves documents from the index, optionally including metadata.
        Required for the 'full' strategy in SearchableMixin.sync_index.

        Args:
            include_metadata: If True, include the 'meta' field in the returned dicts.
                              Metadata should include 'content_hash' if available.
            **kwargs: Additional backend-specific filtering or retrieval options.

        Returns:
            A list of dictionaries, each representing a document.
            Must include at least 'id'. If include_metadata=True, must include 'meta'.

        Raises:
            NotImplementedError: If the service does not support listing documents.
        """
        ...

    def delete_documents(self, ids: List[str]) -> None:
        """
        Deletes documents from the index based on their IDs.
        Required for the 'full' strategy in SearchableMixin.sync_index.

        Args:
            ids: A list of document IDs to delete.

        Raises:
            NotImplementedError: If the service does not support deleting documents by ID.
            RuntimeError: For backend errors during deletion.
        """
        ...

    # Optional: Add methods for getting index stats, etc.
    # def get_index_stats(self, collection_name: str) -> Dict[str, Any]: ... 