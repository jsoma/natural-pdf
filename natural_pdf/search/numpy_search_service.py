import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

import numpy as np

from .search_options import BaseSearchOptions
from .search_service_protocol import (
    Indexable,
    IndexConfigurationError,
    SearchResult,
    SearchServiceProtocol,
)

# Lazy import for SentenceTransformer to avoid heavy loading at module level
# from sentence_transformers import SentenceTransformer


logger = logging.getLogger(__name__)

DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def _get_sentence_transformer(model_name: str):
    """Lazy import and instantiation of SentenceTransformer."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


class NumpySearchService(SearchServiceProtocol):
    """Basic in-memory vector search implementation using NumPy."""

    collection_name: str

    def __init__(
        self,
        collection_name: str,
        persist: bool = False,
        uri: Optional[str] = None,
        embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
    ):
        if persist:
            raise RuntimeError(
                "Persistence requested but LanceDB is not installed. "
                "For persistent vector search, install LanceDB: pip install lancedb"
            )

        self.collection_name = collection_name
        self._embedding_model_name = embedding_model_name
        self.embedding_model = _get_sentence_transformer(self._embedding_model_name)
        self._embedding_dims = len(self.embedding_model.encode("test"))

        # Simple in-memory storage
        self._vectors = []
        self._documents = []
        self._metadata = []
        self._ids = []

        logger.info(
            f"NumpySearchService initialized for collection '{collection_name}' with model '{embedding_model_name}'"
        )

    def index(
        self,
        documents: Iterable[Indexable],
        embedder_device: Optional[str] = None,
        force_reindex: bool = False,
    ) -> None:
        if force_reindex:
            logger.info(
                f"Force reindex requested for collection '{self.collection_name}'. Clearing in-memory vectors."
            )
            self._vectors = []
            self._documents = []
            self._metadata = []
            self._ids = []

        items = list(documents)
        logger.info(f"Indexing {len(items)} documents for collection '{self.collection_name}'")

        if not items:
            logger.warning("No documents provided for indexing. Skipping.")
            return

        texts_to_embed = []
        items_info = []

        for item in items:
            doc_id = item.get_id()
            metadata = item.get_metadata().copy()
            content_obj = item.get_content()
            content_text = ""

            if isinstance(content_obj, str):
                content_text = content_obj
            elif hasattr(content_obj, "extract_text") and callable(
                getattr(content_obj, "extract_text")
            ):
                content_text = content_obj.extract_text()
                if not isinstance(content_text, str):
                    content_text = str(content_obj)
            else:
                content_text = str(content_obj)

            # Try to add content hash to metadata
            try:
                content_hash = item.get_content_hash()
                if content_hash:
                    metadata["content_hash"] = content_hash
            except (AttributeError, NotImplementedError):
                pass
            except Exception as e:
                logger.warning(f"Error getting content_hash for item ID '{doc_id}': {e}")

            texts_to_embed.append(content_text)
            items_info.append({"id": doc_id, "metadata": metadata, "text": content_text})

        if not texts_to_embed:
            logger.warning("No text content to embed. Skipping.")
            return

        logger.info(
            f"Embedding {len(texts_to_embed)} documents using '{self._embedding_model_name}'..."
        )
        generated_embeddings = self.embedding_model.encode(
            texts_to_embed, device=embedder_device, show_progress_bar=len(texts_to_embed) > 10
        )

        for i, item_info in enumerate(items_info):
            self._vectors.append(generated_embeddings[i])
            self._documents.append(item_info["text"])
            self._metadata.append(item_info["metadata"])
            self._ids.append(item_info["id"])

        logger.info(
            f"Successfully indexed {len(texts_to_embed)} documents. Total count: {len(self._vectors)}"
        )

    def search(
        self,
        query: Any,
        options: BaseSearchOptions,
    ) -> List[Dict[str, Any]]:
        if not self._vectors:
            logger.debug("No vectors in index. Returning empty results.")
            return []

        # Process query to text
        query_text = ""
        if isinstance(query, (str, Path)):
            query_text = str(query)
        elif hasattr(query, "extract_text") and callable(getattr(query, "extract_text")):
            query_text = query.extract_text()
            if not query_text or not query_text.strip():
                return []
        else:
            raise TypeError(f"Unsupported query type: {type(query)}")

        logger.info(
            f"Search request for collection '{self.collection_name}' with query type {type(query).__name__}"
        )

        # Encode query and perform similarity search
        query_vector = self.embedding_model.encode(query_text)

        # Convert list to numpy array for batch operations
        vectors_array = np.array(self._vectors)

        # Normalize vectors for cosine similarity
        query_norm = np.linalg.norm(query_vector)
        if query_norm > 0:
            query_vector = query_vector / query_norm

        # Normalize all vectors (avoid division by zero)
        vector_norms = np.linalg.norm(vectors_array, axis=1, keepdims=True)
        valid_indices = vector_norms.flatten() > 0
        vectors_array[valid_indices] = vectors_array[valid_indices] / vector_norms[valid_indices]

        # Calculate cosine similarities
        similarities = np.dot(vectors_array, query_vector)

        # Apply filters if present
        filtered_indices = np.arange(len(similarities))
        if options.filters:
            # Simple filtering for metadata fields
            # This is a basic implementation and doesn't support complex filters like LanceDB
            if isinstance(options.filters, dict):
                for field, value in options.filters.items():
                    new_filtered = []
                    for i in filtered_indices:
                        metadata = self._metadata[i]
                        if field in metadata and metadata[field] == value:
                            new_filtered.append(i)
                    filtered_indices = np.array(new_filtered)
            else:
                logger.warning(
                    f"Complex filter expressions not supported in NumPy backend: {options.filters}"
                )

        # Apply filtering and sort by similarity
        if len(filtered_indices) > 0:
            filtered_similarities = similarities[filtered_indices]
            top_k = min(options.top_k, len(filtered_similarities))
            if top_k == 0:
                return []

            top_indices_within_filtered = np.argsort(filtered_similarities)[-top_k:][::-1]
            top_indices = filtered_indices[top_indices_within_filtered]
        else:
            top_k = min(options.top_k, len(similarities))
            if top_k == 0:
                return []

            top_indices = np.argsort(similarities)[-top_k:][::-1]

        # Format results
        results = []
        for idx in top_indices:
            metadata = self._metadata[idx]
            results.append(
                {
                    "id": self._ids[idx],
                    "content_snippet": self._documents[idx][:200] if self._documents[idx] else "",
                    "score": float(similarities[idx]),
                    "page_number": metadata.get("page_number"),
                    "pdf_path": metadata.get("pdf_path"),
                    "metadata": metadata,
                }
            )

        logger.info(
            f"Search returned {len(results)} results from collection '{self.collection_name}'"
        )
        return results

    def index_exists(self) -> bool:
        return len(self._vectors) > 0

    def delete_index(self) -> bool:
        logger.warning(f"Deleting in-memory index for collection '{self.collection_name}'")
        self._vectors = []
        self._documents = []
        self._metadata = []
        self._ids = []
        return True

    def list_documents(self, include_metadata: bool = False, **kwargs) -> List[Dict]:
        logger.debug(
            f"Listing documents for NumPy collection '{self.collection_name}' (include_metadata={include_metadata})..."
        )

        results = []
        for i, doc_id in enumerate(self._ids):
            doc_info = {"id": doc_id}
            if include_metadata:
                doc_info["meta"] = self._metadata[i]
            results.append(doc_info)

        logger.info(
            f"Retrieved {len(results)} documents from NumPy collection '{self.collection_name}'"
        )
        return results

    def delete_documents(self, ids: List[str]) -> None:
        if not ids:
            logger.debug("No document IDs provided for deletion. Skipping.")
            return

        logger.warning(
            f"Request to delete {len(ids)} documents from NumPy collection '{self.collection_name}'"
        )

        # Find indices to remove
        keep_indices = []
        for i, doc_id in enumerate(self._ids):
            if doc_id not in ids:
                keep_indices.append(i)

        # Create new filtered lists
        self._ids = [self._ids[i] for i in keep_indices]
        self._vectors = [self._vectors[i] for i in keep_indices]
        self._documents = [self._documents[i] for i in keep_indices]
        self._metadata = [self._metadata[i] for i in keep_indices]

        logger.info(f"Deleted documents. Collection now contains {len(self._ids)} documents.")
