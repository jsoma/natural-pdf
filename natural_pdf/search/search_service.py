"""Lightweight semantic search over PDF pages using sentence-transformers."""

import logging
from typing import Any, Dict, List, Tuple

import numpy as np

from natural_pdf.exceptions import SearchError
from natural_pdf.utils.optional_imports import require

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "all-MiniLM-L6-v2"


class SearchService:
    """Stateless semantic search using sentence-transformers embeddings.

    The model is loaded lazily on first use and cached at the class level
    (singleton per model name) so it's shared across all search calls.
    """

    _models: Dict[str, Any] = {}

    @classmethod
    def get_model(cls, model_name: str = DEFAULT_MODEL):
        """Get or create a cached SentenceTransformer model."""
        if model_name not in cls._models:
            logger.info(f"Loading embedding model '{model_name}'...")
            st = require("sentence_transformers")
            cls._models[model_name] = st.SentenceTransformer(model_name)
        return cls._models[model_name]

    @staticmethod
    def page_texts(pages) -> List[str]:
        """Extract the text used for embedding, one string per page."""
        return [page.extract_text() or "" for page in pages]

    @staticmethod
    def text_fingerprint(texts: List[str]) -> str:
        """Stable digest of page texts, used to invalidate embedding caches.

        Each text is length-prefixed before hashing so the encoding is
        injective — separator-based concatenation would collide when texts
        contain the separator at their boundaries.
        """
        import hashlib

        digest = hashlib.sha1()
        for text in texts:
            encoded = text.encode("utf-8", "replace")
            digest.update(f"{len(encoded)}:".encode("ascii"))
            digest.update(encoded)
        return digest.hexdigest()

    @staticmethod
    def encode_texts(texts: List[str], model_name: str = DEFAULT_MODEL) -> np.ndarray:
        """Encode texts into a normalized embedding matrix."""
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        model = SearchService.get_model(model_name)
        payload = model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 20,
        )
        embeddings = np.asarray(payload, dtype=np.float32)
        SearchService._validate_embedding_matrix(
            embeddings, expected_rows=len(texts), source=f"embedding model {model_name!r}"
        )
        return embeddings

    @staticmethod
    def _validate_embedding_matrix(
        embeddings: np.ndarray,
        *,
        expected_rows: int,
        source: str,
    ) -> None:
        """Fail closed on malformed provider embeddings before ranking."""

        if embeddings.ndim != 2:
            raise SearchError(
                f"{source} returned a {embeddings.ndim}-D embedding payload; expected a matrix"
            )
        if embeddings.shape[0] != expected_rows:
            raise SearchError(
                f"{source} returned {embeddings.shape[0]} embeddings for " f"{expected_rows} inputs"
            )
        if embeddings.shape[1] == 0:
            raise SearchError(f"{source} returned embeddings with zero dimensions")

    @staticmethod
    def encode_pages(pages, model_name: str = DEFAULT_MODEL) -> np.ndarray:
        """Encode page texts into an embedding matrix.

        Args:
            pages: Iterable of Page objects with extract_text() method.
            model_name: Sentence-transformers model to use.

        Returns:
            Normalized embedding matrix of shape (num_pages, embedding_dim).
        """
        return SearchService.encode_texts(SearchService.page_texts(pages), model_name=model_name)

    @staticmethod
    def validate_query(query: str, top_k: int) -> None:
        """Validate search arguments; raises ValueError on bad input.

        Called by :meth:`rank`, and also by search entry points before any
        early return so an empty collection rejects the same bad arguments
        a populated one does.
        """
        if not query or query.isspace():
            raise ValueError("Search query cannot be empty.")
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")

    @staticmethod
    def rank(
        query: str,
        page_embeddings: np.ndarray,
        pages: list,
        top_k: int = 5,
        model_name: str = DEFAULT_MODEL,
    ) -> List[Tuple[Any, float]]:
        """Rank pages by semantic similarity to a query.

        Args:
            query: Search query string.
            page_embeddings: Pre-computed normalized embedding matrix.
            pages: List of Page objects (same order as embeddings).
            top_k: Number of results to return.
            model_name: Model to use for encoding the query.

        Returns:
            List of (page, score) tuples sorted by descending relevance.
        """
        SearchService.validate_query(query, top_k)

        page_matrix = np.asarray(page_embeddings, dtype=np.float32)

        # Check dimensionality before asking for a row count.  ``len()`` on a
        # scalar ndarray raises a raw TypeError, and a 1-D empty array is not a
        # valid embedding matrix even though its length is zero.
        if page_matrix.ndim != 2:
            raise SearchError(
                f"Search page cache returned a {page_matrix.ndim}-D embedding payload; "
                "expected a matrix"
            )

        if page_matrix.shape[0] == 0:
            if pages:
                raise SearchError(f"Search received 0 embeddings for {len(pages)} pages")
            return []

        SearchService._validate_embedding_matrix(
            page_matrix,
            expected_rows=len(pages),
            source="Search page cache",
        )

        model = SearchService.get_model(model_name)
        query_emb = model.encode(query, normalize_embeddings=True)
        query_vec = np.asarray(query_emb, dtype=np.float32)
        if query_vec.ndim != 1 or query_vec.shape[0] != page_matrix.shape[1]:
            raise SearchError(
                f"Embedding model {model_name!r} returned query shape "
                f"{query_vec.shape}; expected ({page_matrix.shape[1]},)"
            )
        # Cosine similarity (embeddings are already normalized)
        scores = page_matrix @ query_vec

        k = min(top_k, len(scores))
        top_idx = np.argsort(scores)[-k:][::-1]

        return [(pages[i], float(scores[i])) for i in top_idx]
