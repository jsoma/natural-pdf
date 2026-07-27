"""Lightweight semantic search over PDF pages using sentence-transformers."""

import logging
from typing import Any, Dict, List, Tuple

import numpy as np

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
        """Stable digest of page texts, used to invalidate embedding caches."""
        import hashlib

        digest = hashlib.sha1()
        for text in texts:
            digest.update(text.encode("utf-8", "replace"))
            digest.update(b"\x00")
        return digest.hexdigest()

    @staticmethod
    def encode_texts(texts: List[str], model_name: str = DEFAULT_MODEL) -> np.ndarray:
        """Encode texts into a normalized embedding matrix."""
        model = SearchService.get_model(model_name)
        embeddings = model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 20,
        )
        return np.asarray(embeddings, dtype=np.float32)

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
        if not query or query.isspace():
            raise ValueError("Search query cannot be empty.")
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")

        if len(page_embeddings) == 0:
            return []

        model = SearchService.get_model(model_name)
        query_emb = model.encode(query, normalize_embeddings=True)
        query_vec = np.asarray(query_emb, dtype=np.float32)

        # Cosine similarity (embeddings are already normalized)
        scores = page_embeddings @ query_vec

        k = min(top_k, len(scores))
        top_idx = np.argsort(scores)[-k:][::-1]

        return [(pages[i], float(scores[i])) for i in top_idx]
