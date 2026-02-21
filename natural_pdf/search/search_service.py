"""Lightweight semantic search over PDF pages using sentence-transformers."""

import logging
from typing import Any, Dict, List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "all-MiniLM-L6-v2"


class SearchService:
    """Stateless semantic search using sentence-transformers embeddings.

    The model is loaded lazily on first use and cached at the class level
    (singleton per model name) so it's shared across all search calls.
    """

    _models: Dict[str, Any] = {}

    @classmethod
    def get_model(cls, model_name: str = DEFAULT_MODEL) -> SentenceTransformer:
        """Get or create a cached SentenceTransformer model."""
        if model_name not in cls._models:
            logger.info(f"Loading embedding model '{model_name}'...")
            cls._models[model_name] = SentenceTransformer(model_name)
        return cls._models[model_name]

    @staticmethod
    def encode_pages(pages, model_name: str = DEFAULT_MODEL) -> np.ndarray:
        """Encode page texts into an embedding matrix.

        Args:
            pages: Iterable of Page objects with extract_text() method.
            model_name: Sentence-transformers model to use.

        Returns:
            Normalized embedding matrix of shape (num_pages, embedding_dim).
        """
        texts = [page.extract_text() or "" for page in pages]
        model = SearchService.get_model(model_name)
        embeddings = model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 20,
        )
        return np.asarray(embeddings, dtype=np.float32)

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
