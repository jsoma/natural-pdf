import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple, Union, Literal

# Use object placeholders for external types to avoid direct dependency
BaseRanker = object
BaseEmbedder = object

logger = logging.getLogger(__name__)

# --- Base Search Options ---
@dataclass
class BaseSearchOptions:
    """Base options for search operations."""
    # How many results to return finally (after retrieval and optional reranking)
    top_k: int = 10
    # How many candidates the retriever should fetch initially (relevant if reranking)
    # Defaults to a value based on top_k if reranking, otherwise just top_k
    retriever_top_k: Optional[int] = None
    # Filters applied during retrieval (Haystack standard dictionary format)
    filters: Optional[Dict[str, Any]] = None

    # --- Reranking Configuration ---
    # Option 1: Simple boolean/None
    use_reranker: Optional[bool] = True # True=use default Cohere, False/None=disable
    # Option 2: Provide a specific instance (takes precedence over use_reranker boolean)
    reranker_instance: Optional[BaseRanker] = None
    # Parameters for default Cohere reranker (if use_reranker=True)
    reranker_model: Optional[str] = None # Defaults to "rerank-english-v2.0" in util
    reranker_api_key: Optional[str] = None # Defaults to COHERE_API_KEY env var

    # --- Embedder Configuration (Less common to override per-query, usually set at indexing) ---
    # embedder_instance: Optional[BaseEmbedder] = None # Might be useful for advanced cases

    def __post_init__(self):
        # Validate that top_k values make sense
        if self.retriever_top_k is None:
             # If retriever_top_k isn't set, default it based on reranking needs
             if self.use_reranker:
                 self.retriever_top_k = max(self.top_k * 2, 20) # Fetch more if reranking
             else:
                 self.retriever_top_k = self.top_k
        elif self.retriever_top_k < self.top_k:
            logger.warning(f"retriever_top_k ({self.retriever_top_k}) is less than top_k ({self.top_k}). Retriever should fetch at least as many candidates as the final desired results.")

# --- Text Search Specific Options ---
@dataclass
class TextSearchOptions(BaseSearchOptions):
    """Options specific to text-based semantic search."""
    # Add any text-specific overrides or parameters here if needed in the future
    # e.g., specifying default text reranker model name if different defaults emerge
    # default_text_reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    pass # Currently inherits all base options

# --- MultiModal Search Specific Options ---
@dataclass
class MultiModalSearchOptions(BaseSearchOptions):
    """Options specific to multimodal semantic search."""
    # Flag to potentially use a default multimodal reranker if available
    # (overrides base use_reranker=True if reranker_instance is None)
    use_multimodal_reranker: bool = True # Attempt multimodal rerank if use_reranker=True/None and no instance given
    # e.g., specifying default multimodal embedder/reranker models
    # default_multimodal_embedder_model: str = "sentence-transformers/clip-ViT-B-32-multilingual-v1"
    # default_multimodal_reranker_model: str = "jinaai/jina-reranker-m0" # Example

# --- Union Type ---
# Defines the types allowed for search configuration.
SearchOptions = Union[
    TextSearchOptions,
    MultiModalSearchOptions,
    BaseSearchOptions # Include base for typing flexibility
] 