# natural_pdf/search/haystack_utils.py
import logging
import os
from typing import Optional, Dict, Any, List, Union, Tuple, Type
from pathlib import Path
from PIL import Image # Ensure Image is imported unconditionally
import warnings

from natural_pdf.search.search_options import SearchOptions, TextSearchOptions, MultiModalSearchOptions, BaseSearchOptions

# Set up logger for this module
logger = logging.getLogger(__name__)

# --- Define flag BEFORE trying Haystack imports ---
HAS_HAYSTACK_EXTRAS = False # Default to False

# --- Conditional Haystack Imports (Restoring Error Catching with Traceback Logging) ---
try:
    import haystack
    from haystack import Document as HaystackDocument, Pipeline
    from haystack_integrations.document_stores.chroma import ChromaDocumentStore
    from haystack.document_stores.types import DuplicatePolicy, DocumentStore
    from haystack.components.embedders import (
        SentenceTransformersTextEmbedder,
        SentenceTransformersDocumentEmbedder
    )
    from haystack_integrations.components.retrievers.chroma import ChromaEmbeddingRetriever
    # Keep try/except for optional Cohere
    try:
        from haystack.components.rankers import CohereRanker
    except ImportError:
        CohereRanker = None

    # --- Add ChromaDB embedding function import ---
    try:
         from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
    except ImportError:
         logger.warning("chromadb library not found. Custom embedding models for ChromaDocumentStore may not work.")
         SentenceTransformerEmbeddingFunction = None
    # --- End ChromaDB import ---

    HAS_HAYSTACK_EXTRAS = True # Set to True if imports succeed
    logger.debug("Successfully imported Haystack components.")

except ImportError as e:
    # HAS_HAYSTACK_EXTRAS remains False
    # Log the full error and traceback for debugging
    logger.error(f"Failed to import Haystack components. Search functionality disabled. Error: {e}", exc_info=True)

    # Define dummy types/classes for type hinting and basic checks when extras aren't installed
    BaseDocumentStore = object
    DocumentStore = object # Dummy for protocol
    BaseEmbedder = object # Define dummy BaseEmbedder
    BaseTextEmbedder = object
    HaystackDocument = Dict # Represent as Dict if not available
    Pipeline = None
    SentenceTransformersTextEmbedder = None
    ChromaEmbeddingRetriever = None # Dummy for Embedding Retriever
    CohereRanker = None
    ChromaDocumentStore = None
    DuplicatePolicy = None # Dummy for DuplicatePolicy
    SentenceTransformerEmbeddingFunction = None # Dummy if kept


# Helper function to check availability and raise error
def check_haystack_availability(feature_name: str = "Search"):
    """Raises ImportError if Haystack extras are not installed."""
    if not HAS_HAYSTACK_EXTRAS:
        raise ImportError(
            f"'{feature_name}' requires Haystack extras. "
            "Please install them using: pip install natural-pdf[haystack]"
        )


# ===========================
# Default Component Creators
# ===========================

def create_default_document_store(
    persist_path: str = "./natural_pdf_index",
    collection_name: str = "natural_pdf_default",
    embedding_model: Optional[str] = None # Allow specifying the model
) -> DocumentStore:
    """Creates a default ChromaDB DocumentStore."""
    check_haystack_availability("create_default_document_store")
    logger.debug(f"Creating default ChromaDocumentStore at '{persist_path}' with collection '{collection_name}'")
    if not ChromaDocumentStore: # Should be caught by check_haystack_availability, but double-check
         raise RuntimeError("ChromaDocumentStore is not available despite Haystack extras check.")

    try:
        # Note: For Haystack's Chroma integration, the embedding model is typically handled
        # by the Embedder component in the indexing/query pipeline, not set directly
        # on the DocumentStore initialization.
        # The `embedding_model` parameter passed here might be used later to configure that Embedder.
        store = ChromaDocumentStore(
            persist_path=persist_path,
            collection_name=collection_name,
            # embedding_function parameter removed as it caused issues with Haystack's util
        )
        logger.info(f"Initialized ChromaDocumentStore (Collection: {collection_name})")
        return store
    except Exception as e:
        logger.error(f"Failed to initialize ChromaDocumentStore: {e}", exc_info=True)
        raise RuntimeError(f"Could not create ChromaDocumentStore for collection '{collection_name}'") from e

def create_default_text_embedder(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    device: Optional[str] = None # Add device parameter
) -> SentenceTransformersTextEmbedder:
    """Creates a default SentenceTransformer text embedder."""
    check_haystack_availability("create_default_text_embedder")
    logger.debug(f"Creating default SentenceTransformersTextEmbedder with model '{model_name}'")
    if not SentenceTransformersTextEmbedder:
        raise RuntimeError("SentenceTransformersTextEmbedder not available.")
    try:
        # Use Haystack component which handles device logic
        embedder = SentenceTransformersTextEmbedder(model=model_name, device=device)
        logger.info(f"Initialized SentenceTransformersTextEmbedder (Model: {model_name}, Device: {embedder.device})")
        return embedder
    except Exception as e:
        logger.error(f"Failed to initialize SentenceTransformersTextEmbedder: {e}", exc_info=True)
        raise RuntimeError(f"Could not create SentenceTransformersTextEmbedder with model '{model_name}'") from e

def create_default_multimodal_embedder(*args, **kwargs) -> Any:
    """Stub for creating a default multimodal embedder (Not Implemented)."""
    logger.error("Default multimodal embedder creation is not yet implemented.")
    raise NotImplementedError(
        "Creating a default multimodal embedder requires a custom component or integration not yet implemented."
        " See: https://docs.haystack.deepset.ai/docs/custom-components"
    )

def create_default_text_reranker(
        api_key: Optional[str] = None,
        model_name: str = "rerank-english-v2.0" # Default Cohere model
    ) -> Optional[Any]: # Returns CohereRanker instance or None
     """
     Creates a default Cohere Reranker if available and API key provided.

     Requires COHERE_API_KEY environment variable or api_key argument.
     Requires haystack-cohere integration: pip install haystack-cohere
     """
     check_haystack_availability("create_default_text_reranker (optional)")

     if not CohereRanker:
          logger.debug("CohereRanker component not available (haystack-cohere likely not installed). Skipping reranker creation.")
          return None

     # Check for API key (prefer argument over environment variable)
     cohere_api_key = api_key or os.environ.get("COHERE_API_KEY")
     if not cohere_api_key:
         logger.warning("COHERE_API_KEY not found in arguments or environment variables. Cannot create Cohere Reranker.")
         return None

     logger.debug(f"Creating CohereRanker with model '{model_name}'")
     try:
         # Pass API key via authenticator for better practice if supported, or directly
         # As of haystack 2.0b5, CohereRanker takes api_key directly
         reranker = CohereRanker(api_key=cohere_api_key, model=model_name)
         logger.info(f"Initialized CohereRanker (Model: {model_name})")
         return reranker
     except Exception as e:
         logger.error(f"Failed to initialize CohereRanker: {e}", exc_info=True)
         # Don't raise, just return None as reranker is optional
         return None

# --- Default Document Embedder Creator ---
def create_default_document_embedder(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    device: Optional[str] = None,
    progress_bar: bool = True,
    normalize_embeddings: bool = False # Changed default based on ST documentation
) -> Any: # Return Any as actual type depends on availability
    """Creates a default SentenceTransformersDocumentEmbedder instance.

    Args:
        model_name: The Sentence Transformers model name or path.
        device: The device (e.g., 'cpu', 'cuda') to use.
        progress_bar: Show progress bar during embedding.
        normalize_embeddings: Normalize embeddings to unit length.

    Returns:
        A SentenceTransformersDocumentEmbedder instance or raises ImportError.

    Raises:
        ImportError: If SentenceTransformersDocumentEmbedder is not available.
        RuntimeError: If initialization fails.
    """
    check_haystack_availability("SentenceTransformersDocumentEmbedder")
    if not SentenceTransformersDocumentEmbedder:
        raise ImportError("SentenceTransformersDocumentEmbedder is required but not available.")

    # Use the provided device parameter directly.
    # If None, Haystack component will likely pick a default (e.g., 'cpu' or 'cuda' if available)
    resolved_device = device
    logger.debug(f"Attempting to create SentenceTransformersDocumentEmbedder. Model: {model_name}, Device: {resolved_device or 'auto'}")

    try:
        embedder = SentenceTransformersDocumentEmbedder(
            model=model_name,
            device=resolved_device,
            progress_bar=progress_bar,
            normalize_embeddings=normalize_embeddings,
            # meta_fields_to_embed=config.get('DOC_EMBEDDER_META_FIELDS', []) # Removed reliance on config
            # If embedding meta fields is needed, it should be passed as a parameter
        )
        embedder.warm_up()
        logger.info(f"Created SentenceTransformersDocumentEmbedder. Model: {model_name}, Device: {embedder.device}") # Use embedder.device after init
    except Exception as e:
        logger.error(f"Failed to initialize SentenceTransformersDocumentEmbedder: {e}", exc_info=True)
        raise RuntimeError(f"Failed to initialize SentenceTransformersDocumentEmbedder with model '{model_name}'.") from e

    return embedder


# ===========================
# Helper Functions (Removed _determine_query_embedding)
# ===========================


# ===========================
# Central Search Logic
# ===========================

def _perform_haystack_search(
    query: Union[str, Path, Image.Image],
    document_store: Any, # Use Any for simplicity now
    collection_name: str, # Passed for clarity, but Chroma store instance is collection-specific
    embedder: SentenceTransformersTextEmbedder, # Explicitly expect a text embedder for queries
    options: BaseSearchOptions
) -> List[Dict[str, Any]]:
    """Internal function to perform search using Haystack components (ChromaEmbeddingRetriever)."""
    if not HAS_HAYSTACK_EXTRAS:
        check_haystack_availability("_perform_haystack_search")
        return [] # Should not be reached due to check

    logger.info(f"Performing Haystack search in collection '{collection_name}' (using store: {type(document_store).__name__})...")
    logger.debug(f"  Query type: {type(query).__name__}")
    logger.debug(f"  Options: {options}")

    # --- 1. Embed Query (using the provided text embedder) --- #
    text_query: Optional[str] = None
    query_embedding: Optional[List[float]] = None

    if isinstance(query, str):
        text_query = query # Keep text for potential reranker use
        if not embedder:
            logger.error("Text query provided, but no embedder instance was passed to _perform_haystack_search.")
            return []
        # No need to check type if the type hint is enforced upstream
        # if not isinstance(embedder, SentenceTransformersTextEmbedder):
        #      logger.warning(f"Provided embedder is {type(embedder).__name__}, not SentenceTransformersTextEmbedder. Assuming it works like one for query embedding.")
        try:
            logger.debug(f"Running embedder {type(embedder).__name__} on query text...")
            embedding_result = embedder.run(text=text_query)
            query_embedding = embedding_result.get("embedding")
            if not query_embedding:
                 logger.error(f"Embedder {type(embedder).__name__} failed to return an embedding for the query: '{text_query[:100]}...'")
                 return []
            logger.debug(f"Generated query embedding (Dim: {len(query_embedding)}). Text kept for potential reranking.")
        except Exception as e:
            logger.error(f"Failed to run text embedder on query text: {e}", exc_info=True)
            return []
    elif isinstance(query, Path) or isinstance(query, Image.Image):
        # Currently, this function doesn't support multi-modal query embedding directly
        logger.error(f"Unsupported query type ({type(query).__name__}) for embedding in _perform_haystack_search. Requires text.")
        return []
    else:
        # Handle other unexpected types
        logger.error(f"Unsupported query type: {type(query).__name__}. Requires text.")

    # If we didn't get an embedding (e.g., non-text query), we can't proceed
    if query_embedding is None:
        logger.error("Could not obtain query embedding. Cannot perform search.")
        return []

    # --- 2. Set up Retriever --- #
    # Assumes the document_store is ChromaDocumentStore for this utility function context
    if not ChromaEmbeddingRetriever:
         logger.error("ChromaEmbeddingRetriever not available.")
         return []

    # Ensure retriever_top_k is set (should be by __post_init__)
    retriever_top_k = options.retriever_top_k
    if retriever_top_k is None:
         logger.warning("options.retriever_top_k was None, defaulting to options.top_k for retriever.")
         retriever_top_k = options.top_k

    # Instantiate the EMBEDDING retriever
    retriever = ChromaEmbeddingRetriever(document_store=document_store,
                                         filters=options.filters or {}, # Pass filters here
                                         top_k=retriever_top_k)

    logger.debug(f"Initialized ChromaEmbeddingRetriever (Top K: {retriever.top_k}, Filters: {retriever.filters})")


    # --- 3. Set up Optional Reranker --- #
    reranker_instance = None
    if options.use_reranker in [True, None]: # Check specifically for True or None
        logger.debug("Attempting to initialize reranker...")
        # Currently only supports default text reranker (Cohere)
        reranker_instance = create_default_text_reranker(api_key=options.reranker_api_key,
                                                         model_name=options.reranker_model or "rerank-english-v2.0")
        if reranker_instance:
             # Ensure reranker top_k matches final desired top_k
             reranker_instance.top_k = options.top_k # Set the final top_k for the reranker
             logger.info(f"Using reranker: {type(reranker_instance).__name__} (Final Top K: {options.top_k})")
        else:
            logger.warning("Reranker requested (use_reranker=True/None) but could not be initialized (check API key/installation). Proceeding without reranking.")


    # --- 4. Build and Run Pipeline --- #
    if not Pipeline:
         logger.error("Haystack Pipeline class not available.")
         return []

    search_pipeline = Pipeline()
    search_pipeline.add_component("retriever", retriever)

    # Define pipeline input based on EMBEDDING retriever needs
    pipeline_input = {"retriever": {"query_embedding": query_embedding}}
    last_component_name = "retriever"

    if reranker_instance:
        search_pipeline.add_component("reranker", reranker_instance)
        search_pipeline.connect("retriever.documents", "reranker.documents")
        # Reranker also needs the query text and final top_k
        if text_query is None:
            logger.error("Reranker requires text query, but it was not available (query might not have been text).")
            # Handle this case - maybe skip reranker or raise error?
            # For now, let's skip reranker if text is missing
            logger.warning("Skipping reranker because text query is missing.")
            reranker_instance = None # Effectively remove it from the logic below
            last_component_name = "retriever" # Reset last component
            # Remove reranker component if added? Less clean. Let's just not add its input.
        else:
            pipeline_input["reranker"] = {"query": text_query, "top_k": options.top_k} # Pass query and final top_k
            last_component_name = "reranker"
            logger.debug("Added reranker to pipeline and configured input.")
    else:
        # No reranker was initialized or it was skipped
        last_component_name = "reranker"
        logger.debug("Added reranker to pipeline.")


    logger.info("Running Haystack search pipeline...")
    try:
        result = search_pipeline.run(pipeline_input)
        logger.info("Haystack search pipeline finished.")

    except Exception as e:
        logger.error(f"Haystack search pipeline failed: {e}", exc_info=True)
        return []

    # --- 5. Process Results --- #
    final_documents: List[HaystackDocument] = []
    # Check output based on last component in the pipeline
    if last_component_name in result and result[last_component_name].get("documents"):
        final_documents = result[last_component_name]["documents"]
        logger.debug(f"Processed results from '{last_component_name}' ({len(final_documents)} documents).")
    else:
        logger.warning(f"Search pipeline component '{last_component_name}' returned no documents or unexpected output format. Result keys: {result.keys()}")
        return []

    # Convert Haystack Documents to the desired output format
    output_results = []
    for doc in final_documents: # Correctly loop over final_documents
         # Check if doc is actually a Haystack Document object or potentially a dict
         doc_id = getattr(doc, 'id', None)
         doc_score = getattr(doc, 'score', 0.0)
         doc_content = getattr(doc, 'content', None)
         doc_meta = getattr(doc, 'meta', {})

         meta = doc_meta or {}
         output = {
             "pdf_path": meta.get("pdf_path", "Unknown"),
             "page_number": meta.get("page_number", -1),
             "score": doc_score if doc_score is not None else 0.0, # Handle potential None score
             "content_snippet": doc_content[:200] + "..." if doc_content else "", # Add snippet
             "metadata": meta,
             # "haystack_document": doc # Optionally include the full Haystack doc
         }
         output_results.append(output)

    logger.info(f"Returning {len(output_results)} relevant results.")
    return output_results

