import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

import lancedb
import pyarrow as pa

from .search_options import BaseSearchOptions
from .search_service_protocol import Indexable, IndexConfigurationError, SearchServiceProtocol

# Lazy import for SentenceTransformer to avoid heavy loading at module level
# from sentence_transformers import SentenceTransformer


logger = logging.getLogger(__name__)

DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_LANCEDB_PERSIST_PATH = "./lancedb_data"


def _get_sentence_transformer(model_name: str):
    """Lazy import and instantiation of SentenceTransformer."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


class LanceDBSearchService(SearchServiceProtocol):
    """LanceDB-based implementation of the search service protocol."""

    collection_name: str

    def __init__(
        self,
        collection_name: str,
        persist: bool = False,
        uri: Optional[str] = None,
        embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
    ):
        self.collection_name = collection_name
        self._persist = persist
        self._uri = uri
        self._embedding_model_name = embedding_model_name
        self._embedding_dims: Optional[int] = None
        self._db = None
        self._table = None

        self.embedding_model = _get_sentence_transformer(self._embedding_model_name)
        test_embedding = self.embedding_model.encode("test")
        self._embedding_dims = len(test_embedding)

        if self._persist:
            self._uri = self._uri if self._uri else DEFAULT_LANCEDB_PERSIST_PATH
            logger.info(f"Initializing Persistent LanceDB client at path: {self._uri}")
            Path(self._uri).mkdir(parents=True, exist_ok=True)
        else:
            self._temp_dir_obj = tempfile.TemporaryDirectory()
            self._uri = self._temp_dir_obj.name
            logger.info(f"Initializing In-Memory LanceDB client using temp path: {self._uri}")

        self._db = lancedb.connect(self._uri)
        self._open_or_create_table()
        logger.info(
            f"LanceDBSearchService initialized. Table '{self.collection_name}' (persist={self._persist} at '{self._uri}'). Model: '{self._embedding_model_name}', Dims: {self._embedding_dims}"
        )

    def _get_schema(self) -> pa.Schema:
        if self._embedding_dims is None:
            raise RuntimeError("Embedding dimensions not determined. Cannot create schema.")

        return pa.schema(
            [
                pa.field("id", pa.string(), nullable=False),
                pa.field("vector", pa.list_(pa.float32(), list_size=self._embedding_dims)),
                pa.field("text", pa.string()),
                pa.field("metadata_json", pa.string()),
            ]
        )

    def _open_or_create_table(self):
        if self._db is None:
            raise RuntimeError("LanceDB connection not established.")

        table_names = self._db.table_names()

        if self.collection_name in table_names:
            logger.debug(f"Opening existing LanceDB table: {self.collection_name}")
            self._table = self._db.open_table(self.collection_name)
        else:
            logger.debug(f"Creating new LanceDB table: {self.collection_name} with schema.")
            schema = self._get_schema()
            self._table = self._db.create_table(self.collection_name, schema=schema, mode="create")

    def __del__(self):
        if not self._persist and hasattr(self, "_temp_dir_obj") and logger:
            logger.debug(f"Cleaning up temporary directory for in-memory LanceDB: {self._uri}")
            self._temp_dir_obj.cleanup()

    def index(
        self,
        documents: Iterable[Indexable],
        embedder_device: Optional[str] = None,
        force_reindex: bool = False,
    ) -> None:
        indexable_list = list(documents)
        logger.info(
            f"Index request for table='{self.collection_name}', docs={len(indexable_list)}, model='{self._embedding_model_name}', force={force_reindex}"
        )

        if self._table is None or self._db is None:
            raise RuntimeError(f"LanceDB table '{self.collection_name}' not initialized.")

        if not indexable_list:
            logger.warning("No documents provided for indexing. Skipping.")
            return

        if force_reindex:
            logger.warning(
                f"Force reindex requested for table '{self.collection_name}'. Deleting existing table and recreating."
            )
            self._db.drop_table(self.collection_name)
            self._open_or_create_table()
            logger.info(f"Table '{self.collection_name}' deleted and recreated.")

        data_to_add = []
        texts_to_embed: List[str] = []
        original_items_info: List[Dict[str, Any]] = []

        import json

        for item in indexable_list:
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

            try:
                content_hash = item.get_content_hash()
                if content_hash:
                    metadata["content_hash"] = content_hash
            except (AttributeError, NotImplementedError):
                pass
            except Exception as e:
                logger.warning(f"Error getting content_hash for item ID '{doc_id}': {e}")

            # Ensure doc_id is not None - use a fallback if needed
            if doc_id is None:
                # Generate a unique ID based on content hash or position in the list
                try:
                    doc_id = f"auto_{item.get_content_hash() if hasattr(item, 'get_content_hash') else hash(content_text)}"
                except:
                    doc_id = f"auto_{len(texts_to_embed)}"

            texts_to_embed.append(content_text)
            original_items_info.append(
                {"id": doc_id, "metadata_json": json.dumps(metadata), "text": content_text}
            )

        if not texts_to_embed:
            logger.warning("No text content to embed. Skipping.")
            return

        logger.info(
            f"Embedding {len(texts_to_embed)} documents using '{self._embedding_model_name}'..."
        )
        generated_embeddings = self.embedding_model.encode(
            texts_to_embed, device=embedder_device, show_progress_bar=len(texts_to_embed) > 10
        )

        for i, item_info in enumerate(original_items_info):
            data_to_add.append(
                {
                    "id": item_info["id"],
                    "vector": generated_embeddings[i].tolist(),
                    "text": item_info["text"],
                    "metadata_json": item_info["metadata_json"],
                }
            )

        if not data_to_add:
            logger.warning("No data prepared for LanceDB. Skipping add.")
            return

        # Create a PyArrow table with the same schema as the LanceDB table
        schema = self._get_schema()
        arrays = [
            pa.array([item["id"] for item in data_to_add], type=pa.string()),
            pa.array([item["vector"] for item in data_to_add]),
            pa.array([item["text"] for item in data_to_add], type=pa.string()),
            pa.array([item["metadata_json"] for item in data_to_add], type=pa.string()),
        ]
        table = pa.Table.from_arrays(arrays, schema=schema)

        logger.info(
            f"Adding/updating {len(data_to_add)} documents to LanceDB table '{self.collection_name}'."
        )
        self._table.merge_insert(
            "id"
        ).when_matched_update_all().when_not_matched_insert_all().execute(
            table,
        )
        logger.info(
            f"Successfully added/updated {len(data_to_add)} documents. Table count: {self._table.count_rows()}"
        )

    def search(
        self,
        query: Any,
        options: BaseSearchOptions,
    ) -> List[Dict[str, Any]]:
        if self._table is None:
            raise RuntimeError(f"LanceDB table '{self.collection_name}' not initialized.")

        logger.info(
            f"Search request for table='{self.collection_name}', query_type={type(query).__name__}, options={options}"
        )
        query_text = ""
        if isinstance(query, (str, Path)):
            query_text = str(query)
        elif hasattr(query, "extract_text") and callable(getattr(query, "extract_text")):
            query_text = query.extract_text()
            if not query_text or not query_text.strip():
                return []
        else:
            raise TypeError(f"Unsupported query type: {type(query)}")

        query_vector = self.embedding_model.encode(query_text).tolist()

        lancedb_filter = None
        if options.filters:
            if isinstance(options.filters, str):
                lancedb_filter = options.filters
            elif isinstance(options.filters, dict):
                filter_parts = []
                for k, v in options.filters.items():
                    if isinstance(v, str):
                        filter_parts.append(f"{k} = '{v}'")
                    else:
                        filter_parts.append(f"{k} = {v}")
                if filter_parts:
                    lancedb_filter = " AND ".join(filter_parts)
                logger.warning(
                    f"Filter conversion from dict is basic: {options.filters} -> {lancedb_filter}. For metadata_json, use SQL path expressions."
                )

        search_query = self._table.search(query_vector).limit(options.top_k)
        if lancedb_filter:
            search_query = search_query.where(lancedb_filter)

        results_df = search_query.to_df()
        final_results: List[Dict[str, Any]] = []
        import json

        for _, row in results_df.iterrows():
            metadata = {}
            if "metadata_json" in row and row["metadata_json"]:
                try:
                    metadata = json.loads(row["metadata_json"])
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse metadata_json for id {row.get('id')}")

            score = 1 - row["_distance"] if "_distance" in row else 0.0

            final_results.append(
                {
                    "id": row.get("id"),
                    "content_snippet": row["text"][:200] if "text" in row and row["text"] else "",
                    "score": score,
                    "page_number": metadata.get("page_number"),
                    "pdf_path": metadata.get("pdf_path"),
                    "metadata": metadata,
                }
            )
        logger.info(
            f"Search returned {len(final_results)} results from LanceDB table '{self.collection_name}'."
        )
        return final_results

    def delete_index(self) -> bool:
        if self._db is None:
            logger.warning("LanceDB connection not initialized. Cannot delete index.")
            return False
        logger.warning(f"Request to delete LanceDB table '{self.collection_name}'.")

        self._db.drop_table(self.collection_name)
        self._table = None
        logger.info(f"LanceDB table '{self.collection_name}' deleted successfully.")
        return True

    def index_exists(self) -> bool:
        if self._db is None:
            return False
        exists = self.collection_name in self._db.table_names()
        if exists:
            tbl = self._db.open_table(self.collection_name)
            count = tbl.count_rows()
            logger.debug(
                f"LanceDB table '{self.collection_name}' found with {count} documents. Exists: {count > 0}"
            )
            return count > 0

        logger.debug(f"LanceDB table '{self.collection_name}' not found in db.table_names().")
        return False

    def list_documents(self, include_metadata: bool = False, **kwargs) -> List[Dict]:
        if self._table is None:
            raise RuntimeError("Table not initialized")
        logger.debug(
            f"Listing documents for LanceDB table '{self.collection_name}' (include_metadata={include_metadata})..."
        )

        select_columns = ["id"]
        if include_metadata:
            select_columns.append("metadata_json")

        lancedb_filter = kwargs.get("filters")

        query = self._table.to_lance().scanner(columns=select_columns, filter=lancedb_filter)
        results_table = query.to_table()
        results_list = results_table.to_pylist()

        formatted_docs: List[Dict[str, Any]] = []
        import json

        for row in results_list:
            doc_data: Dict[str, Any] = {"id": row.get("id")}
            if include_metadata and "metadata_json" in row and row["metadata_json"]:
                try:
                    metadata = json.loads(row["metadata_json"])
                    doc_data["meta"] = metadata
                except json.JSONDecodeError:
                    doc_data["meta"] = {}
            formatted_docs.append(doc_data)
        logger.info(
            f"Retrieved {len(formatted_docs)} documents from LanceDB table '{self.collection_name}'."
        )
        return formatted_docs

    def delete_documents(self, ids: List[str]) -> None:
        if self._table is None:
            raise RuntimeError("Table not initialized")
        if not ids:
            logger.debug("No document IDs provided for deletion. Skipping.")
            return

        id_filter_string = ", ".join([f"'{doc_id}'" for doc_id in ids])
        delete_condition = f"id IN ({id_filter_string})"
        logger.warning(
            f"Request to delete {len(ids)} documents from LanceDB table '{self.collection_name}' with condition: {delete_condition}"
        )

        self._table.delete(delete_condition)
        logger.info(
            f"Successfully requested deletion of {len(ids)} documents. Table count now: {self._table.count_rows()}"
        )
