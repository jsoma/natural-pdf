"""Tests for the simplified semantic search system."""

import numpy as np
import pytest


class FakePage:
    """Minimal page stub for testing search without loading real PDFs."""

    def __init__(self, text, page_number=0):
        self._text = text
        self.page_number = page_number

    def extract_text(self):
        return self._text


class FakeModel:
    """Fake sentence-transformers model that returns deterministic embeddings."""

    def __init__(self, dim=4):
        self.dim = dim

    def encode(self, texts, normalize_embeddings=False, show_progress_bar=False):
        if isinstance(texts, str):
            texts = [texts]
            single = True
        else:
            single = False

        embeddings = []
        for text in texts:
            # Generate a simple deterministic embedding from the text hash
            rng = np.random.RandomState(hash(text) % 2**31)
            vec = rng.randn(self.dim).astype(np.float32)
            if normalize_embeddings:
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm
            embeddings.append(vec)

        result = np.array(embeddings)
        return result[0] if single else result


@pytest.fixture(autouse=True)
def mock_model(monkeypatch):
    """Replace the real SentenceTransformer with a fake for all tests."""
    from natural_pdf.search import search_service

    fake = FakeModel()
    monkeypatch.setattr(search_service.SearchService, "_models", {"all-MiniLM-L6-v2": fake})
    return fake


class TestSearchService:
    def test_encode_pages(self):
        from natural_pdf.search.search_service import SearchService

        pages = [FakePage("hello world"), FakePage("foo bar"), FakePage("baz qux")]
        embeddings = SearchService.encode_pages(pages)

        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (3, 4)

    def test_encode_empty_pages(self):
        from natural_pdf.search.search_service import SearchService

        embeddings = SearchService.encode_pages([])
        assert embeddings.shape[0] == 0

    def test_rank_returns_sorted(self):
        from natural_pdf.search.search_service import SearchService

        pages = [
            FakePage("financial report quarterly earnings"),
            FakePage("legal contract terms and conditions"),
            FakePage("invoice payment terms billing"),
        ]
        embeddings = SearchService.encode_pages(pages)

        results = SearchService.rank("payment invoice", embeddings, pages, top_k=3)

        assert len(results) == 3
        # Results should be (page, score) tuples
        for page, score in results:
            assert isinstance(page, FakePage)
            assert isinstance(score, float)

        # Scores should be in descending order
        scores = [score for _, score in results]
        assert scores == sorted(scores, reverse=True)

    def test_rank_top_k_limits(self):
        from natural_pdf.search.search_service import SearchService

        pages = [FakePage(f"page {i}") for i in range(10)]
        embeddings = SearchService.encode_pages(pages)

        results = SearchService.rank("query", embeddings, pages, top_k=3)
        assert len(results) == 3

    def test_rank_top_k_exceeds_pages(self):
        from natural_pdf.search.search_service import SearchService

        pages = [FakePage("only page")]
        embeddings = SearchService.encode_pages(pages)

        results = SearchService.rank("query", embeddings, pages, top_k=10)
        assert len(results) == 1

    def test_rank_empty(self):
        from natural_pdf.search.search_service import SearchService

        embeddings = np.array([]).reshape(0, 4)
        results = SearchService.rank("query", embeddings, [], top_k=5)
        assert results == []

    @pytest.mark.parametrize(
        "embeddings",
        [None, np.array(1.0), np.array([], dtype=np.float32)],
        ids=["none", "scalar", "one-dimensional-empty"],
    )
    def test_rank_rejects_non_matrix_page_embeddings(self, embeddings):
        from natural_pdf.exceptions import SearchError
        from natural_pdf.search.search_service import SearchService

        with pytest.raises(SearchError, match="expected a matrix"):
            SearchService.rank("query", embeddings, [])

    def test_get_model_caching(self):
        from natural_pdf.search.search_service import SearchService

        model1 = SearchService.get_model("all-MiniLM-L6-v2")
        model2 = SearchService.get_model("all-MiniLM-L6-v2")
        assert model1 is model2

    def test_encode_rejects_provider_result_count_mismatch(self, monkeypatch):
        from natural_pdf.exceptions import SearchError
        from natural_pdf.search.search_service import SearchService

        model = SearchService.get_model()
        monkeypatch.setattr(
            model,
            "encode",
            lambda *_args, **_kwargs: np.ones((1, 4), dtype=np.float32),
        )
        with pytest.raises(SearchError, match="1 embeddings for 2 inputs"):
            SearchService.encode_texts(["one", "two"])

    def test_rank_rejects_wrong_query_embedding_shape(self, monkeypatch):
        from natural_pdf.exceptions import SearchError
        from natural_pdf.search.search_service import SearchService

        model = SearchService.get_model()
        monkeypatch.setattr(
            model,
            "encode",
            lambda *_args, **_kwargs: np.ones((1, 4), dtype=np.float32),
        )
        with pytest.raises(SearchError, match="query shape"):
            SearchService.rank(
                "query",
                np.ones((1, 4), dtype=np.float32),
                [FakePage("one")],
            )

    def test_model_errors_propagate_unchanged(self, monkeypatch):
        from natural_pdf.search.search_service import SearchService

        model = SearchService.get_model()

        def fail(*_args, **_kwargs):
            raise RuntimeError("provider failed")

        monkeypatch.setattr(model, "encode", fail)
        with pytest.raises(RuntimeError, match="provider failed"):
            SearchService.encode_texts(["one"])
        with pytest.raises(RuntimeError, match="provider failed"):
            SearchService.rank(
                "query",
                np.ones((1, 4), dtype=np.float32),
                [FakePage("one")],
            )

    def test_rank_rejects_page_embedding_count_mismatch(self):
        from natural_pdf.exceptions import SearchError
        from natural_pdf.search.search_service import SearchService

        with pytest.raises(SearchError, match="1 embeddings for 2 inputs"):
            SearchService.rank(
                "query",
                np.ones((1, 4), dtype=np.float32),
                [FakePage("one"), FakePage("two")],
            )


class TestPDFSearch:
    def test_pdf_search_returns_page_collection(self):
        pdf = _make_pdf()
        results = pdf.search("test query")

        from natural_pdf.core.page_collection import PageCollection

        assert isinstance(results, PageCollection)

    def test_pdf_search_attaches_scores(self):
        pdf = _make_pdf()
        results = pdf.search("test query")

        for page in results:
            assert hasattr(page, "_search_score")
            assert isinstance(page._search_score, float)

    def test_pdf_search_top_k(self):
        pdf = _make_pdf()
        results = pdf.search("test query", top_k=1)
        assert len(results) <= 1

    def test_pdf_search_caches_embeddings(self):
        pdf = _make_pdf()

        # First search triggers encoding
        pdf.search("query one")
        assert hasattr(pdf, "_search_embeddings")
        assert "all-MiniLM-L6-v2" in pdf._search_embeddings

        # Store reference to cached embeddings
        cached = pdf._search_embeddings["all-MiniLM-L6-v2"]

        # Second search should reuse cache while the text is unchanged
        pdf.search("query two")
        assert pdf._search_embeddings["all-MiniLM-L6-v2"] is cached

    def test_pdf_search_cache_invalidated_when_text_changes(self, monkeypatch):
        """After a text mutation (e.g. apply_ocr), search must re-encode
        instead of ranking against stale embeddings."""
        pdf = _make_pdf()
        pdf.search("query one")
        cached = pdf._search_embeddings["all-MiniLM-L6-v2"]

        page = pdf.pages[0]
        monkeypatch.setattr(page, "extract_text", lambda *a, **kw: "completely new ocr text")

        pdf.search("query two")
        assert pdf._search_embeddings["all-MiniLM-L6-v2"] is not cached

    def test_rank_rejects_bad_inputs(self):
        from natural_pdf.search.search_service import SearchService

        embeddings = np.zeros((2, 4), dtype=np.float32)
        pages = [FakePage("a"), FakePage("b")]
        with pytest.raises(ValueError, match="top_k"):
            SearchService.rank("query", embeddings, pages, top_k=0)
        with pytest.raises(ValueError, match="query"):
            SearchService.rank("   ", embeddings, pages)

    def test_pdf_search_scores_descending(self):
        pdf = _make_pdf()
        results = pdf.search("test")

        scores = [p._search_score for p in results]
        assert scores == sorted(scores, reverse=True)

    def test_pdf_search_rejects_bad_inputs(self):
        """PDF.search validates before computing embeddings."""
        pdf = _make_pdf()
        with pytest.raises(ValueError, match="query"):
            pdf.search("   ")
        with pytest.raises(ValueError, match="top_k"):
            pdf.search("valid query", top_k=0)


class TestCollectionSearchValidation:
    def test_empty_collection_search_rejects_empty_query(self):
        """An empty collection must reject bad arguments exactly like a
        populated one, not short-circuit past validation."""
        from natural_pdf.core.pdf_collection import PDFCollection

        collection = PDFCollection([])
        with pytest.raises(ValueError, match="query"):
            collection.search("")
        with pytest.raises(ValueError, match="query"):
            collection.search("   ")

    def test_empty_collection_search_rejects_bad_top_k(self):
        from natural_pdf.core.pdf_collection import PDFCollection

        collection = PDFCollection([])
        with pytest.raises(ValueError, match="top_k"):
            collection.search("valid query", top_k=0)

    def test_empty_collection_valid_args_returns_empty(self):
        from natural_pdf.core.pdf_collection import PDFCollection

        collection = PDFCollection([])
        results = collection.search("valid query")
        assert len(results) == 0


class TestTextFingerprint:
    def test_stable_for_same_texts(self):
        from natural_pdf.search.search_service import SearchService

        assert SearchService.text_fingerprint(["a", "b"]) == SearchService.text_fingerprint(
            ["a", "b"]
        )

    def test_sensitive_to_boundaries(self):
        from natural_pdf.search.search_service import SearchService

        assert SearchService.text_fingerprint(["ab", "c"]) != SearchService.text_fingerprint(
            ["a", "bc"]
        )

    def test_no_collision_on_boundary_nuls(self):
        """The old NUL-separated concatenation hashed these two identically."""
        from natural_pdf.search.search_service import SearchService

        assert SearchService.text_fingerprint(["a\x00", "b"]) != SearchService.text_fingerprint(
            ["a", "\x00b"]
        )


def _make_pdf():
    """Create a real PDF object from the test file."""
    from natural_pdf import PDF

    return PDF("pdfs/01-practice.pdf")
