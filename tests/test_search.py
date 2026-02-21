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

    def test_get_model_caching(self):
        from natural_pdf.search.search_service import SearchService

        model1 = SearchService.get_model("all-MiniLM-L6-v2")
        model2 = SearchService.get_model("all-MiniLM-L6-v2")
        assert model1 is model2


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

        # Second search should reuse cache
        pdf.search("query two")
        assert pdf._search_embeddings["all-MiniLM-L6-v2"] is cached

    def test_pdf_search_scores_descending(self):
        pdf = _make_pdf()
        results = pdf.search("test")

        scores = [p._search_score for p in results]
        assert scores == sorted(scores, reverse=True)


def _make_pdf():
    """Create a real PDF object from the test file."""
    from natural_pdf import PDF

    return PDF("pdfs/01-practice.pdf")
