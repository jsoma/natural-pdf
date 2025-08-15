"""Match results for visual similarity search"""

from typing import TYPE_CHECKING, Any, Iterator, List, Optional, Tuple

from natural_pdf.elements.region import Region

if TYPE_CHECKING:
    from natural_pdf.core.page_collection import PageCollection
    from natural_pdf.elements.element_collection import ElementCollection


class Match(Region):
    """A region that was found via visual similarity search"""

    def __init__(self, page, bbox, confidence, source_example=None, metadata=None):
        """
        Initialize a Match object.

        Args:
            page: Page containing the match
            bbox: Bounding box of the match
            confidence: Similarity confidence (0-1)
            source_example: The example/template that led to this match
            metadata: Additional metadata about the match
        """
        super().__init__(page, bbox)
        self.confidence = confidence
        self.source_example = source_example
        self.metadata = metadata or {}

    @property
    def pdf(self):
        """Get the PDF containing this match"""
        return self.page.pdf

    def __repr__(self):
        return f"<Match page={self.page.number} confidence={self.confidence:.2f} bbox={self.bbox}>"


class MatchResults:
    """
    Collection of Match objects with transformation methods.

    Matches are automatically sorted by confidence (highest first), so:
    - matches[0] is the best match
    - Iteration yields matches from best to worst
    - The .top(n) method returns the n best matches

    Example:
        >>> matches = page.find_similar(logo_region)
        >>> print(f"Found {len(matches)} matches")
        >>>
        >>> # Best match
        >>> best = matches[0]
        >>> print(f"Best match confidence: {best.confidence:.3f}")
        >>>
        >>> # Top 5 matches
        >>> for match in matches.top(5):
        ...     print(f"Confidence: {match.confidence:.3f} at page {match.page.number}")
        >>>
        >>> # All matches above 90% confidence
        >>> high_conf = matches.filter_by_confidence(0.9)
    """

    def __init__(self, matches: List[Match]):
        """Initialize with list of Match objects, automatically sorted by confidence"""
        # Import here to avoid circular import
        from natural_pdf.elements.element_collection import ElementCollection

        # Sort matches by confidence (highest first)
        sorted_matches = sorted(matches, key=lambda m: m.confidence, reverse=True)

        # Create a base ElementCollection
        self._collection = ElementCollection(sorted_matches)
        self._matches = sorted_matches

    def __len__(self):
        return len(self._matches)

    def __iter__(self):
        return iter(self._matches)

    def __getitem__(self, key):
        return self._matches[key]

    def filter(self, filter_func) -> "MatchResults":
        """Filter matches by a function"""
        filtered = [m for m in self if filter_func(m)]
        return MatchResults(filtered)

    def filter_by_confidence(self, min_confidence: float) -> "MatchResults":
        """Filter matches by minimum confidence"""
        return self.filter(lambda m: m.confidence >= min_confidence)

    def top(self, n: int) -> "MatchResults":
        """
        Get the top N matches with highest confidence.

        Args:
            n: Number of top matches to return

        Returns:
            New MatchResults with only the top N matches

        Example:
            >>> matches = page.find_similar(logo)
            >>> best_5 = matches.top(5)
            >>> for match in best_5:
            ...     print(f"Confidence: {match.confidence:.3f}")
        """
        # Since matches are already sorted by confidence, just take first n
        top_matches = self._matches[:n]
        return MatchResults(top_matches)

    def pages(self):
        """Get unique pages containing matches"""
        # Import here to avoid circular import
        from natural_pdf.core.page_collection import PageCollection

        # Get unique pages while preserving order
        seen = set()
        unique_pages = []
        for match in self:
            if match.page not in seen:
                seen.add(match.page)
                unique_pages.append(match.page)

        # Attach matches to each page
        for page in unique_pages:
            page._matches = MatchResults([m for m in self if m.page == page])

        return PageCollection(unique_pages)

    def pdfs(self):
        """Get unique PDFs containing matches"""
        # Import here to avoid circular import
        from natural_pdf.core.pdf_collection import PDFCollection

        # Get unique PDFs while preserving order
        seen = set()
        unique_pdfs = []
        for match in self:
            if match.pdf not in seen:
                seen.add(match.pdf)
                unique_pdfs.append(match.pdf)

        # Attach matches to each PDF
        for pdf in unique_pdfs:
            pdf._matches = MatchResults([m for m in self if m.pdf == pdf])

        return PDFCollection(unique_pdfs)

    def group_by_page(self) -> Iterator[Tuple[Any, "MatchResults"]]:
        """Group matches by page"""
        from itertools import groupby

        # Sort by PDF filename and page number
        sorted_matches = sorted(self, key=lambda m: (getattr(m.pdf, "filename", ""), m.page.number))

        for page, matches in groupby(sorted_matches, key=lambda m: m.page):
            yield page, MatchResults(list(matches))

    def sort_by_confidence(self, descending: bool = True) -> "MatchResults":
        """Sort matches by confidence score"""
        sorted_matches = sorted(self, key=lambda m: m.confidence, reverse=descending)
        return MatchResults(sorted_matches)

    def regions(self):
        """Get all matches as an ElementCollection of regions"""
        # Import here to avoid circular import
        from natural_pdf.elements.element_collection import ElementCollection

        # Matches are already Region objects, so just wrap them
        return ElementCollection(list(self))

    def show(self, **kwargs):
        """Show all matches using ElementCollection.show()"""
        # Get regions and show them
        return self.regions().show(**kwargs)

    def __repr__(self):
        if len(self) == 0:
            return "<MatchResults: empty>"
        elif len(self) == 1:
            return f"<MatchResults: 1 match>"
        else:
            conf_range = (
                f"{min(m.confidence for m in self):.2f}-{max(m.confidence for m in self):.2f}"
            )
            return f"<MatchResults: {len(self)} matches, confidence {conf_range}>"
