import os
import glob as py_glob
import logging
from typing import List, Optional, Dict, Any, Union, Iterable, Set, TYPE_CHECKING, Type
from pathlib import Path
from PIL import Image
import re # Added for safe path generation
import copy # Added for copying options
from tqdm import tqdm

# Set up logger early
logger = logging.getLogger(__name__)

from natural_pdf.core.pdf import PDF
from natural_pdf.elements.region import Region 

# --- Search Imports ---
try:
    from natural_pdf.search.search_service_protocol import (
         SearchServiceProtocol, SearchOptions, Indexable
     )
    from natural_pdf.search.searchable_mixin import SearchableMixin
except ImportError as e:
    logger_init = logging.getLogger(__name__)
    logger_init.error(f"Failed to import search components. Search functionality disabled. Error: {e}", exc_info=True)
    # Dummy definitions
    class SearchableMixin: pass
    SearchServiceProtocol, SearchOptions, Indexable = object, object, object

from natural_pdf.search.searchable_mixin import SearchableMixin # Import the new mixin

class PDFCollection(SearchableMixin): # Inherit from the mixin
    def __init__(self,
                 source: Union[str, Iterable[Union[str, 'PDF']]],
                 recursive: bool = True,
                 **pdf_options: Any):
        """
        Initializes a collection of PDF documents from various sources.

        Args:
            source: The source of PDF documents. Can be:
                - An iterable (e.g., list) of existing PDF objects.
                - An iterable (e.g., list) of file paths/URLs/globs (strings).
                - A single file path/URL/directory/glob string.
            recursive: If source involves directories or glob patterns,
                       whether to search recursively (default: True).
            **pdf_options: Keyword arguments passed to the PDF constructor.
        """
        self._pdfs: List['PDF'] = []
        self._pdf_options = pdf_options # Store options for potential slicing later
        self._recursive = recursive # Store setting for potential slicing

        # Dynamically import PDF class within methods to avoid circular import at module load time
        PDF = self._get_pdf_class()

        if hasattr(source, '__iter__') and not isinstance(source, str):
             source_list = list(source)
             if not source_list: return # Empty list source
             if isinstance(source_list[0], PDF):
                  if all(isinstance(item, PDF) for item in source_list):
                       self._pdfs = source_list # Direct assignment
                       # Don't adopt search context anymore
                       return
                  else: raise TypeError("Iterable source has mixed PDF/non-PDF objects.")
             # If it's an iterable but not PDFs, fall through to resolve sources

        # Resolve string, iterable of strings, or single string source to paths/URLs
        resolved_paths_or_urls = self._resolve_sources_to_paths(source)
        self._initialize_pdfs(resolved_paths_or_urls, PDF) # Pass PDF class

        self._iter_index = 0

        # Initialize internal search service reference
        self._search_service: Optional[SearchServiceProtocol] = None

    @staticmethod
    def _get_pdf_class():
        """Helper method to dynamically import the PDF class."""
        try:
            # Import needs to resolve path correctly
            from natural_pdf.core.pdf import PDF
            return PDF
        except ImportError as e:
            logger.error("Could not import PDF class from natural_pdf.core.pdf. Ensure it exists and there are no circular imports at runtime.")
            raise ImportError("PDF class is required but could not be imported.") from e

    # --- Internal Helpers ---

    def _is_url(self, s: str) -> bool: return s.startswith(('http://', 'https://'))
    def _has_glob_magic(self, s: str) -> bool: return py_glob.has_magic(s)

    def _execute_glob(self, pattern: str) -> Set[str]:
        """Glob for paths and return a set of valid PDF paths."""
        found_paths = set()
        try:
            # Use iglob for potentially large directories/matches
            paths_iter = py_glob.iglob(pattern, recursive=self._recursive)
            for path_str in paths_iter:
                 # Use Path object for easier checking
                 p = Path(path_str)
                 if p.is_file() and p.suffix.lower() == ".pdf":
                      found_paths.add(str(p.resolve())) # Store resolved absolute path
        except Exception as e:
            logger.error(f"Error processing glob pattern '{pattern}': {e}")
        return found_paths

    def _resolve_sources_to_paths(self, source: Union[str, Iterable[str]]) -> List[str]:
        """Resolves various source types into a list of unique PDF paths/URLs."""
        final_paths = set()
        sources_to_process = []

        if isinstance(source, str):
            sources_to_process.append(source)
        elif hasattr(source, '__iter__'):
            sources_to_process.extend(list(source))
        else: # Should not happen based on __init__ checks, but safeguard
             raise TypeError(f"Unexpected source type in _resolve_sources_to_paths: {type(source)}")

        for item in sources_to_process:
             if not isinstance(item, str):
                  logger.warning(f"Skipping non-string item in source list: {type(item)}")
                  continue
             
             item_path = Path(item)
             
             if self._is_url(item):
                 final_paths.add(item) # Add URL directly
             elif self._has_glob_magic(item):
                 glob_results = self._execute_glob(item)
                 final_paths.update(glob_results)
             elif item_path.is_dir():
                 # Use glob to find PDFs in directory, respecting recursive flag
                 dir_pattern = str(item_path / "**" / "*.pdf") if self._recursive else str(item_path / "*.pdf")
                 dir_glob_results = self._execute_glob(dir_pattern)
                 final_paths.update(dir_glob_results)
             elif item_path.is_file() and item_path.suffix.lower() == ".pdf":
                 final_paths.add(str(item_path.resolve())) # Add resolved file path
             else:
                 logger.warning(f"Source item ignored (not a valid URL, directory, file, or glob): {item}")
                 
        return sorted(list(final_paths))

    def _initialize_pdfs(self, paths_or_urls: List[str], PDF_cls: Type):
        """Initializes PDF objects from a list of paths/URLs."""
        logger.info(f"Initializing {len(paths_or_urls)} PDF objects...")
        failed_count = 0
        for path_or_url in tqdm(paths_or_urls, desc="Loading PDFs"):
            try:
                pdf_instance = PDF_cls(path_or_url, **self._pdf_options)
                self._pdfs.append(pdf_instance)
            except Exception as e:
                 logger.error(f"Failed to load PDF: {path_or_url}. Error: {e}", exc_info=False) # Keep log concise
                 failed_count += 1
        logger.info(f"Successfully initialized {len(self._pdfs)} PDFs. Failed: {failed_count}")

    # --- Public Factory Class Methods (Simplified) ---

    @classmethod
    def from_paths(cls, paths_or_urls: List[str], **pdf_options: Any) -> 'PDFCollection':
        """Creates a PDFCollection explicitly from a list of file paths or URLs."""
        # __init__ can handle List[str] directly now
        return cls(paths_or_urls, **pdf_options)

    @classmethod
    def from_glob(cls, pattern: str, recursive: bool = True, **pdf_options: Any) -> 'PDFCollection':
        """Creates a PDFCollection explicitly from a single glob pattern."""
        # __init__ can handle single glob string directly
        return cls(pattern, recursive=recursive, **pdf_options)

    @classmethod
    def from_globs(cls, patterns: List[str], recursive: bool = True, **pdf_options: Any) -> 'PDFCollection':
        """Creates a PDFCollection explicitly from a list of glob patterns."""
         # __init__ can handle List[str] containing globs directly
        return cls(patterns, recursive=recursive, **pdf_options)

    @classmethod
    def from_directory(cls, directory_path: str, recursive: bool = True, **pdf_options: Any) -> 'PDFCollection':
        """Creates a PDFCollection explicitly from PDF files within a directory."""
        # __init__ can handle single directory string directly
        return cls(directory_path, recursive=recursive, **pdf_options)

    # --- Core Collection Methods ---
    def __len__(self) -> int:
        return len(self._pdfs)

    def __getitem__(self, key) -> Union['PDF', 'PDFCollection']:
        # Use dynamic import here as well
        PDF = self._get_pdf_class()
        if isinstance(key, slice):
            # Create a new collection with the sliced PDFs and original options
            new_collection = PDFCollection.__new__(PDFCollection) # Create blank instance
            new_collection._pdfs = self._pdfs[key]
            new_collection._pdf_options = self._pdf_options
            new_collection._recursive = self._recursive
            # Search context is not copied/inherited anymore
            return new_collection
        elif isinstance(key, int):
            # Check bounds
            if 0 <= key < len(self._pdfs):
                return self._pdfs[key]
            else:
                 raise IndexError(f"PDF index {key} out of range (0-{len(self._pdfs)-1}).")
        else:
             raise TypeError(f"PDF indices must be integers or slices, not {type(key)}.")

    def __iter__(self):
        return iter(self._pdfs)

    def __repr__(self) -> str:
        # Removed search status
        return f"<PDFCollection(count={len(self)})>"

    @property
    def pdfs(self) -> List['PDF']:
         """Returns the list of PDF objects held by the collection."""
         return self._pdfs

    # --- Other Methods (e.g., apply_ocr - could leverage service in future?) ---
    def apply_ocr(self, *args, **kwargs):
        PDF = self._get_pdf_class()
        # Delegate to individual PDF objects
        logger.info("Applying OCR to relevant PDFs in collection...")
        results = []
        for pdf in self._pdfs:
             # We need to figure out which pages belong to which PDF if batching here
             # For now, simpler to call on each PDF
             try:
                 # Assume apply_ocr exists on PDF and accepts similar args
                 pdf.apply_ocr(*args, **kwargs)
             except Exception as e:
                 logger.error(f"Failed applying OCR to {pdf.path}: {e}", exc_info=True)
        return self

    # --- Advanced Method Placeholders ---
    # Placeholder for categorize removed as find_relevant is now implemented

    def categorize(self, categories: List[str], **kwargs):
        """Categorizes PDFs in the collection based on content or features."""
        # Implementation requires integrating with classification models or logic
        raise NotImplementedError("categorize requires classification implementation.") 

    # --- Mixin Required Implementation --- 
    def get_indexable_items(self) -> Iterable[Indexable]:
        """Yields Page objects from the collection, conforming to Indexable."""
        if not self._pdfs:
             return # Return empty iterator if no PDFs
             
        for pdf in self._pdfs:
             if not pdf.pages: # Handle case where a PDF might have 0 pages after loading
                 logger.warning(f"PDF '{pdf.path}' has no pages. Skipping.")
                 continue
             for page in pdf.pages:
                 # Optional: Add filtering here if needed (e.g., skip empty pages)
                 # Assuming Page object conforms to Indexable
                 # We might still want the empty page check here for efficiency
                 # if not page.extract_text(use_exclusions=False).strip():
                 #     logger.debug(f"Skipping empty page {page.page_number} from PDF '{pdf.path}'.")
                 #     continue
                 yield page 