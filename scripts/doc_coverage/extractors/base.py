"""Base data structures for code sample extraction."""

from dataclasses import dataclass


@dataclass
class CodeSample:
    """A code sample extracted from documentation.

    Attributes:
        source: The Python source code.
        file_path: Path to the file containing this sample.
        location: Location within the file (line number or cell index).
    """

    source: str
    file_path: str
    location: str
