from __future__ import annotations

from typing import Any, Optional, Tuple, Union

ExclusionEntry = Tuple[Any, Optional[str], str]
ExclusionSpec = Union[ExclusionEntry, Tuple[Any, Optional[str]]]

__all__ = ["ExclusionEntry", "ExclusionSpec"]
