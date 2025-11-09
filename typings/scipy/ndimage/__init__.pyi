from typing import Any, Sequence, Tuple

def binary_closing(
    input: Any, structure: Any | None = ..., iterations: int = ..., **kwargs: Any
) -> Any: ...
def binary_opening(
    input: Any, structure: Any | None = ..., iterations: int = ..., **kwargs: Any
) -> Any: ...
def find_objects(input: Any, max_label: int | None = ...) -> Any: ...
def gaussian_filter1d(
    input: Any,
    sigma: float,
    axis: int = ...,
    order: int = ...,
    output: Any = ...,
    mode: str = ...,
    cval: float = ...,
    truncate: float = ...,
) -> Any: ...
def label(
    input: Any, structure: Any | None = ..., output: Any = ..., origin: int = ...
) -> Tuple[Any, int]: ...
def gaussian_filter(
    input: Any,
    sigma: float | Sequence[float],
    order: int | Sequence[int] = ...,
    output: Any = ...,
    mode: str = ...,
    cval: float = ...,
    truncate: float = ...,
) -> Any: ...
