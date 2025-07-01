"""Sequence wrapper for table data with convenient DataFrame helpers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Iterator, List, Optional, Union


class TableResult(Sequence):
    """List-of-rows plus `.df` / `.to_df()` helpers.

    The object behaves like an immutable sequence of rows (each row is a
    list of cell values) but offers an easy hand-off to *pandas*.
    """

    _IMMUTABLE_MESSAGE = "TableResult is read-only; convert to list(result) if you need to mutate"

    def __init__(self, rows: Optional[List[List[Any]]] = None) -> None:
        # Normalise to list of list so that Sequence operations work as expected
        self._rows: List[List[Any]] = list(rows or [])

    # ---------------------------------------------------------------------
    # Sequence API
    # ---------------------------------------------------------------------
    def __getitem__(self, index):  # type: ignore[override]
        return self._rows[index]

    def __len__(self) -> int:  # type: ignore[override]
        return len(self._rows)

    def __iter__(self) -> Iterator[List[Any]]:  # type: ignore[override]
        return iter(self._rows)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------
    @property
    def df(self):
        """Quick property alias → calls :py:meth:`to_df` with default args."""
        return self.to_df()

    def to_df(
        self,
        header: Union[str, int, List[int], None] = "first",
        index_col=None,
        skip_repeating_headers=None,
        **kwargs,
    ):
        """Convert to *pandas* DataFrame.

        Parameters
        ----------
        header : "first" | int | list[int] | None, default "first"
            • "first" – use row 0 as column names.\n            • int       – use that row index.\n            • list[int] – multi-row header.\n            • None/False– no header.
        index_col : same semantics as pandas, forwarded.
        skip_repeating_headers : bool, optional
            Whether to remove body rows that exactly match the header row(s).
            Defaults to True when header is truthy, False otherwise.
            Useful for PDFs where headers repeat throughout the table body.
        **kwargs  : forwarded to :pyclass:`pandas.DataFrame`.
        """
        try:
            import pandas as pd  # type: ignore
        except ModuleNotFoundError as exc:
            raise ImportError(
                "pandas is required for TableResult.to_df(); install via `pip install pandas`."
            ) from exc

        rows = self._rows
        if not rows:
            return pd.DataFrame()

        # Determine default for skip_repeating_headers based on header parameter
        if skip_repeating_headers is None:
            skip_repeating_headers = header is not None and header is not False

        # Determine header rows and body rows
        body = rows
        hdr = None
        if header == "first":
            hdr = rows[0]
            body = rows[1:]
        elif header is None or header is False:
            hdr = None
        elif isinstance(header, int):
            hdr = rows[header]
            body = rows[:header] + rows[header + 1 :]
        elif isinstance(header, (list, tuple)):
            hdr_rows = [rows[i] for i in header]
            body = [r for idx, r in enumerate(rows) if idx not in header]
            hdr = hdr_rows
        else:
            raise ValueError("Invalid value for header parameter")

        # Skip repeating headers in body if requested
        if skip_repeating_headers and hdr is not None and body:
            original_body_len = len(body)
            if isinstance(hdr, list) and len(hdr) > 0 and not isinstance(hdr[0], list):
                # Single header row (most common case)
                body = [row for row in body if row != hdr]
            elif isinstance(hdr, list) and len(hdr) > 0 and isinstance(hdr[0], list):
                # Multi-row header (less common)
                hdr_set = {tuple(h) if isinstance(h, list) else h for h in hdr}
                body = [
                    row
                    for row in body
                    if (tuple(row) if isinstance(row, list) else row) not in hdr_set
                ]

            skipped_count = original_body_len - len(body)
            if skipped_count > 0:
                # Could add logging here if desired
                pass

        df = pd.DataFrame(body, columns=hdr)
        if index_col is not None and not df.empty:
            df.set_index(
                df.columns[index_col] if isinstance(index_col, int) else index_col, inplace=True
            )

        if kwargs:
            df = pd.DataFrame(df, **kwargs)
        return df

    # ------------------------------------------------------------------
    # Block mutating operations to keep result read-only
    # ------------------------------------------------------------------
    def _readonly(self, *args, **kwargs):
        raise TypeError(self._IMMUTABLE_MESSAGE)

    append = extend = insert = __setitem__ = __delitem__ = clear = pop = remove = _readonly  # type: ignore

    # Nice repr in notebooks
    def __repr__(self) -> str:  # noqa: D401 (simple)
        preview = "…" if len(self._rows) > 5 else ""
        return f"TableResult(rows={len(self._rows)}{preview})"
