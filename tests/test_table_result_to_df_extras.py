"""Additional tests for TableResult.to_df conveniences."""

import pandas as pd
import pytest

from natural_pdf.tables.result import TableResult


def test_to_df_skiprows_integer():
    rows = [
        ["meta", "meta"],
        ["junk", "junk"],
        ["col1", "col2"],
        ["a1", "b1"],
        ["a2", "b2"],
    ]
    table = TableResult(rows)

    df = table.to_df(skiprows=2)

    assert list(df.columns) == ["col1", "col2"]
    assert df.shape == (2, 2)
    assert df.iloc[0].tolist() == ["a1", "b1"]


def test_to_df_skiprows_iterable():
    rows = [
        ["col1", "col2"],
        ["repeat", "repeat"],
        ["row1", "row2"],
        ["row3", "row4"],
    ]
    table = TableResult(rows)

    df = table.to_df(skiprows=[1])

    assert list(df.columns) == ["col1", "col2"]
    assert df.shape == (2, 2)
    assert df.iloc[-1].tolist() == ["row3", "row4"]


def test_to_df_dtype_argument():
    rows = [
        ["col1", "col2"],
        ["1", "2"],
    ]
    table = TableResult(rows)

    df = table.to_df(header="first", dtype="string")

    assert df.dtypes.tolist() == [pd.StringDtype(), pd.StringDtype()]


def test_to_df_copy_argument():
    rows = [
        ["col1", "col2"],
        ["1", "2"],
    ]
    table = TableResult(rows)

    df = table.to_df(header="first", copy=True)

    assert df.equals(pd.DataFrame([["1", "2"]], columns=["col1", "col2"]))


def test_to_df_skiprows_negative_raises():
    table = TableResult([["col1"], ["value"]])

    with pytest.raises(ValueError, match="skiprows must be >= 0"):
        table.to_df(skiprows=-1)
