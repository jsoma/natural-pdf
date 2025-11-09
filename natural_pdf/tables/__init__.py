"""Tables public exports."""

from natural_pdf.engine_registry import register_table_engine, register_table_function

from .result import TableResult

__all__ = ["TableResult", "register_table_engine", "register_table_function"]
