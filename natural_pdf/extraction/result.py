from collections.abc import Mapping
from typing import Any, Generic, Optional, TypeVar

from pydantic import BaseModel, Field

# Generic type for the Pydantic model used in the schema
T_Schema = TypeVar("T_Schema", bound=BaseModel)


class StructuredDataResult(BaseModel, Generic[T_Schema], Mapping):
    """
    Represents the result of a structured data extraction operation.

    Contains the extracted data, success status, and error information.
    """

    data: Optional[T_Schema] = Field(None, description="Validated data model or None on failure")
    success: bool = Field(..., description="Whether extraction succeeded")
    error_message: Optional[str] = Field(None, description="Error details if extraction failed")
    raw_output: Optional[Any] = Field(None, description="Raw output from the language model")
    model_used: Optional[str] = Field(None, description="Identifier of the language model used")

    class Config:
        arbitrary_types_allowed = True

    # ---------------------------------------------------------------------
    # Mapping interface implementation
    # ---------------------------------------------------------------------

    def _as_dict(self) -> dict:
        """Return the underlying data as a plain dict (Pydantic v1 & v2 safe)."""
        if hasattr(self, "model_dump"):
            # Pydantic v2
            return self.model_dump()
        else:
            # Pydantic v1
            return self.dict()

    def __iter__(self):
        """Iterate over keys, preserving insertion order guaranteed in Py≥3.7."""
        return iter(self._as_dict())

    def __getitem__(self, key):
        try:
            return self._as_dict()[key]
        except KeyError as exc:
            raise KeyError(key) from exc

    def __len__(self):
        return len(self._as_dict())
