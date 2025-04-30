from typing import Any, Generic, Optional, TypeVar

from pydantic import BaseModel, Field

# Generic type for the Pydantic model used in the schema
T_Schema = TypeVar("T_Schema", bound=BaseModel)


class StructuredDataResult(BaseModel, Generic[T_Schema]):
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
