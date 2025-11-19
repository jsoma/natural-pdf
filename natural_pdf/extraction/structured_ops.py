import base64
import io
import logging
from typing import Any, Dict, List, Optional, Type

from PIL import Image
from pydantic import BaseModel

from natural_pdf.extraction.result import StructuredDataResult

logger = logging.getLogger(__name__)

DEFAULT_TEXT_MODEL = "gpt-4o-mini"
DEFAULT_VISION_MODEL = "gpt-4o"


def structured_data_is_available() -> bool:
    """Checks if the structured data dependencies are installed."""
    try:
        import pydantic  # noqa: F401

        return True
    except ImportError:
        logger.warning("Pydantic is required for structured data extraction.")
        return False


def _prepare_llm_messages(
    content: Any, prompt: Optional[str], using: str, schema: Type[BaseModel]
) -> List[Dict[str, Any]]:
    """Prepare message payloads for a structured LLM call."""
    system_prompt = (
        prompt
        or f"Extract the information corresponding to the fields in the {schema.__name__} schema. Respond only with the structured data."
    )

    messages: List[Dict[str, Any]] = [{"role": "system", "content": system_prompt}]

    if using == "text":
        messages.append({"role": "user", "content": str(content)})
    elif using == "vision":
        if isinstance(content, Image.Image):
            buffered = io.BytesIO()
            content.save(buffered, format="PNG")
            base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Extract information from this image based on the schema.",
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                        },
                    ],
                }
            )
        else:
            raise TypeError(f"Content must be a PIL Image for using='vision', got {type(content)}")
    else:
        raise ValueError(f"Unsupported value for 'using': {using}")

    return messages


def extract_structured_data(
    *,
    content: Any,
    schema: Type[BaseModel],
    client: Any,
    prompt: Optional[str] = None,
    using: str = "text",
    model: Optional[str] = None,
    **kwargs,
) -> StructuredDataResult:
    """Extract structured data from the provided content using the configured LLM client."""
    if isinstance(content, list) and using == "vision":
        if len(content) == 1:
            content = content[0]
        elif len(content) > 1:
            logger.error("Vision extraction not supported for multi-page PDFs")
            raise NotImplementedError(
                "Batch image extraction on multi-page PDF objects is not supported. Apply to individual pages or regions instead."
            )

    selected_model = model or (DEFAULT_VISION_MODEL if using == "vision" else DEFAULT_TEXT_MODEL)
    messages = _prepare_llm_messages(content, prompt, using, schema)

    logger.debug(
        "Structured data extract request: using='%s', schema='%s', model='%s'",
        using,
        schema.__name__,
        selected_model,
    )
    completion = client.beta.chat.completions.parse(
        model=selected_model, messages=messages, response_format=schema, **kwargs
    )
    parsed_data = completion.choices[0].message.parsed
    return StructuredDataResult(
        data=parsed_data,
        success=True,
        error_message=None,
        raw_output=completion,
        model_used=selected_model,
    )
