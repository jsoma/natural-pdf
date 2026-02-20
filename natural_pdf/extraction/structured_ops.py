import base64
import io
import json
import logging
from typing import Any, Dict, List, Optional, Type

from PIL import Image
from pydantic import BaseModel

from natural_pdf.extraction.result import StructuredDataResult

logger = logging.getLogger(__name__)

DEFAULT_TEXT_MODEL = "gpt-4o-mini"
DEFAULT_VISION_MODEL = "gpt-4o"

# Errors that indicate transport / auth problems — never swallow these.
_TRANSPORT_ERRORS = (ConnectionError, TimeoutError, OSError)


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


def _prepare_json_schema_messages(
    content: Any, prompt: Optional[str], using: str, schema: Type[BaseModel]
) -> List[Dict[str, Any]]:
    """Prepare messages with JSON schema instruction embedded in the prompt.

    Used for Tier 2/3 fallback when the client does not support native
    structured output.
    """
    schema_json = schema.model_json_schema()
    schema_instruction = (
        f"\n\nRespond ONLY with valid JSON matching this schema:\n"
        f"```json\n{json.dumps(schema_json, indent=2)}\n```"
    )
    base_prompt = (
        prompt
        or f"Extract the information corresponding to the fields in the {schema.__name__} schema."
    )
    return _prepare_llm_messages(content, base_prompt + schema_instruction, using, schema)


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
    """Extract structured data using a tiered approach.

    Tier 1: ``client.beta.chat.completions.parse(response_format=schema)``
        — native structured output (OpenAI, etc.)
    Tier 2: ``client.chat.completions.create(response_format={"type": "json_object"})``
        + JSON parse — works with Ollama, LM Studio, vLLM json_object mode
    Tier 3: ``client.chat.completions.create()`` + fuzzy JSON extraction
        — broadest compatibility
    """
    from natural_pdf.extraction.json_parser import parse_json_response

    if isinstance(content, list) and using == "vision":
        if len(content) == 1:
            content = content[0]
        elif len(content) > 1:
            logger.error("Vision extraction not supported for multi-page PDFs")
            raise NotImplementedError(
                "Batch image extraction on multi-page PDF objects is not supported. "
                "Apply to individual pages or regions instead."
            )

    selected_model = model or (DEFAULT_VISION_MODEL if using == "vision" else DEFAULT_TEXT_MODEL)
    messages = _prepare_llm_messages(content, prompt, using, schema)

    logger.debug(
        "Structured data extract request: using='%s', schema='%s', model='%s'",
        using,
        schema.__name__,
        selected_model,
    )

    # ---- Tier 1: Native structured output ---- #
    # First, check if the client even supports beta.chat.completions.parse
    parse_fn = getattr(
        getattr(getattr(client, "beta", None), "chat", None),
        "completions",
        None,
    )
    parse_fn = getattr(parse_fn, "parse", None)

    if parse_fn is not None:
        try:
            completion = parse_fn(
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
        except _TRANSPORT_ERRORS:
            raise
        except Exception as exc:
            # API returned an error for response_format — try json_object mode
            exc_str = str(exc).lower()
            if (
                "response_format" in exc_str
                or "structured_output" in exc_str
                or "invalid_response" in exc_str
            ):
                logger.debug("Tier 1 failed (%s), trying Tier 2.", exc)
            else:
                raise
    else:
        logger.debug("Tier 1 (structured output) not available, trying Tier 2.")

    # ---- Tier 2: JSON object mode ---- #
    json_messages = _prepare_json_schema_messages(content, prompt, using, schema)
    try:
        completion = client.chat.completions.create(
            model=selected_model,
            messages=json_messages,
            response_format={"type": "json_object"},
            **kwargs,
        )
        raw_text = completion.choices[0].message.content or ""
        parsed_data = parse_json_response(raw_text, schema)
        return StructuredDataResult(
            data=parsed_data,
            success=True,
            error_message=None,
            raw_output=completion,
            model_used=selected_model,
        )
    except (TypeError, NotImplementedError):
        logger.debug("Tier 2 (json_object) not available, trying Tier 3.")
    except _TRANSPORT_ERRORS:
        raise
    except ValueError:
        # JSON parse / validation failure from parse_json_response — try Tier 3
        logger.debug(
            "Tier 2: model returned invalid JSON, retrying without json_object constraint."
        )
    except Exception as exc:
        exc_str = str(exc).lower()
        if "response_format" in exc_str or "json" in exc_str:
            logger.debug("Tier 2 failed (%s), trying Tier 3.", exc)
        else:
            raise

    # ---- Tier 3: Free text + fuzzy JSON extraction ---- #
    raw_text = None
    try:
        completion = client.chat.completions.create(
            model=selected_model,
            messages=json_messages,
            **kwargs,
        )
        raw_text = completion.choices[0].message.content or ""
        parsed_data = parse_json_response(raw_text, schema)
        return StructuredDataResult(
            data=parsed_data,
            success=True,
            error_message=None,
            raw_output=completion,
            model_used=selected_model,
        )
    except _TRANSPORT_ERRORS:
        raise
    except Exception as exc:
        return StructuredDataResult(
            data=None,
            success=False,
            error_message=f"All structured output tiers failed: {exc}",
            raw_output=raw_text,
            model_used=selected_model,
        )
