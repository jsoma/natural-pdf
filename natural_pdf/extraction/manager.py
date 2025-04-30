import base64
import io
import logging
from typing import Any, Optional, Type

from PIL import Image
from pydantic import BaseModel

from natural_pdf.extraction.result import StructuredDataResult

logger = logging.getLogger(__name__)


class StructuredDataManager:
    """
    Manages the process of extracting structured data from elements using LLMs.

    This manager is typically accessed via `pdf.get_manager('structured_data')`.
    It is stateless and relies on parameters passed during method calls.
    """

    DEFAULT_TEXT_MODEL = "gpt-4o-mini"
    DEFAULT_VISION_MODEL = "gpt-4o"

    def __init__(self):
        """Initializes the manager."""
        logger.info("Initialized StructuredDataManager.")

    def is_available(self) -> bool:
        """Checks if necessary dependencies are available."""
        try:
            import pydantic

            return True
        except ImportError:
            logger.warning("Pydantic is required for structured data extraction.")
            return False

    def _prepare_llm_messages(
        self, content: Any, prompt: Optional[str], using: str, schema: Type[BaseModel]
    ) -> list:
        """Prepares the message list for the LLM API call."""
        system_prompt = (
            prompt
            or f"Extract the information corresponding to the fields in the {schema.__name__} schema. Respond only with the structured data."
        )

        messages = [{"role": "system", "content": system_prompt}]

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
                raise TypeError(
                    f"Content must be a PIL Image for using='vision', got {type(content)}"
                )
        else:
            raise ValueError(f"Unsupported value for 'using': {using}")

        return messages

    def extract(
        self,
        content: Any,
        schema: Type[BaseModel],
        client: Any,
        prompt: Optional[str] = None,
        using: str = "text",
        model: Optional[str] = None,
        **kwargs,
    ) -> StructuredDataResult:
        """
        Extract structured data from content using an LLM.

        Args:
            content: Text string or Image object
            schema: Pydantic model class for the desired structure
            client: Initialized LLM client (e.g., OpenAI client)
            prompt: Optional user-provided instructions
            using: Modality ('text' or 'vision')
            model: Specific LLM model identifier
            **kwargs: Additional parameters for the LLM API call

        Returns:
            StructuredDataResult object
        """
        logger.debug(f"Extract request: using='{using}', schema='{schema.__name__}'")

        if isinstance(content, list) and using == "vision":
            if len(content) == 1:
                content = content[0]
            elif len(content) > 1:
                logger.error("Vision extraction not supported for multi-page PDFs")
                raise NotImplementedError(
                    "Batch image extraction on multi-page PDF objects is not supported. Apply to individual pages or regions instead."
                )

        selected_model = model or (
            self.DEFAULT_VISION_MODEL if using == "vision" else self.DEFAULT_TEXT_MODEL
        )
        messages = self._prepare_llm_messages(content, prompt, using, schema)

        try:
            logger.debug(f"Extracting with model '{selected_model}'")
            completion = client.beta.chat.completions.parse(
                model=selected_model, messages=messages, response_format=schema, **kwargs
            )
            parsed_data = completion.choices[0].message.parsed
            return StructuredDataResult(
                data=parsed_data, success=True, error_message=None, model=selected_model
            )
        except Exception as e:
            logger.error(f"Extraction failed: {str(e)}")
            return StructuredDataResult(
                data=None, success=False, error_message=str(e), model=selected_model
            )
