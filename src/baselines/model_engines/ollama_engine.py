"""
LiteLLM-compatible wrapper for Ollama.
"""

import logging
from typing import Any, Dict, List, Optional, Union

from src.baselines.model_engines.base_engine import BaseModelEngine
from src.config.baselines_model_config import ModelConfig
from src.utils.ollama_client import OllamaClient

logger = logging.getLogger(__name__)


class OllamaModelEngine(BaseModelEngine):
    """
    LiteLLM-compatible wrapper for Ollama to work with standardized interface.
    """

    def __init__(
        self,
        ollama_host: str = "http://localhost:11434/",
        model_path: Optional[str] = None,
        config: Optional[ModelConfig] = None,
        task: str = "writing",
    ):
        # Initialize base class
        super().__init__(model_path=model_path, config=config, task=task)

        # Ensure we're using ollama mode
        if self.config.mode != "ollama":
            self.config.mode = "ollama"
            logger.info("Switched to ollama mode for model configuration")

        # Create ollama client
        self.client = OllamaClient(host=ollama_host)

        # Get model name from path for Ollama (which uses different naming)
        self.model = self.config.ollama_model_mapping.get(
            self.model_path, self.model_path
        )

        # LiteLLM compatibility attributes
        self.model_name = self.model
        self.kwargs = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        logger.info(f"OllamaModelEngine initialized with model: {self.model}")

    def __call__(self, messages=None, **kwargs):
        """Make wrapper callable for compatibility."""
        if "max_tokens" not in kwargs and "max_output_tokens" not in kwargs:
            kwargs["max_tokens"] = self.max_tokens

        if messages is not None:
            return self.complete(messages, **kwargs)

        # Handle string prompts
        if isinstance(kwargs.get("prompt"), str):
            return self._create_response(
                self._generate(kwargs["prompt"], **kwargs), kwargs["prompt"]
            )

        # Default case
        return self.complete(str(kwargs), **kwargs)

    def list_available_models(self) -> List[str]:
        """List available models in Ollama."""
        try:
            return self.client.list_models()
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return [self.model]

    def generate(
        self,
        prompt: str,
        max_length: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """Generate content from a prompt."""
        # Use provided params or defaults
        max_length = max_length or self.max_tokens
        temperature = temperature if temperature is not None else self.temperature

        return self._generate(
            prompt,
            max_tokens=max_length,
            temperature=temperature,
        )

    def complete(self, messages: Union[str, List[Dict]], **kwargs) -> Any:
        """Complete messages in chat format."""
        try:
            # Parse messages to a prompt if needed
            prompt, system_prompt = self._parse_messages(messages)

            # Generate response
            response_text = self._generate(
                prompt=prompt, system=system_prompt, **kwargs
            )

            return self._create_response(response_text, prompt)

        except Exception as e:
            logger.error(f"Completion failed: {e}")
            return self._create_error_response(str(e))

    def _generate(self, prompt: str, **kwargs) -> str:
        """Direct generation method using Ollama client."""
        system = kwargs.get("system", None)

        response_text = self.client.call_api(
            prompt=prompt,
            system=system,
            model=kwargs.get("model", self.model),
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
        )
        return response_text

    def _parse_messages(self, messages: Union[str, List[Dict]]) -> tuple:
        """Parse various message formats."""
        if isinstance(messages, str):
            return messages, None

        if not isinstance(messages, list):
            return str(messages), None

        prompt_parts = []
        system_prompt = None

        for message in messages:
            if not isinstance(message, dict):
                prompt_parts.append(str(message))
                continue

            role = message.get("role", "").lower()
            content = message.get("content", "")

            if role == "system":
                system_prompt = content
            elif role == "user":
                prompt_parts.append(content)
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
            else:
                prompt_parts.append(content)

        return "\n".join(prompt_parts), system_prompt

    def _create_response(self, content: str, prompt: str) -> Any:
        """Create LiteLLM-compatible response object."""

        class Response:
            def __init__(self, content, model):
                self.content = content
                self.model = model
                self.choices = [
                    type(
                        "Choice",
                        (),
                        {
                            "message": type("Message", (), {"content": content}),
                            "text": content,
                        },
                    )
                ]

        return Response(content, self.model)

    def _create_error_response(self, error_msg: str) -> Any:
        """Create an error response object."""

        class ErrorResponse:
            def __init__(self, error):
                self.error = error
                self.content = f"Error: {error}"
                self.choices = []

        return ErrorResponse(error_msg)
