"""
Simplified Ollama engine that focuses purely on Ollama API interaction.
Configuration is handled by ConfigContext.
"""

import logging
import re
from ollama import Client
from pydantic import BaseModel
from pydantic_core import ValidationError
from typing import Any, Dict, List, Optional, Type, TypeVar, Union

from src.engines.base_engine import BaseEngine
from src.utils.json_normalizer import normalize_llm_json

logger = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)

# Default Ollama host (UKP server)
DEFAULT_OLLAMA_HOST = "http://10.167.31.201:11434/"


class OllamaEngine(BaseEngine):
    """
    Ollama engine that provides both simple and LiteLLM-compatible interfaces.
    Simplified to focus only on Ollama API calls - no configuration parsing.
    """

    def __init__(
        self,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        host: str = DEFAULT_OLLAMA_HOST,
        task: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize Ollama engine with parameters from ConfigContext.

        Args:
            model: Ollama model name
            temperature: Generation temperature
            max_tokens: Maximum tokens to generate
            host: Ollama server URL
            **kwargs: Additional parameters
        """
        super().__init__(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            task=task,
            **kwargs,
        )

        self.host = host
        self.client = Client(host=host)
        self._available_models = None

        logger.info(
            f"OllamaEngine initialized with host: {host}, model: {model}, max_tokens: {max_tokens}"
        )

    def __call__(self, messages=None, **kwargs):
        """Make client callable for STORM compatibility."""
        if "max_tokens" not in kwargs and "max_output_tokens" not in kwargs:
            kwargs["max_tokens"] = self.max_tokens

        if messages is not None:
            return self.complete(messages, **kwargs)

        # Handle string prompts
        if isinstance(kwargs.get("prompt"), str):
            return self.complete(kwargs["prompt"], **kwargs)

        # Default case
        return self.complete(str(kwargs), **kwargs)

    def complete(self, messages: Union[str, List[Dict]], **kwargs) -> Any:
        """LiteLLM-compatible completion method."""
        try:
            # Parse messages using base class method
            prompt, system_prompt = self._parse_messages(messages)

            # Get parameters (allow override)
            model = kwargs.get("model", self.model)
            temperature = kwargs.get("temperature", self.temperature)
            max_tokens = kwargs.get("max_tokens", self.max_tokens)

            # Generate response using Ollama API
            response_text = self.call_api(
                prompt=prompt,
                system_prompt=system_prompt,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            # Create and return LiteLLM-compatible response
            return self._create_response(response_text, prompt)

        except Exception as e:
            logger.error(f"Completion failed: {e}")
            return self._create_error_response(str(e))

    def call_api(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
    ) -> str:
        """
        Direct Ollama API call - returns just the text content.
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            options = {
                "temperature": temperature or self.temperature,
                "num_predict": max_tokens or self.max_tokens,
            }
            if stop:
                options["stop"] = stop

            response = self.client.chat(
                model=model or self.model,
                messages=messages,
                options=options,
            )

            # Store token usage data if available
            self.last_usage = None
            if hasattr(response, "prompt_eval_count") or hasattr(
                response, "eval_count"
            ):
                # FIX: Default to 0 if attributes are None to prevent TypeError
                prompt_tokens = getattr(response, "prompt_eval_count") or 0
                completion_tokens = getattr(response, "eval_count") or 0

                self.last_usage = {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                }
                # Update total usage stats
                self._update_total_usage(self.last_usage)

            content = (
                response.message.content if response.message.content is not None else ""
            )

            # Clean up thinking tags
            # Method 1: Remove closed thinking tags
            # 1. Clean up closed thinking tags (ROBUST)
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)

            # 2. Clean up unclosed tags at start/end (HEURISTIC, but simple)
            content = re.sub(r"^\s*<think>", "", content, flags=re.IGNORECASE)
            content = re.sub(r"</think>\s*$", "", content, flags=re.IGNORECASE)

            # 3. Aggressively remove any leading lines that are likely garbage/thinking
            lines = content.split("\n")
            content_lines = [
                line
                for line in lines
                if line.strip()
                and len(line.strip()) > 30
                and not line.strip().startswith(("//", "#", "*"))
            ]

            final_content = "\n".join(content_lines) if content_lines else content
            return final_content.strip() if isinstance(final_content, str) else ""

        except Exception as e:
            logger.error(f"Ollama API call failed: {e}")
            raise RuntimeError(f"Ollama API error: {e}")

    def call_structured_api(
        self,
        prompt: str,
        output_schema: Type[T],
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
    ) -> T:
        """
        Calls the Ollama API, forcing the output to be a valid JSON object
        that conforms to the provided Pydantic schema, and returns the
        validated Pydantic object.
        """
        messages = []

        # 1. Prepare system prompt and instructions for JSON output
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        allowed_keys = self._allowed_keys_from_schema(output_schema)
        guard = (
            "Return ONLY a single JSON object with these keys and no others:\n"
            + ", ".join(f'"{k}"' for k in allowed_keys)
        )
        messages.append({"role": "system", "content": guard})

        # Add instruction for the model to follow the schema
        schema_instruction = (
            "You must output a single JSON object that strictly adheres to the "
            "following schema. Do not include any other text, reasoning, or markdown "
            "outside the JSON object."
        )

        # 2. Add the user prompt and the structure requirement
        messages.append(
            {"role": "user", "content": f"{prompt}\n\n{schema_instruction}"}
        )

        try:
            # 3. Get parameters
            options = {
                "temperature": temperature or self.temperature,
                "num_predict": max_tokens or self.max_tokens,
                # CRITICAL: Set context window large enough for prompt + completion
                # Otherwise the model will truncate output when context fills up
                "num_ctx": 16384,  # Increased context window to prevent truncation
            }
            if stop:
                options["stop"] = stop

            logger.debug(
                f"Calling Ollama with num_predict={options['num_predict']}, temperature={options['temperature']}, num_ctx={options['num_ctx']}"
            )

            # 4. Call Ollama API with JSON format
            response = self.client.chat(
                model=model or self.model,
                messages=messages,
                options=options,
                # CRITICAL: This forces Ollama to output JSON
                format="json",
                # Although not directly supported by all Ollama models, passing
                # the schema in the prompt/context is the best practice.
            )

            # Store token usage data if available
            self.last_usage = None
            if hasattr(response, "prompt_eval_count") or hasattr(
                response, "eval_count"
            ):
                # FIX: Default to 0 if attributes are None to prevent TypeError
                prompt_tokens = getattr(response, "prompt_eval_count") or 0
                completion_tokens = getattr(response, "eval_count") or 0

                self.last_usage = {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                }
                # Update total usage stats
                self._update_total_usage(self.last_usage)

                logger.debug(
                    f"Response tokens: prompt={prompt_tokens}, completion={completion_tokens}, requested_max={options['num_predict']}"
                )

            raw_json_string = (
                response.message.content.strip()
                if response.message.content is not None
                else ""
            )

            # 5. Normalize and validate the JSON against the Pydantic model
            # First try direct validation (fast path)
            try:
                validated_model = output_schema.model_validate_json(raw_json_string)
                return validated_model
            except ValidationError as direct_error:
                # Direct validation failed - try normalization
                logger.debug("Direct validation failed, attempting normalization")

                normalized_data = normalize_llm_json(raw_json_string)

                if normalized_data is None:
                    # JSON extraction completely failed (likely truncated)
                    # Log the first 500 and last 200 chars to help debug
                    logger.error(
                        f"Failed to extract valid JSON from LLM response. "
                        f"Response length: {len(raw_json_string)} chars. "
                        f"Error: {direct_error}"
                    )
                    logger.error(
                        f"Response preview (first 500 chars): {raw_json_string[:500]}"
                    )
                    logger.error(
                        f"Response end (last 200 chars): {raw_json_string[-200:]}"
                    )
                    raise direct_error

                # Try validating the normalized data
                try:
                    validated_model = output_schema.model_validate(normalized_data)
                    logger.debug("Validation succeeded after normalization")
                    return validated_model
                except ValidationError as norm_error:
                    logger.error(
                        f"Validation failed even after normalization. "
                        f"Normalized data: {normalized_data}. "
                        f"Error: {norm_error}"
                    )
                    raise norm_error

        except Exception as e:
            logger.error(f"Ollama Structured API call failed: {e}", exc_info=True)
            raise RuntimeError(f"Ollama Structured API error: {e}")

    def list_available_models(self) -> List[str]:
        """List available models on the Ollama server."""
        if self._available_models is None:
            try:
                response = self.client.list()
                self._available_models = [
                    model.model for model in response.models if model.model is not None
                ]
                logger.info(f"Available Ollama models: {self._available_models}")
            except Exception as e:
                logger.error(f"Failed to list models: {e}")
                self._available_models = []
        # Ensure return type is List[str]
        return [m for m in self._available_models if isinstance(m, str)]

    def is_available(self) -> bool:
        """Check if Ollama server is accessible."""
        try:
            self.list_available_models()
            if self._available_models is None:
                return False
            return len(self._available_models) > 0
        except Exception:
            return False

    def _allowed_keys_from_schema(self, schema: Type[BaseModel]) -> List[str]:
        # Pydantic v2: model_fields; v1: __fields__
        if hasattr(schema, "model_fields"):
            return list(schema.model_fields.keys())  # type: ignore[attr-defined]
        return list(schema.__fields__.keys())  # type: ignore[attr-defined]
