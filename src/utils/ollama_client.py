"""
Unified Ollama client with both simple and LiteLLM-compatible interfaces.
Replaces both OllamaClient and OllamaLiteLLMWrapper for simplicity.
"""

import time

import json
import logging
import re
from ollama import Client
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class OllamaClient:
    """
    Ollama client that provides both simple and LiteLLM-compatible interfaces.
    Can be used directly for simple calls or as a LiteLLM-compatible wrapper for STORM.
    """

    def __init__(
        self,
        host: str = "http://localhost:11434/",
        model: str = "qwen3:4b",
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ):
        self.host = host
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = Client(host=host)
        self._available_models = None

        # LiteLLM compatibility attributes
        self.model_name = model
        self.kwargs = {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        logger.info(f"OllamaClient initialized with host: {host}, model: {model}")

    # ========== Simple Interface (OllamaClient compatibility) ==========

    def call_api(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs,
    ) -> str:
        """
        Simple API call interface - returns just the text content.
        Compatible with the original OllamaClient interface.
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = self.client.chat(
                model=model or self.model,
                messages=messages,
            )

            content = response.message.content
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
            return content.strip()

        except Exception as e:
            logger.error(f"Ollama API call failed: {e}")
            raise RuntimeError(f"Ollama API error: {e}")

    def list_models(self) -> List[str]:
        """List available models on the Ollama server."""
        if self._available_models is None:
            try:
                response = self.client.list()
                self._available_models = [model.model for model in response.models]
                logger.info(f"Available Ollama models: {self._available_models}")
            except Exception as e:
                logger.error(f"Failed to list models: {e}")
                self._available_models = []
        return self._available_models

    def is_available(self) -> bool:
        """Check if Ollama server is accessible."""
        try:
            self.list_models()
            return len(self._available_models) > 0
        except:
            return False

    def extract_json(self, text: str) -> Optional[Dict]:
        """Extract JSON from text (compatibility method)."""
        json_pattern = r"({[\s\S]*})"
        match = re.search(json_pattern, text)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
        return None

    # ========== LiteLLM Interface (OllamaLiteLLMWrapper compatibility) ==========

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
            # Parse messages
            prompt, system_prompt = self._parse_messages(messages)

            # Get parameters
            model = kwargs.get("model", self.model)
            temperature = kwargs.get("temperature", self.temperature)
            max_tokens = kwargs.get("max_tokens", self.max_tokens)

            # Generate response using simple interface
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

    def _parse_messages(self, messages: Union[str, List[Dict]]) -> tuple:
        """Parse various message formats."""
        if isinstance(messages, str):
            return messages, None

        if not isinstance(messages, list):
            return str(messages), None

        prompt_parts = []
        system_prompt = None

        for message in messages:
            if isinstance(message, dict):
                role = message.get("role", "")
                content = message.get("content", "")

                if role == "system":
                    system_prompt = content
                elif role in ["user", "assistant"]:
                    prompt_parts.append(content)
            else:
                prompt_parts.append(str(message))

        return "\n".join(prompt_parts), system_prompt

    def _create_response(self, content: str, prompt: str) -> Any:
        """Create LiteLLM-compatible response object."""

        class Response:
            def __init__(self, content, model):
                self.content = content
                self.choices = [
                    type(
                        "Choice",
                        (),
                        {
                            "message": type(
                                "Message", (), {"content": content, "role": "assistant"}
                            )(),
                            "finish_reason": "stop",
                            "index": 0,
                        },
                    )()
                ]
                self.model = model
                self.usage = type(
                    "Usage",
                    (),
                    {
                        "prompt_tokens": len(prompt.split()),
                        "completion_tokens": len(content.split()),
                        "total_tokens": len(prompt.split()) + len(content.split()),
                    },
                )()
                self.id = f"ollama-{int(time.time())}"
                self.object = "chat.completion"
                self.created = int(time.time())

                # Dict-like interface
                self._data = {
                    "choices": self.choices,
                    "model": self.model,
                    "usage": self.usage,
                    "id": self.id,
                    "object": self.object,
                    "created": self.created,
                }

            def __getitem__(self, key):
                if isinstance(key, int):
                    return self._data["choices"][key].message.content
                else:
                    return self._data[key]

            def get(self, key, default=None):
                return self._data.get(key, default)

            def __str__(self):
                return self.content

        return Response(content, self.model)

    def _create_error_response(self, error_msg: str) -> Any:
        """Create error response in LiteLLM format."""
        return self._create_response(f"Error: {error_msg}", "")


# Backward compatibility alias
OllamaLiteLLMWrapper = OllamaClient
