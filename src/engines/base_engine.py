"""
Unified base engine interface for standardization across different backends.
"""

import time
from abc import ABC, abstractmethod

import logging
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class BaseEngine(ABC):
    """
    Abstract base class defining the unified interface for model engines.
    All configuration is handled by ConfigContext - engines just handle inference.
    """

    def __init__(
        self,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        task: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize engine with minimal parameters from ConfigContext.

        Args:
            model: Model name/path
            temperature: Generation temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Backend-specific parameters (host, device, etc.)
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        # Store optional task label for logging/instrumentation
        self.task = task

        # Store backend-specific kwargs for child classes
        self.backend_kwargs = kwargs

        # LiteLLM compatibility attributes
        self.model_name = model
        self.kwargs = {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        # Token usage tracking
        self.last_usage = None
        self.total_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "calls": 0,
        }

        logger.info(
            f"{self.__class__.__name__} initialized with model: {model} for task: {task}"
        )

    @abstractmethod
    def __call__(self, messages=None, **kwargs):
        """
        Make engine callable for compatibility with STORM and other frameworks.

        Args:
            messages: Optional messages format input
            **kwargs: Additional parameters

        Returns:
            Generation results (LiteLLM-compatible response object)
        """

    @abstractmethod
    def complete(self, messages: Union[str, List[Dict]], **kwargs) -> Any:
        """
        Complete messages and return response object.

        Args:
            messages: Chat messages to complete (string or list of dicts)
            **kwargs: Additional parameters

        Returns:
            LiteLLM-compatible response object
        """

    @abstractmethod
    def list_available_models(self) -> List[str]:
        """
        List available models for the engine.

        Returns:
            List of available model names
        """

    def _create_response(self, content: str, prompt: str) -> Any:
        """
        Create LiteLLM-compatible response object.
        Standardized across all engines for consistency.

        Args:
            content: Generated content
            prompt: Original prompt (for token counting)

        Returns:
            LiteLLM-compatible response object
        """

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
                self.id = f"engine-{int(time.time())}"
                self.object = "chat.completion"
                self.created = int(time.time())

                # Dict-like interface for compatibility
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

    def _parse_messages(self, messages: Union[str, List[Dict]]) -> tuple:
        """
        Parse various message formats into prompt and system_prompt.
        Standardized across all engines.

        Args:
            messages: Input messages (string or chat format)

        Returns:
            tuple: (prompt, system_prompt)
        """
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

    def _create_error_response(self, error_msg: str) -> Any:
        """Create error response in LiteLLM format."""

        class ErrorResponse:
            def __init__(self, error_msg, model_name):
                self.choices = []
                self.model = model_name
                self.error = error_msg
                self.usage = None
                self.id = f"error-{int(time.time())}"
                self.object = "error"
                self.created = int(time.time())

        return ErrorResponse(error_msg, self.model_name)

    def get_usage_stats(self) -> dict:
        """Get current token usage statistics."""
        return {"last_usage": self.last_usage, "total_usage": self.total_usage.copy()}

    def reset_usage_stats(self) -> dict:
        """Reset and return the current usage stats."""
        stats = self.get_usage_stats()
        self.total_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "calls": 0,
        }
        self.last_usage = None
        return stats

    def _update_total_usage(self, usage_data: dict):
        """Update total usage with data from last call."""
        if usage_data:
            self.total_usage["prompt_tokens"] += usage_data.get("prompt_tokens", 0)
            self.total_usage["completion_tokens"] += usage_data.get(
                "completion_tokens", 0
            )
            self.total_usage["total_tokens"] += usage_data.get("total_tokens", 0)
            self.total_usage["calls"] += 1
