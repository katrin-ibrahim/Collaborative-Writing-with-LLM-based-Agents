"""
Base model engine interface for standardization across different backends.
"""

from abc import ABC, abstractmethod

import logging
from typing import List

logger = logging.getLogger(__name__)


from abc import ABC, abstractmethod

from typing import List


class BaseModelEngine(ABC):
    """
    Abstract base class defining the interface for model engines.

    Both LocalModelEngine and OllamaLiteLLMWrapper should implement this interface.
    """

    @abstractmethod
    def __call__(self, messages=None, **kwargs):
        """
        Make engine callable for compatibility with STORM and other frameworks.

        Args:
            messages: Optional messages format input
            **kwargs: Additional parameters

        Returns:
            Generation results
        """

    @abstractmethod
    def complete(self, messages, **kwargs):
        """
        Complete messages and return response object.

        Args:
            messages: Chat messages to complete
            **kwargs: Additional parameters

        Returns:
            Response object - compatible with STORM and LiteLLM
        """

    def extract_content(self, response) -> str:
        """
        Extract string content from a response object.

        Args:
            response: Response object from complete() method

        Returns:
            Extracted string content
        """
        if isinstance(response, str):
            return response
        elif hasattr(response, "content"):
            return response.content
        elif hasattr(response, "choices") and response.choices:
            return response.choices[0].message.content
        else:
            return str(response)

    def list_available_models(self) -> List[str]:
        """
        List available models for the engine.

        Returns:
            List of available model names
        """
        return [self.model_path]
