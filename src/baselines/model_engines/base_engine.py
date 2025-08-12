"""
Base model engine interface for standardization across different backends.
"""

from abc import ABC, abstractmethod

import logging
from typing import List, Optional

from src.config.baselines_model_config import ModelConfig

logger = logging.getLogger(__name__)


class BaseModelEngine(ABC):
    """
    Abstract base class defining the interface for model engines.

    Both LocalModelEngine and OllamaLiteLLMWrapper should implement this interface.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        config: Optional[ModelConfig] = None,
        task: str = "writing",
    ):
        self.config = config or ModelConfig()
        self.task = task
        self.model_path = model_path

        if not self.model_path:
            self.model_path = self.config.get_model_path(task)

        # Common parameters all engines should have
        self.temperature = self.config.get_temperature_for_task(task)
        self.max_tokens = self.config.get_token_limit_for_task(task)

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
        Complete messages in chat format.

        Args:
            messages: Chat messages to complete
            **kwargs: Additional parameters

        Returns:
            Completion result
        """

    def list_available_models(self) -> List[str]:
        """
        List available models for the engine.

        Returns:
            List of available model names
        """
        return [self.model_path]
