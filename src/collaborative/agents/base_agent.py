# src/agents/base_agent.py
from abc import ABC, abstractmethod

import logging
from typing import Any

from src.config.config_context import ConfigContext

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Abstract base class for all agents in the system."""

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

        # Use OllamaClient with model config if provided
        self.api_client = ConfigContext.get_client("writing")

    @abstractmethod
    def process(self, input_data: Any) -> Any:
        """Main processing method that each agent must implement."""

    def get_task_client(self, task: str):
        """Get client configured for specific task."""
        return ConfigContext.get_client(task)
