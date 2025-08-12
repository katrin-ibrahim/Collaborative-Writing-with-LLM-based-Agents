# src/agents/base_agent.py
from abc import ABC, abstractmethod

import logging
from typing import Any, Dict

from src.utils.ollama_client import OllamaClient

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Abstract base class for all agents in the system."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # Use OllamaClient for now (can be extended later)
        self.api_client = OllamaClient()

    @abstractmethod
    def process(self, input_data: Any) -> Any:
        """Main processing method that each agent must implement."""
