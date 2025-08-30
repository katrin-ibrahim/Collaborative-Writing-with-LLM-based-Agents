# src/agents/base_agent.py
from abc import ABC, abstractmethod

import logging
from typing import Any

from src.utils.clients import OllamaClient

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Abstract base class for all agents in the system."""

    def __init__(self, retrieval_config, collaboration_config, model_config=None):
        self.retrieval_config = retrieval_config
        self.collaboration_config = collaboration_config
        self.logger = logging.getLogger(self.__class__.__name__)

        # Use OllamaClient with model config if provided
        if model_config:
            from src.config.baselines_model_config import ModelConfig

            if isinstance(model_config, ModelConfig):
                # Use model config for default task parameters
                model = model_config.get_model_for_task("writing")
                temperature = model_config.get_temperature_for_task("writing")
                max_tokens = model_config.get_token_limit_for_task("writing")
                self.api_client = OllamaClient(
                    model=model, temperature=temperature, max_tokens=max_tokens
                )
            else:
                self.api_client = OllamaClient()
        else:
            self.api_client = OllamaClient()

    @abstractmethod
    def process(self, input_data: Any) -> Any:
        """Main processing method that each agent must implement."""
