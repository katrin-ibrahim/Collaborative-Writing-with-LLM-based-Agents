"""
Simplified Ollama engine that delegates to OllamaClient.
"""

import logging
from typing import List, Optional

from src.baselines.base_engine import BaseModelEngine
from src.config.baselines_model_config import ModelConfig
from src.utils.ollama_client import OllamaClient

logger = logging.getLogger(__name__)


class OllamaModelEngine(BaseModelEngine):
    """
    Simplified Ollama engine that delegates to OllamaClient.
    Eliminates duplicate LiteLLM compatibility logic.
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

        # Get model name from path for Ollama (which uses different naming)
        model_name = self.config.ollama_model_mapping.get(
            self.model_path, self.model_path
        )

        # Create client with all the configuration
        self.client = OllamaClient(
            host=ollama_host,
            model=model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        logger.info(f"OllamaModelEngine initialized with model: {model_name}")

    def __call__(self, *args, **kwargs):
        """Delegate to unified client."""
        return self.client(*args, **kwargs)

    def list_available_models(self) -> List[str]:
        """List available models in Ollama."""
        return self.client.list_models()

    def complete(self, messages, **kwargs):
        """Complete messages and return response object."""
        return self.client.complete(messages, **kwargs)
