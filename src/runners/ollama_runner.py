"""
Ollama runner using factory pattern for collaborative methods.
Handles Ollama client initialization, inherits orchestration from BaseRunner.
"""

import logging
from typing import Optional

from src.config.baselines_model_config import ModelConfig
from src.config.collaboration_config import CollaborationConfig
from src.config.retrieval_config import RetrievalConfig
from src.runners.base_runner import BaseRunner
from src.utils.clients import OllamaClient
from src.utils.io import OutputManager

logger = logging.getLogger(__name__)


class OllamaRunner(BaseRunner):
    """
    Ollama runner for collaborative methods.
    Handles Ollama client initialization, uses factory for all methods.
    """

    def __init__(
        self,
        ollama_host: str = None,
        model_config: Optional[ModelConfig] = None,
        output_manager: Optional[OutputManager] = None,
        retrieval_config: Optional[RetrievalConfig] = None,
        collaboration_config: Optional[CollaborationConfig] = None,
    ):
        # Initialize base class
        super().__init__(
            model_config=model_config,
            output_manager=output_manager,
            retrieval_config=retrieval_config,
            collaboration_config=collaboration_config,
        )

        # Determine ollama_host: model config takes priority over parameter
        if self.model_config and self.model_config.ollama_host:
            self.ollama_host = self.model_config.ollama_host
            logger.info(f"Using ollama_host from model config: {self.ollama_host}")
        elif ollama_host:
            self.ollama_host = ollama_host
            logger.info(f"Using ollama_host from parameter: {self.ollama_host}")
        else:
            self.ollama_host = "http://10.167.31.201:11434/"
            logger.info(f"Using default ollama_host: {self.ollama_host}")

        # Create and verify Ollama client
        self.client = OllamaClient(host=self.ollama_host)
        if not self.client.is_available():
            raise RuntimeError(f"Ollama server not available at {self.ollama_host}")

        # Log available models
        available_models = self.client.list_models()
        logger.info(
            f"Connected to Ollama with {len(available_models)} models available"
        )

        # Log configuration summary
        logger.info(f"OllamaRunner initialized:")
        logger.info(f"  - Host: {self.ollama_host}")
        logger.info(f"  - Model mode: {self.model_config.mode}")
        logger.info(
            f"  - Collaboration max_iterations: {self.collaboration_config.max_iterations}"
        )
        logger.info(f"  - Supported methods: {self.get_supported_methods()}")

    def get_client(self):
        """Get the Ollama client for method initialization."""
        return self.client
