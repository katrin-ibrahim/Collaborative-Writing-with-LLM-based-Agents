"""
Refactored local baseline runner implementation.
Uses standard BaseRunner interface for consistency with Ollama version.
"""

import logging
from typing import Optional

from baselines.main_runner_base import BaseRunner
from src.baselines.model_engines.local_engine import LocalModelEngine
from src.baselines.rag_runner import run_rag
from src.config.baselines_model_config import ModelConfig
from src.utils.data_models import Article
from src.utils.output_manager import OutputManager

logger = logging.getLogger(__name__)


class LocalBaselineRunner(BaseRunner):
    def __init__(
        self,
        model_config: Optional[ModelConfig] = None,
        output_manager: Optional[OutputManager] = None,
        device: str = "auto",
        model_path: Optional[str] = None,
    ):
        # Initialize base class
        super().__init__(model_config=model_config, output_manager=output_manager)

        # Ensure we're using local mode
        self.model_config.mode = "local"

        # Store device for model initialization
        self.device = device

        # Store model path
        self.model_path = model_path

        # Cache for model engines
        self._engine_cache = {}

        logger.info("LocalBaselineRunner initialized")
        logger.info(f"Using device: {self.device}")

    def get_model_engine(self, task: str) -> LocalModelEngine:
        """
        Get an appropriate LocalModelEngine for the specified task.
        Uses caching to avoid creating multiple instances for the same task.

        Args:
            task: Task name (e.g. "writing", "critique")

        Returns:
            LocalModelEngine instance
        """
        # Return cached engine if it exists
        if task in self._engine_cache:
            return self._engine_cache[task]

        # Create a new engine
        model_path = self.model_path or self.model_config.get_model_path(task)

        engine = LocalModelEngine(
            model_path=model_path,
            device=self.device,
            config=self.model_config,
            task=task,
        )

        # Cache the engine
        self._engine_cache[task] = engine

        return engine

    # run_direct_prompting is implemented in the BaseRunner class

    def run_storm(self, topic: str) -> Article:
        """
        Run STORM using local models.

        Args:
            topic: The topic to generate an article about

        Returns:
            Generated Article object
        """
        # This is a placeholder, STORM requires a separate implementation
        logger.warning(f"STORM is not currently implemented for local models: {topic}")
        return None

    def run_rag(self, topic: str) -> Article:
        """
        Run RAG using local models.

        Args:
            topic: The topic to generate an article about

        Returns:
            Generated Article object
        """
        # Get the writing engine
        engine = self.get_model_engine("writing")

        # Run the unified RAG implementation
        article = run_rag(engine, topic)

        # Save article if output manager available
        if article and self.output_manager:
            self.output_manager.save_article(article, "rag")

        return article
