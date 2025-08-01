"""
Ultra-thin Ollama baseline runner - only responsible for client/engine initialization and STORM.
All shared logic moved to BaseRunner.
"""

import logging
from typing import Optional

from src.baselines.baseline_runner_base import BaseRunner
from src.baselines.model_engines.ollama_engine import OllamaModelEngine
from src.config.baselines_model_config import ModelConfig
from src.utils.baselines_utils import error_article
from src.utils.data_models import Article
from src.utils.ollama_client import OllamaClient
from src.utils.output_manager import OutputManager

# STORM imports - only here to avoid local runner conflicts
from .configure_storm import setup_storm_runner

logger = logging.getLogger(__name__)


class BaselineRunner(BaseRunner):
    """
    Ultra-thin Ollama runner - handles client/engine init + STORM only.
    All other methods (direct, rag, batch) implemented in BaseRunner.
    """

    def __init__(
        self,
        ollama_host: str = "http://10.167.31.201:11434/",
        model_config: Optional[ModelConfig] = None,
        output_manager: Optional[OutputManager] = None,
    ):
        # Initialize base class
        super().__init__(model_config=model_config, output_manager=output_manager)

        # Store Ollama-specific parameters
        self.ollama_host = ollama_host

        # Create and verify Ollama client
        self.client = OllamaClient(host=ollama_host)
        if not self.client.is_available():
            raise RuntimeError(f"Ollama server not available at {ollama_host}")

        # Engine cache
        self._engine_cache = {}

        # Log available models
        available_models = self.client.list_models()
        logger.info(
            f"Connected to Ollama with {len(available_models)} models available"
        )

    def get_model_engine(self, task: str) -> OllamaModelEngine:
        """Get cached OllamaModelEngine for the specified task."""
        if task in self._engine_cache:
            return self._engine_cache[task]

        # Create new engine
        model_path = self.model_config.get_model_path(task)

        engine = OllamaModelEngine(
            ollama_host=self.ollama_host,
            model_path=model_path,
            config=self.model_config,
            task=task,
        )

        # Cache and return
        self._engine_cache[task] = engine
        return engine

    def get_supported_methods(self):
        """Return methods supported by Ollama runner."""
        return ["direct", "storm", "rag"]  # Full method support

    def run_storm(self, topic: str) -> Article:
        """
        Run STORM - only method unique to Ollama runner.
        (direct and rag implemented in BaseRunner)
        """
        logger.info(f"Running STORM for: {topic}")

        try:
            # Setup STORM runner
            storm_runner, storm_output_dir = setup_storm_runner(
                client=self.client,
                config=self.model_config,
                storm_output_dir=(
                    self.output_manager.get_storm_output_dir(topic)
                    if self.output_manager
                    else None
                ),
            )

            # Run STORM
            storm_runner.run(
                topic=topic,
                do_research=True,
                do_generate_outline=True,
                do_generate_article=True,
                do_polish_article=True,
            )

            # Extract STORM output and create Article
            from .runner_utils import extract_storm_output_ollama

            content, storm_metadata = extract_storm_output_ollama(
                topic, storm_output_dir
            )

            # Create Article object
            article = Article(
                title=topic,
                content=content,
                sections={},
                metadata={
                    "method": "storm",
                    "model": self.model_config.get_model_for_task("writing"),
                    "word_count": len(content.split()) if content else 0,
                    "storm_metadata": storm_metadata,
                },
            )

            # Save article if output manager available
            if self.output_manager:
                self.output_manager.save_article(article, "storm")

            logger.info(f"STORM completed for {topic}")
            return article

        except Exception as e:
            logger.error(f"STORM failed for '{topic}': {e}")
            return error_article(topic, str(e), "storm")

    # All other methods (run_direct, run_rag, run_*_batch) implemented in BaseRunner
