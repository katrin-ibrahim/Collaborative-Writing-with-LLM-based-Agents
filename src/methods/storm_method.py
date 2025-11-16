"""
STORM method implementation using knowledge_storm library.
"""

import tempfile
import time
from pathlib import Path

import logging
from knowledge_storm import STORMWikiRunner, STORMWikiRunnerArguments

from src.config.config_context import ConfigContext
from src.methods.base_method import BaseMethod
from src.utils.data import Article
from src.utils.setup_storm import (
    setup_storm_config,
    setup_storm_retrieval,
)

logger = logging.getLogger(__name__)


class StormMethod(BaseMethod):
    """
    STORM method using the knowledge_storm library with ConfigContext.

    This method integrates STORM with the current architecture using
    ConfigContext for configuration and OllamaEngine for model calls.
    """

    def __init__(self):
        super().__init__()

    def run(self, topic: str) -> Article:
        """
        Generate article using STORM knowledge_storm library.

        Args:
            topic: Topic to write about

        Returns:
            Generated article with STORM metadata
        """
        logger.info(f"Running STORM method for: {topic}")

        # Reset usage counters at start
        task_models = self._get_task_models_for_method()
        self._reset_all_client_usage(task_models)

        start_time = time.time()
        try:
            # Setup STORM configuration using ConfigContext
            lm_config = setup_storm_config()
            retrieval_manager = setup_storm_retrieval()

            # Create temporary output directory
            with tempfile.TemporaryDirectory() as temp_dir:
                storm_output_dir = Path(temp_dir) / "storm_output"
                storm_output_dir.mkdir(parents=True, exist_ok=True)

                # Setup STORM runner arguments
                engine_args = STORMWikiRunnerArguments(output_dir=str(storm_output_dir))

                # Create STORM runner
                storm_runner = STORMWikiRunner(
                    engine_args, lm_config, retrieval_manager
                )

                # Run STORM
                storm_runner.run(topic=topic)

                # Extract STORM output
                from src.utils.article import extract_storm_output

                content = extract_storm_output(storm_output_dir, topic)

            generation_time = time.time() - start_time
            content_words = len(content.split()) if content else 0

            # Collect token usage statistics
            token_usage = self._collect_token_usage(task_models)

            # Get model info from ConfigContext
            writing_engine = ConfigContext.get_client("writer")

            article = Article(
                title=topic,
                content=content,
                sections={},
                metadata={
                    "method": "storm",
                    "generation_time": generation_time,
                    "word_count": content_words,
                    "model": writing_engine.model,
                    "temperature": writing_engine.temperature,
                    "token_usage": token_usage,
                },
            )

            logger.info(
                f"STORM method completed for {topic} ({content_words} words, {generation_time:.2f}s, {token_usage['total_tokens']} tokens)"
            )
            return article

        except Exception as e:
            logger.error(f"STORM method failed for '{topic}': {e}")
            generation_time = time.time() - start_time
            return Article(
                title=topic,
                content=f"Error generating article with STORM method: {str(e)}",
                sections={},
                metadata={
                    "method": "storm",
                    "error": str(e),
                    "generation_time": generation_time,
                    "word_count": 0,
                    "token_usage": self._collect_token_usage(task_models),
                },
            )
