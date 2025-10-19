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
from utils.setup_storm import (
    get_storm_config_params,
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

        try:
            start_time = time.time()

            # Setup STORM configuration using ConfigContext
            lm_config = setup_storm_config()
            retrieval_manager = setup_storm_retrieval()
            storm_params = get_storm_config_params()

            # Create temporary output directory
            with tempfile.TemporaryDirectory() as temp_dir:
                storm_output_dir = Path(temp_dir) / "storm_output"
                storm_output_dir.mkdir(parents=True, exist_ok=True)

                # Setup STORM runner arguments
                engine_args = STORMWikiRunnerArguments(
                    output_dir=str(storm_output_dir),
                    max_conv_turn=storm_params["max_conv_turn"],
                    max_perspective=storm_params["max_perspective"],
                    search_top_k=storm_params["search_top_k"],
                    max_thread_num=storm_params["max_thread_num"],
                )

                # Create STORM runner
                storm_runner = STORMWikiRunner(
                    engine_args, lm_config, retrieval_manager
                )

                # Run STORM
                storm_runner.run(
                    topic=topic,
                    do_research=True,
                    do_generate_outline=True,
                    do_generate_article=True,
                    do_polish_article=True,
                )

                # Extract STORM output
                from src.utils.article import extract_storm_output

                content = extract_storm_output(storm_output_dir, topic)

            generation_time = time.time() - start_time
            content_words = len(content.split()) if content else 0

            # Get model info from ConfigContext
            writing_engine = ConfigContext.get_client("writing")

            article = Article(
                title=topic,
                content=content,
                sections={},
                metadata={
                    "method": "storm",
                    "generation_time": generation_time,
                    "word_count": content_words,
                    "storm_config": storm_params,
                    "model": writing_engine.model,
                    "temperature": writing_engine.temperature,
                },
            )

            logger.info(
                f"STORM method completed for {topic} ({content_words} words, {generation_time:.2f}s)"
            )
            return article

        except Exception as e:
            logger.error(f"STORM method failed for '{topic}': {e}")
            return Article(
                title=topic,
                content=f"Error generating article with STORM method: {str(e)}",
                sections={},
                metadata={
                    "method": "storm",
                    "error": str(e),
                    "generation_time": (
                        time.time() - start_time if "start_time" in locals() else 0
                    ),
                    "word_count": 0,
                },
            )
