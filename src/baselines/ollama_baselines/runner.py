"""
Refactored Ollama baseline runner implementation.
Uses standard BaseRunner interface for consistency.
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import logging
from typing import List, Optional

from src.baselines.model_engines.ollama_engine import OllamaModelEngine
from src.baselines.rag_runner import run_rag
from src.baselines.runner_base import BaseRunner
from src.config.baselines_model_config import ModelConfig
from src.utils.baselines_utils import error_article
from src.utils.data_models import Article
from src.utils.ollama_client import OllamaClient
from src.utils.output_manager import OutputManager

from .configure_storm import setup_storm_runner

logger = logging.getLogger(__name__)


class BaselineRunner(BaseRunner):
    """
    Ollama baseline runner with standardized architecture.
    """

    def __init__(
        self,
        ollama_host: str = "http://10.167.31.201:11434/",
        model_config: Optional[ModelConfig] = None,
        output_manager: Optional[OutputManager] = None,
    ):
        # Initialize base class
        super().__init__(model_config=model_config, output_manager=output_manager)

        # Ensure we're using ollama mode
        self.model_config.mode = "ollama"

        # Store Ollama host
        self.ollama_host = ollama_host

        # Create Ollama client
        self.client = OllamaClient(host=ollama_host)

        # Verify server availability
        if not self.client.is_available():
            raise RuntimeError(f"Ollama server not available at {ollama_host}")

        # Cache for model engines
        self._engine_cache = {}

        # Log available models
        available_models = self.client.list_models()
        logger.info(
            f"Connected to Ollama with {len(available_models)} models available"
        )

    def get_model_engine(self, task: str) -> OllamaModelEngine:
        """
        Get an appropriate OllamaModelEngine for the specified task.
        Uses caching to avoid creating multiple instances for the same task.

        Args:
            task: Task name (e.g. "writing", "critique")

        Returns:
            OllamaModelEngine instance
        """
        # Return cached engine if it exists
        if task in self._engine_cache:
            return self._engine_cache[task]

        # Create a new engine
        model_path = self.model_config.get_model_path(task)

        engine = OllamaModelEngine(
            ollama_host=self.ollama_host,
            model_path=model_path,
            config=self.model_config,
            task=task,
        )

        # Cache the engine
        self._engine_cache[task] = engine

        return engine

    # run_direct_prompting is implemented in the BaseRunner class

    def run_storm(self, topic: str) -> Article:
        """
        Run STORM using the Ollama backend.

        Args:
            topic: The topic to generate an article about

        Returns:
            Generated Article object
        """
        logger.info(f"Running STORM for: {topic}")

        try:
            start_time = time.time()

            # Setup STORM runner with output going to the proper output directory
            storm_output_base = None
            if self.output_manager:
                # Use the output manager's dedicated method for STORM output
                storm_output_base = self.output_manager.setup_storm_output_dir(topic)

            storm_runner, storm_output_dir = setup_storm_runner(
                client=self.client,
                config=self.model_config,
                storm_output_dir=storm_output_base,
            )

            # Run STORM
            logger.info(
                f"Running STORM on topic: {topic} with output dir: {storm_output_dir}"
            )
            # The run method doesn't return the output dir, it's actually passed when creating the runner
            storm_runner.run(topic)
            logger.info(
                f"STORM run completed. Using output directory: {storm_output_dir}"
            )

            # Extract content and metadata
            from .runner_utils import extract_storm_output_ollama

            logger.info(f"Extracting STORM output from: {storm_output_dir}")
            content, storm_metadata = extract_storm_output_ollama(
                topic, storm_output_dir
            )
            logger.info(
                f"Extracted content length: {len(content)}, metadata: {storm_metadata}"
            )

            generation_time = time.time() - start_time
            content_words = len(content.split()) if content else 0

            # Create article - topics should already be strings now
            article = Article(
                title=topic,
                content=content,
                sections={},
                metadata={
                    "method": "storm",
                    "model": self.model_config.get_model_for_task("writing"),
                    "word_count": content_words,
                    "generation_time": generation_time,
                    "storm_config": storm_metadata,
                },
            )

            # Save article
            if self.output_manager:
                self.output_manager.save_article(article, "storm")

            logger.info(
                f"STORM completed for {topic} ({content_words} words, {generation_time:.2f}s)"
            )
            return article

        except Exception as e:
            logger.error(f"STORM failed for '{topic}': {e}")
            # Topics should already be strings now
            return error_article(topic, str(e), "storm")

    def run_rag(self, topic: str) -> Article:
        """
        Run RAG using the Ollama backend.

        Args:
            topic: The topic to generate an article about

        Returns:
            Generated Article object
        """
        # Get the writing engine
        engine = self.get_model_engine("writing")

        try:
            # Initialize Wikipedia search retrieval module
            from .wikipedia_rm import WikipediaSearchRM

            retrieval_system = WikipediaSearchRM(
                k=3
            )  # Default to retrieving top 3 results

            # Generate search queries function
            def generate_queries_with_ollama(engine, topic, num_queries=5):
                from .runner_utils import generate_search_queries

                return generate_search_queries(
                    self.client, self.model_config, topic, num_queries=num_queries
                )

            # Run the unified RAG implementation with ollama-specific query generation
            article = run_rag(
                engine,
                topic,
                context_retriever=retrieval_system,
                generate_queries_func=generate_queries_with_ollama,
            )

            # Save article if output manager available
            if article and self.output_manager:
                self.output_manager.save_article(article, "rag")

            return article

        except Exception as e:
            logger.error(f"RAG failed for '{topic}': {e}")
            return error_article(topic, str(e), "rag")

    # ---------------------------------------- Batch Processing ----------------------------------------
    def run_direct_batch(
        self, topics: List[str], max_workers: int = 2
    ) -> List[Article]:
        """Run direct prompting in parallel for multiple topics."""
        max_workers = min(max_workers, 2)  # Prevent overwhelming the model

        remaining_topics = self.filter_completed_topics(topics, "direct")
        if not remaining_topics:
            logger.info("All direct prompting topics already completed")
            return []

        logger.info(
            f"Running Direct Prompting batch for {len(remaining_topics)} topics"
        )
        results = []

        def run_topic(topic):
            try:
                if self.state_manager:
                    self.state_manager.mark_topic_in_progress(topic, "direct")

                article = self.run_direct_prompting(topic)

                if self.state_manager:
                    self.state_manager.mark_topic_completed(topic, "direct")

                return article

            except Exception as e:
                if self.state_manager:
                    self.state_manager.cleanup_in_progress_topic(topic, "direct")
                logger.error(f"Direct batch failed for {topic}: {e}")
                return error_article(topic, str(e), "direct_batch")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(run_topic, topic): topic for topic in remaining_topics
            }
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as e:
                    topic = futures[future]
                    logger.error(f"Direct batch failed for {topic}: {e}")
                    results.append(error_article(topic, str(e), "direct_batch"))

        return results

    def run_storm_batch(self, topics: List[str], max_workers: int = 1) -> List[Article]:
        """Run STORM in parallel for multiple topics."""
        remaining_topics = self.filter_completed_topics(topics, "storm")
        if not remaining_topics:
            logger.info("All STORM topics already completed")
            return []

        logger.info(f"Running STORM batch for {len(remaining_topics)} topics")
        results = []

        def run_topic(topic):
            try:
                if self.state_manager:
                    self.state_manager.mark_topic_in_progress(topic, "storm")

                article = self.run_storm(topic)

                if self.state_manager:
                    self.state_manager.mark_topic_completed(topic, "storm")

                return article

            except Exception as e:
                if self.state_manager:
                    self.state_manager.cleanup_in_progress_topic(topic, "storm")
                logger.error(f"STORM batch failed for {topic}: {e}")
                return error_article(topic, str(e), "storm_batch")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(run_topic, topic): topic for topic in remaining_topics
            }
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as e:
                    topic = futures[future]
                    logger.error(f"STORM batch failed for {topic}: {e}")
                    results.append(error_article(topic, str(e), "storm_batch"))

        return results

    def run_rag_batch(self, topics: List[str], max_workers: int = 2) -> List[Article]:
        """Run RAG in parallel for multiple topics."""
        remaining_topics = self.filter_completed_topics(topics, "rag")
        if not remaining_topics:
            logger.info("All RAG topics already completed")
            return []

        logger.info(f"Running RAG batch for {len(remaining_topics)} topics")
        results = []

        def run_topic(topic):
            try:
                if self.state_manager:
                    self.state_manager.mark_topic_in_progress(topic, "rag")

                article = self.run_rag(topic)

                if self.state_manager:
                    self.state_manager.mark_topic_completed(topic, "rag")

                return article

            except Exception as e:
                if self.state_manager:
                    self.state_manager.cleanup_in_progress_topic(topic, "rag")
                logger.error(f"RAG batch failed for {topic}: {e}")
                return error_article(topic, str(e), "rag_batch")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(run_topic, topic): topic for topic in remaining_topics
            }
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as e:
                    topic = futures[future]
                    logger.error(f"RAG batch failed for {topic}: {e}")
                    results.append(error_article(topic, str(e), "rag_batch"))

        return results
