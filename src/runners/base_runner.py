"""
BaseRunner with factory-based orchestration for collaborative methods.
Removes direct method implementations, uses method factory pattern.
"""

import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed

import logging
from typing import List, Optional

from src.config.baselines_model_config import ModelConfig
from src.config.collaboration_config import CollaborationConfig
from src.config.retrieval_config import DEFAULT_RETRIEVAL_CONFIG, RetrievalConfig
from src.methods.factory import create_method, get_supported_methods
from src.utils.article import error_article
from src.utils.data import Article
from src.utils.io import OutputManager

logger = logging.getLogger(__name__)


class BaseRunner(ABC):
    """
    Base runner using factory pattern for all methods.
    Concrete runners only handle client/engine initialization.
    """

    def __init__(
        self,
        model_config: Optional[ModelConfig] = None,
        output_manager: Optional[OutputManager] = None,
        retrieval_config: Optional[RetrievalConfig] = None,
        collaboration_config: Optional[CollaborationConfig] = None,
    ):
        self.model_config = model_config or ModelConfig()
        self.output_manager = output_manager
        self.retrieval_config = retrieval_config or DEFAULT_RETRIEVAL_CONFIG
        self.collaboration_config = collaboration_config or CollaborationConfig()
        self.state_manager = None

    @abstractmethod
    def get_client(self):
        """Get the appropriate client for method initialization."""

    def get_supported_methods(self) -> List[str]:
        """Return list of methods supported by this runner."""
        return get_supported_methods()

    def set_state_manager(self, state_manager):
        """Set the state manager for checkpoint handling."""
        self.state_manager = state_manager

    def filter_completed_topics(self, topics: List[str], method: str) -> List[str]:
        """Filter out completed topics for the specified method."""
        if not self.state_manager:
            return topics

        return [
            topic
            for topic in topics
            if not self.state_manager.is_complete(topic, method)
        ]

    def run_single_topic(self, topic: str, method: str) -> Article:
        """Run a single method on a topic using factory pattern."""
        logger.info(f"Running {method} for topic: {topic}")

        start_time = time.time()

        try:
            # Create method configuration by merging all configs
            method_config = {
                **self.retrieval_config.__dict__,
                **self.collaboration_config.to_dict(),
            }

            # Get client for this method
            client = self.get_client()

            # Create method instance using factory
            method_instance = create_method(method, client, method_config)

            # Run the method
            article = method_instance.run(topic)

            # Add timing metadata
            execution_time = time.time() - start_time
            article.metadata.update(
                {
                    "execution_time": execution_time,
                    "runner": self.__class__.__name__,
                }
            )

            logger.info(f"✓ {method} completed for {topic} in {execution_time:.2f}s")
            return article

        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(
                f"✗ {method} failed for {topic} after {execution_time:.2f}s: {e}"
            )
            return error_article(topic, str(e), method)

    def run_topics(
        self, topics: List[str], method: str, max_workers: int = None
    ) -> List[Article]:
        """Run a method on multiple topics in parallel."""
        if max_workers is None:
            max_workers = 1 if len(topics) < 5 else 2  # Conservative parallelism

        if max_workers == 1:
            # Sequential execution
            return [self.run_single_topic(topic, method) for topic in topics]

        # Parallel execution
        logger.info(
            f"Running {method} on {len(topics)} topics with {max_workers} workers"
        )

        results = [None] * len(topics)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_index = {
                executor.submit(self.run_method, topic, method): i
                for i, topic in enumerate(topics)
            }

            # Collect results
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    results[index] = future.result()
                except Exception as e:
                    topic = topics[index]
                    logger.error(f"Failed to get result for {topic}: {e}")
                    results[index] = error_article(topic, str(e), method)

        return results

    def run(self, topics: List[str], methods: List[str]) -> List[Article]:
        """Run multiple topics and methods - main orchestration."""
        results = []
        supported_methods = self.get_supported_methods()

        for method in methods:
            if method not in supported_methods:
                logger.warning(
                    f"Method '{method}' not supported by this runner. "
                    f"Supported: {supported_methods}"
                )
                continue

            # Filter completed topics if state manager available
            method_topics = self.filter_completed_topics(topics, method)

            if not method_topics:
                logger.info(f"All topics already completed for method '{method}'")
                continue

            logger.info(f"Running {method} on {len(method_topics)} topics")

            # Run method on all topics
            method_results = self.run_topics(method_topics, method)
            results.extend(method_results)

            # Save progress if state manager available
            if self.state_manager:
                for topic, article in zip(method_topics, method_results):
                    if "error" not in article.metadata:
                        self.state_manager.mark_complete(topic, method)
                        if self.output_manager:
                            self.output_manager.save_article(article, method)

        return results
