import time

import logging
from typing import List

from src.config.config_context import ConfigContext
from src.methods.factory import create_method
from src.utils.article import error_article
from src.utils.data import Article

logger = logging.getLogger(__name__)


class Runner:
    def __init__(self, output_manager=None):
        self.output_manager = output_manager
        self.state_manager = None

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

            # Create method instance using factory
            method_instance = create_method(method)

            # Run the method
            article = method_instance.run(topic)

            # Add timing metadata
            execution_time = time.time() - start_time
            article.metadata.update(
                {
                    "execution_time": execution_time,
                    "runner": ConfigContext.get_backend(),
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

    def run(self, topics: List[str], methods: List[str]) -> List[Article]:
        """Run multiple topics and methods - main orchestration."""
        results = []

        for method in methods:
            # Filter completed topics if state manager available
            method_topics = self.filter_completed_topics(topics, method)

            if not method_topics:
                logger.info(f"All topics already completed for method '{method}'")
                continue

            logger.info(f"Running {method} on {len(method_topics)} topics")

            # Run method on all topics
            method_results = [self.run_single_topic(topic, method) for topic in topics]

            results.extend(method_results)

            # Save articles and mark progress
            for topic, article in zip(method_topics, method_results):
                if self.output_manager:
                    self.output_manager.save_article(article, method)

                # Mark complete if state manager available
                if self.state_manager:
                    self.state_manager.mark_complete(topic, method)

        return results
