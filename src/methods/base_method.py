# src/methods/base_method.py
"""
Abstract base class for all methods in the AI Writer Agent Framework.
"""

from abc import ABC, abstractmethod

from typing import List

from src.utils.data import Article


class BaseMethod(ABC):
    """
    Abstract base class for all writing methods.

    Methods include: direct, rag, storm, writer, writer_reviewer, etc.
    Each method implements its own approach to generating articles.
    """

    @abstractmethod
    def run(self, topic: str) -> Article:
        """
        Run the method on a single topic.

        Args:
            topic: The topic to generate an article about

        Returns:
            Generated article
        """

    def run_batch(self, topics: List[str]) -> List[Article]:
        """
        Run the method on multiple topics.

        Default implementation runs sequentially. Can be overridden
        for parallel processing in subclasses.

        Args:
            topics: List of topics to generate articles about

        Returns:
            List of generated articles
        """
        return [self.run(topic) for topic in topics]

    def get_method_name(self) -> str:
        """Get the method name for logging and metadata."""
        return self.__class__.__name__.replace("Method", "").lower()
