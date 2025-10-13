# src/methods/base_method.py
"""
Abstract base class for all methods in the AI Writer Agent Framework.
"""

from abc import ABC, abstractmethod

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

    def get_method_name(self) -> str:
        """Get the method name for logging and metadata."""
        return self.__class__.__name__.replace("Method", "").lower()
