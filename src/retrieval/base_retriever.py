"""
Base retriever interface for unified retrieval architecture.
Eliminates inconsistent interfaces across different retrieval sources.
"""

from abc import ABC, abstractmethod

from typing import List, Union


class BaseRetriever(ABC):
    """
    Abstract base class for all retrieval systems.
    Defines the standard interface that all retrievers must implement.
    """

    @abstractmethod
    def search(
        self,
        queries: Union[str, List[str]],
        max_results: int = 8,
        format_type: str = "rag",
        topic: str = None,
        **kwargs
    ) -> List[str]:
        """
        Search for content using the provided queries.

        Args:
            queries: Single query string or list of query strings
            max_results: Maximum number of results to return
            format_type: Output format ("rag" for passages, "storm" for structured data)
            **kwargs: Additional retrieval parameters

        Returns:
            List of passages (format_type="rag") or structured results (format_type="storm")
        """

    def get_source_name(self) -> str:
        """Return the name of this retrieval source."""
        return self.__class__.__name__
