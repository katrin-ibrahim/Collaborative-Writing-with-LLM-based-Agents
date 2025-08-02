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

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the retrieval service is available."""

    def get_source_name(self) -> str:
        """Return the name of this retrieval source."""
        return self.__class__.__name__

    def clean_query(self, query: str) -> str:
        """
        Clean and normalize search queries from any source.
        Single point of truth for all query cleaning logic.
        """
        if not query:
            return ""

        cleaned = query.strip()
        if not cleaned:
            return ""

        import re

        # Remove numbering and bullets (from LLM output)
        cleaned = re.sub(r"^\d+[\.\)\-\:]\s*", "", cleaned)
        cleaned = re.sub(r"^[-•*→]\s*", "", cleaned)

        # Remove system prefixes (from various sources)
        cleaned = re.sub(
            r"^(?:Query|Search|Find)\s*\d*\s*:\s*", "", cleaned, flags=re.IGNORECASE
        )

        # Skip category headers (anything ending with ":")
        if cleaned.endswith(":"):
            return ""

        # Remove quotes and extra punctuation
        cleaned = re.sub(r'^["\'](.+)["\']$', r"\1", cleaned)
        cleaned = cleaned.strip(".,!?")

        # Normalize whitespace
        cleaned = " ".join(cleaned.split())

        # Skip instruction artifacts and meta-text
        skip_patterns = [
            r"^(example|format|now generate)",
            r"(search terms|wikipedia|one per line)",
            r"^(note|important|critical)",
            r"(historical background and origins|key people.*involved)",  # Common LLM headers
        ]

        for pattern in skip_patterns:
            if re.search(pattern, cleaned.lower()):
                return ""

        # Only keep substantial queries
        if len(cleaned) < 3 or len(cleaned.split()) < 2:
            return ""

        # Filter out overly generic terms
        generic_terms = {
            "query",
            "search",
            "information",
            "details",
            "about",
            "overview",
            "background",
        }
        if cleaned.lower() in generic_terms:
            return ""

        return cleaned
