"""
Unified Retrieval Manager - Single entry point for all retrieval operations.
Eliminates the wrapper hell and provides consistent interface for RAG and STORM.
"""

import logging
from typing import Dict, List, Union

from src.retrieval.base_retriever import BaseRetriever

logger = logging.getLogger(__name__)


class RM:
    """
    Unified Retrieval Manager that provides a single, clean interface
    for all retrieval operations across different backends and formats.

    Replaces the complex WikipediaSearchRM hierarchy with a simple,
    consistent interface for both RAG and STORM workflows.
    """

    def __init__(self, retriever: BaseRetriever, cache_results: bool = True):
        """
        Initialize the unified retrieval manager.

        Args:
            retriever: The base retriever implementation to use
            cache_results: Whether to cache retrieval results
        """
        self.retriever = retriever
        self.cache_results = cache_results
        self._result_cache = {} if cache_results else None

        # Verify retriever is available - fail clearly if not
        if not self.retriever.is_available():
            raise RuntimeError(
                f"Retriever {self.retriever.get_source_name()} is not available. Check your connection and dependencies."
            )

        logger.info(f"RM initialized with {self.retriever.get_source_name()}")

    def search(
        self,
        queries: Union[str, List[str]],
        max_results: int = 8,
        format_type: str = "rag",
        deduplicate: bool = True,
        topic: str = None,
        **kwargs,
    ) -> List[str]:
        """
        Single entry point for all retrieval operations.

        Args:
            queries: Single query or list of queries
            max_results: Maximum results to return
            format_type: "rag" for passages, "storm" for structured data
            deduplicate: Whether to remove duplicate results
            **kwargs: Additional parameters passed to retriever

        Returns:
            List of passages (strings) for RAG format
            List of structured dicts for STORM format
        """
        # Normalize input, if a single string is provided, convert to list
        if isinstance(queries, str):
            query_list = [queries]
        else:
            query_list = list(queries)

        # Check cache first
        cache_key = None
        if self.cache_results:
            cache_key = self._generate_cache_key(query_list, max_results, format_type)
            if cache_key in self._result_cache:
                logger.debug("Returning cached results")
                return self._result_cache[cache_key]

        try:
            # Perform the actual retrieval
            results = self.retriever.search(
                queries=query_list,
                max_results=max_results,
                format_type=format_type,
                topic=topic,
                **kwargs,
            )

            # Deduplicate if requested
            if deduplicate:
                results = self._deduplicate_results(results, format_type)

            # Limit results
            final_results = results[:max_results]

            # Cache results
            if self.cache_results and cache_key:
                self._result_cache[cache_key] = final_results

            logger.info(
                f"Retrieved {len(final_results)} results for {len(query_list)} queries"
            )
            return final_results

        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            raise RuntimeError(f"Retrieval operation failed: {e}") from e

    def get_stats(self) -> Dict:
        """Get retrieval statistics."""
        stats = {
            "retriever_type": self.retriever.get_source_name(),
            "cache_enabled": self.cache_results,
            "retriever_available": self.retriever.is_available(),
        }

        if self.cache_results:
            stats["cache_size"] = len(self._result_cache)

        return stats

    def clear_cache(self):
        """Clear the result cache."""
        if self.cache_results:
            self._result_cache.clear()
            logger.info("Retrieval cache cleared")

    def _generate_cache_key(
        self, queries: List[str], max_results: int, format_type: str
    ) -> str:
        """Generate a cache key for the given parameters."""
        import hashlib

        key_data = f"{';'.join(sorted(queries))}:{max_results}:{format_type}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def _deduplicate_results(self, results: List, format_type: str) -> List:
        """Remove duplicate results based on format type."""
        if format_type == "rag":
            # For RAG format (list of strings), deduplicate by content
            seen = set()
            unique_results = []
            for result in results:
                if isinstance(result, str):
                    # Simple content-based deduplication
                    content_hash = hash(result.strip().lower())
                    if content_hash not in seen:
                        seen.add(content_hash)
                        unique_results.append(result)
            return unique_results

        elif format_type == "storm":
            # For STORM format (list of dicts), deduplicate by URL or content
            seen_urls = set()
            unique_results = []
            for result in results:
                if isinstance(result, dict):
                    url = result.get("url", "")
                    content = result.get("content", "")

                    # Use URL if available, otherwise use content hash
                    identifier = url if url else hash(content.strip().lower())

                    if identifier not in seen_urls:
                        seen_urls.add(identifier)
                        unique_results.append(result)
            return unique_results

        else:
            # Unknown format, return as-is
            return results


class RetrievalFactory:
    """
    Factory for creating RM instances with different retrieval backends.
    Makes it easy to switch between retrieval sources.
    """

    @staticmethod
    def create_wikipedia_rm(max_articles: int = 3, max_sections: int = 3) -> RM:
        """Create a RM with Wikipedia retrieval."""
        from src.retrieval.wiki_rm import WikiRM

        wikipedia_retriever = WikiRM(
            max_articles=max_articles, max_sections=max_sections
        )

        return RM(retriever=wikipedia_retriever)
