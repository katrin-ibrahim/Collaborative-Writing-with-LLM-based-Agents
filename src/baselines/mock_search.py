"""
Mock search retrieval module for STORM integration.

This replaces DuckDuckGo search to avoid rate limiting during testing.
"""

import logging

logger = logging.getLogger(__name__)


class MockSearchRM:
    """
    Mock search retrieval manager that returns fake but realistic search results.

    Used to avoid DuckDuckGo rate limiting during STORM testing.
    Provides the exact interface that STORM expects.
    """

    def __init__(self, k=3):
        self.k = k
        logger.info(f"MockSearchRM initialized with k={k}")

    def __call__(self, query_or_queries, exclude_urls=None, **kwargs):
        """Make the object callable for STORM compatibility."""
        logger.debug(f"MockSearchRM.__call__ with queries: {query_or_queries}")
        return self.retrieve(query_or_queries, exclude_urls, **kwargs)

    def retrieve(self, query_or_queries, exclude_urls=None, **kwargs):
        """
        Generate mock search results in the format STORM expects.

        Args:
            query_or_queries: Single query string or list of queries
            exclude_urls: URLs to exclude (ignored in mock)
            **kwargs: Additional search parameters (ignored in mock)

        Returns:
            List of search result dictionaries with required STORM structure
        """
        # Handle both single query and multiple queries
        if isinstance(query_or_queries, list):
            queries = query_or_queries
        else:
            queries = [query_or_queries]

        logger.debug(f"MockSearchRM processing {len(queries)} queries")

        results = []
        for i, query in enumerate(queries):
            for j in range(self.k):
                # STORM expects this exact structure
                result = {
                    "url": f"https://mocksite{j+1}.com/article_{i+1}",
                    "snippets": [
                        f"Mock search result {j+1} for query: {query}. "
                        f"This contains relevant information about the topic and "
                        f"provides detailed context for article generation. "
                        f"Additional mock content to make it realistic."
                    ],
                    "title": f"Article about {query} - Result {j+1}",
                    "description": f"Detailed information about {query}",
                }
                results.append(result)

        final_results = results[: self.k * len(queries)]
        logger.info(
            f"MockSearchRM returned {len(final_results)} results for {len(queries)} queries"
        )

        return final_results
