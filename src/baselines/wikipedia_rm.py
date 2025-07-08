import logging
import re
from typing import Any, Dict, List

from knowledge.wikipedia_retriever import WikipediaRetriever

logger = logging.getLogger(__name__)


class WikipediaSearchRM:
    """
    High-performance Wikipedia search retrieval manager for STORM integration.

    Optimized for speed while maintaining content quality.
    """

    def __init__(self, k: int = 3):
        self.k = min(k, 5)  # Limit k to max 5 for performance
        self.retriever = WikipediaRetriever()
        logger.info(f"Optimized WikipediaSearchRM initialized with k={self.k}")

    def __call__(self, query_or_queries, exclude_urls=None, **kwargs):
        """Make the object callable for STORM compatibility."""
        logger.debug(f"WikipediaSearchRM.__call__ with queries: {query_or_queries}")
        return self.retrieve(query_or_queries, exclude_urls, **kwargs)

    def retrieve(
        self, query_or_queries, exclude_urls=None, **kwargs
    ) -> List[Dict[str, Any]]:
        """
        High-performance Wikipedia content retrieval in STORM format.

        Args:
            query_or_queries: Single query string or list of queries
            exclude_urls: URLs to exclude (applied to Wikipedia URLs)
            **kwargs: Additional search parameters

        Returns:
            List of search result dictionaries in STORM format
        """
        # Handle both single query and multiple queries
        if isinstance(query_or_queries, list):
            queries = query_or_queries
        else:
            queries = [query_or_queries]

        # Limit number of queries for performance
        queries = queries[:5]  # Max 5 queries to prevent excessive API calls

        logger.info(f"WikipediaSearchRM processing {len(queries)} queries")

        excluded_urls = set(exclude_urls or [])
        results = []

        for i, query in enumerate(queries, 1):
            logger.debug(f"Processing query {i}/{len(queries)}: '{query}'")

            try:
                # Clean the query to remove STORM prefixes and artifacts
                cleaned_query = self._clean_storm_query(query)

                if not cleaned_query or len(cleaned_query.strip()) < 2:
                    logger.debug(
                        f"Query too short after cleaning: '{query}' -> '{cleaned_query}'"
                    )
                    continue  # Skip empty queries instead of creating fallbacks

                # Get Wikipedia content for this query with reduced limits
                wiki_snippets = self.retriever.get_wiki_content(
                    topic=cleaned_query,
                    max_articles=2,  # Reduced from 3 for speed
                    max_sections=2,  # Reduced from 3 for speed
                )

                # Convert to STORM format
                query_results = self._convert_to_storm_format(
                    wiki_snippets, excluded_urls
                )

                if query_results:
                    results.extend(query_results)
                    logger.debug(
                        f"Added {len(query_results)} results for query: '{cleaned_query}'"
                    )
                else:
                    logger.debug(f"No results found for query: '{cleaned_query}'")

                # Early termination if we have enough results
                if len(results) >= self.k * 3:  # Stop if we have plenty of results
                    logger.debug(f"Early termination: collected {len(results)} results")
                    break

            except Exception as e:
                logger.warning(f"Wikipedia search failed for query '{query}': {e}")
                continue  # Skip failed queries instead of creating fallbacks

        # If no results at all, create minimal fallback
        if not results and queries:
            cleaned_main_query = self._clean_storm_query(queries[0])
            if cleaned_main_query:
                results = [self._create_fallback_result(cleaned_main_query)]

        # Limit total results and sort by relevance if available
        final_results = self._optimize_results(results, self.k * len(queries))

        logger.info(
            f"WikipediaSearchRM returned {len(final_results)} results for {len(queries)} queries"
        )
        return final_results

    def _optimize_results(self, results: List[Dict], max_results: int) -> List[Dict]:
        """Optimize and limit results for best performance."""
        if not results:
            return results

        # Remove duplicates based on URL
        seen_urls = set()
        unique_results = []

        for result in results:
            url = result.get("url", "")
            if url not in seen_urls:
                seen_urls.add(url)
                unique_results.append(result)

        # Limit results
        return unique_results[:max_results]

    def _clean_storm_query(self, query: str) -> str:
        """Clean STORM-generated queries quickly and efficiently."""
        if not query:
            return ""

        cleaned = query.strip()

        # Remove STORM query prefixes (case insensitive) - optimized regex
        cleaned = re.sub(r"^(?:[Qq]uery|[Ss]earch)\s*\d*\s*:\s*", "", cleaned)
        cleaned = re.sub(r"^\d+\.\s*", "", cleaned)  # "1. ", "2. ", etc.
        cleaned = re.sub(r"^[-*]\s*", "", cleaned)  # "- " or "* "

        # Remove quotes and clean whitespace
        cleaned = cleaned.replace('"', "").replace("'", "")
        cleaned = " ".join(cleaned.split())

        # Quick check for generic terms
        if (
            cleaned.lower() in {"query", "search", "information", "details", "about"}
            or len(cleaned) < 2
        ):
            return ""

        logger.debug(f"Cleaned STORM query: '{query}' -> '{cleaned}'")
        return cleaned

    def _convert_to_storm_format(
        self, wiki_snippets: List[Dict], excluded_urls: set
    ) -> List[Dict[str, Any]]:
        """Convert Wikipedia snippets to STORM format efficiently."""
        storm_results = []

        for snippet in wiki_snippets:
            url = snippet.get("url", "")

            # Skip if URL is in exclusion list
            if url in excluded_urls:
                continue

            content = snippet.get("content", "")
            if not content or len(content.strip()) < 50:
                continue

            # Limit content length for performance
            content = content[:1500] if len(content) > 1500 else content

            # STORM expects this exact structure
            storm_result = {
                "url": url,
                "snippets": [content],  # STORM expects a list of snippets
                "title": self._create_title(snippet),
                "description": self._create_description(snippet),
            }

            storm_results.append(storm_result)

        return storm_results

    def _create_title(self, snippet: Dict) -> str:
        """Create a title from Wikipedia snippet metadata."""
        title = snippet.get("title", "Wikipedia Article")
        section = snippet.get("section", "")

        if section and section != "Summary":
            return f"{title} - {section}"
        else:
            return title

    def _create_description(self, snippet: Dict) -> str:
        """Create a description from Wikipedia snippet metadata."""
        title = snippet.get("title", "Wikipedia Article")
        section = snippet.get("section", "")

        if section and section != "Summary":
            return f"Wikipedia article '{title}', section: {section}"
        else:
            return f"Wikipedia article: {title}"

    def _create_fallback_result(self, query: str) -> Dict[str, Any]:
        """Create a minimal fallback result when Wikipedia search fails."""
        return {
            "url": f'https://en.wikipedia.org/wiki/{query.replace(" ", "_")}',
            "snippets": [
                f"Information about {query}. This topic may require additional research for comprehensive coverage."
            ],
            "title": f"Wikipedia: {query}",
            "description": f"Wikipedia article about {query}",
        }
