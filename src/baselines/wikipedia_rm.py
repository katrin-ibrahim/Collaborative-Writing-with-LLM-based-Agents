import logging
import re
from typing import Any, Dict, List

from knowledge.wikipedia_retriever import WikipediaRetriever

logger = logging.getLogger(__name__)


class WikipediaSearchRM:
    """
    Wikipedia-based search retrieval manager for STORM integration.

    Provides real Wikipedia content while maintaining STORM's expected interface.
    """

    def __init__(self, k: int = 3):
        self.k = k
        self.retriever = WikipediaRetriever()
        logger.info(f"WikipediaSearchRM initialized with k={k}")

    def __call__(self, query_or_queries, exclude_urls=None, **kwargs):
        """Make the object callable for STORM compatibility."""
        logger.debug(f"WikipediaSearchRM.__call__ with queries: {query_or_queries}")
        return self.retrieve(query_or_queries, exclude_urls, **kwargs)

    def retrieve(
        self, query_or_queries, exclude_urls=None, **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Retrieve Wikipedia content in STORM's expected format.

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

        logger.info(f"WikipediaSearchRM processing {len(queries)} queries: {queries}")

        excluded_urls = set(exclude_urls or [])
        logger.debug(f"Exclude URLs: {excluded_urls}")
        logger.debug(f"Additional kwargs: {kwargs}")

        results = []

        for i, query in enumerate(queries, 1):
            logger.info(f"Processing query {i}/{len(queries)}: '{query}'")

            try:
                # Clean the query to remove STORM prefixes and artifacts
                cleaned_query = self._clean_storm_query(query)

                if not cleaned_query or len(cleaned_query.strip()) < 2:
                    logger.warning(
                        f"Query too short after cleaning: '{query}' -> '{cleaned_query}'"
                    )
                    results.append(self._create_fallback_result(query))
                    continue

                # Get Wikipedia content for this query
                wiki_snippets = self.retriever.get_wiki_content(
                    topic=cleaned_query,
                    max_articles=self.k,
                    max_sections=2,  # Reduced to avoid too much content
                )

                # Convert to STORM format
                query_results = self._convert_to_storm_format(
                    wiki_snippets, excluded_urls
                )

                if not query_results:
                    logger.warning(f"No results found for query: '{cleaned_query}'")
                    query_results = [self._create_fallback_result(cleaned_query)]

                results.extend(query_results)

            except Exception as e:
                logger.warning(f"Wikipedia search failed for query '{query}': {e}")
                results.append(self._create_fallback_result(query))

        # Limit total results
        final_results = results[: self.k * len(queries)]
        logger.info(
            f"WikipediaSearchRM returned {len(final_results)} results for {len(queries)} queries"
        )

        return final_results

    def _clean_storm_query(self, query: str) -> str:
        """Clean STORM-generated queries to extract the actual search term."""
        if not query:
            return ""

        cleaned = query.strip()

        # Remove STORM query prefixes (case insensitive)
        patterns_to_remove = [
            r"^[Qq]uery\s*\d*\s*:\s*",  # "query 1:", "Query 2:", etc.
            r"^[Ss]earch\s*\d*\s*:\s*",  # "search 1:", etc.
            r"^\d+\.\s*",  # "1. ", "2. ", etc.
            r"^-\s*",  # "- "
            r"^\*\s*",  # "* "
        ]

        for pattern in patterns_to_remove:
            cleaned = re.sub(pattern, "", cleaned)

        # Remove quotes that might interfere
        cleaned = cleaned.replace('"', "").replace("'", "")

        # Clean up extra whitespace
        cleaned = " ".join(cleaned.split())

        # If the query is still very generic or empty, return empty
        generic_terms = {"query", "search", "information", "details", "about"}
        if cleaned.lower() in generic_terms or len(cleaned) < 2:
            return ""

        logger.debug(f"Cleaned STORM query: '{query}' -> '{cleaned}'")
        return cleaned

    def _convert_to_storm_format(
        self, wiki_snippets: List[Dict], excluded_urls: set
    ) -> List[Dict[str, Any]]:
        """Convert Wikipedia snippets to STORM's expected format."""
        storm_results = []

        for snippet in wiki_snippets:
            url = snippet.get("url", "")

            # Skip if URL is in exclusion list
            if url in excluded_urls:
                logger.debug(f"Skipping excluded URL: {url}")
                continue

            content = snippet.get("content", "")
            if not content or len(content.strip()) < 50:
                continue

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
        snippet.get("retrieval_method", "wikipedia")

        if section and section != "Summary":
            return f"Wikipedia article '{title}', section: {section}"
        else:
            return f"Wikipedia article: {title}"

    def _create_fallback_result(self, query: str) -> Dict[str, Any]:
        """Create a fallback result when Wikipedia search fails."""
        cleaned_query = self._clean_storm_query(query) or "information"

        return {
            "url": f'https://en.wikipedia.org/wiki/{cleaned_query.replace(" ", "_")}',
            "snippets": [
                f"General information about {cleaned_query}. This topic may require further research for comprehensive coverage."
            ],
            "title": f"Wikipedia: {cleaned_query}",
            "description": f"Wikipedia article about {cleaned_query}",
        }
