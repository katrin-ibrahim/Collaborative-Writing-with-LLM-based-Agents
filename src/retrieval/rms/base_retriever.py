"""
Base retriever interface for unified retrieval architecture.
Eliminates inconsistent interfaces across different retrieval sources.
"""

import hashlib
from abc import ABC, abstractmethod

import json
import logging
import os
from typing import Dict, List, Optional, Union

from src.config.retrieval_config import RetrievalConfig
from src.utils.content.filter import ContentFilter
from src.utils.content.scorer import RelevanceScorer
from src.utils.experiment.experiment_setup import find_project_root

logger = logging.getLogger(__name__)


class BaseRetriever(ABC):
    """
    Abstract base class for all retrieval systems.
    Defines the standard interface that all retrievers must implement.
    """

    def __init__(
        self,
        cache_dir: str = "data/wiki_cache",
        cache_results: bool = True,
        config: RetrievalConfig = None,
        format_type: str = "rag",
    ):
        """Initialize base retriever with caching functionality."""
        self.cache_dir = cache_dir
        self.cache_results = cache_results
        self._result_cache = {} if cache_results else None
        self.semantic_cache = {}

        # Store config for use in search orchestration
        self.config = config or RetrievalConfig()

        # Initialize content processing utilities
        self.content_filter = ContentFilter(find_project_root())
        self.relevance_scorer = RelevanceScorer()
        self.semantic_enabled = self.config.semantic_filtering_enabled
        self.format_type = format_type

        # Create cache directory if it doesn't exist
        if cache_results:
            os.makedirs(cache_dir, exist_ok=True)

    def __call__(self, query=None, k=None, *args, **kwargs):
        """
        Allow direct call to search method for convenience.
        Handles STORM's calling convention: search_rm(query, k=5)
        """
        if query is not None:
            # STORM-style call: search_rm(query, k=5)
            max_results = k if k is not None else kwargs.get("max_results", None)
            return self.search(
                query_or_queries=query, max_results=max_results, **kwargs
            )
        else:
            # Standard call: search_rm(queries=..., max_results=...)
            return self.search(*args, **kwargs)

    # =================== Cache Methods ===================

    def _generate_result_cache_key(
        self, queries: List[str], max_results: int, format_type: str
    ) -> str:
        """Generate a cache key for search results."""
        queries_str = "|".join(sorted(queries))
        content = f"{queries_str}_{max_results}_{format_type}_{self.__class__.__name__}"
        return hashlib.md5(content.encode()).hexdigest()

    def _get_cache_key(self, query: str) -> str:
        """Generate cache key for a single query."""
        content = f"{query}_{self.config.results_per_query}_{self.config.max_content_pieces}_{self.__class__.__name__}"
        return hashlib.md5(content.encode()).hexdigest()

    def _load_from_cache(self, cache_key: str) -> Optional[List[Dict]]:
        """Load results from cache file."""
        if not self.cache_results:
            return None

        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")

        if os.path.exists(cache_file):
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    cached_data = json.load(f)
                    logger.debug(f"Loaded cached results for key: {cache_key}")
                    return cached_data
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to load cache file {cache_file}: {e}")
                # Clean up corrupted cache file
                try:
                    os.remove(cache_file)
                except OSError:
                    pass

        return None

    def _save_to_cache(self, cache_key: str, results: List[Dict]) -> None:
        """Save results to cache file."""
        if not self.cache_results or not results:
            return

        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")

        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
                logger.debug(f"Saved {len(results)} results to cache: {cache_key}")
        except (IOError, TypeError) as e:
            logger.warning(f"Failed to save cache file {cache_file}: {e}")

    def clear_cache(self) -> None:
        """Clear all cached results."""
        if self._result_cache:
            self._result_cache.clear()

        if self.semantic_cache:
            self.semantic_cache.clear()

        # Clear file cache
        if self.cache_results and os.path.exists(self.cache_dir):
            try:
                for file in os.listdir(self.cache_dir):
                    if file.endswith(".json"):
                        os.remove(os.path.join(self.cache_dir, file))
                logger.info("Cleared all cache files")
            except OSError as e:
                logger.warning(f"Failed to clear cache directory: {e}")

    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        stats = {
            "memory_cache_size": len(self._result_cache) if self._result_cache else 0,
            "semantic_cache_size": (
                len(self.semantic_cache) if self.semantic_cache else 0
            ),
            "file_cache_size": 0,
        }

        if self.cache_results and os.path.exists(self.cache_dir):
            try:
                cache_files = [
                    f for f in os.listdir(self.cache_dir) if f.endswith(".json")
                ]
                stats["file_cache_size"] = len(cache_files)
            except OSError:
                pass

        return stats

    def search(
        self,
        *args,
        max_results: int = None,
        topic: str = None,
        deduplicate: bool = True,
        query_or_queries: Union[str, List[str]] = None,
        **kwargs,
    ) -> List:
        """
        Unified search method that handles query processing, caching, and result formatting.
        Subclasses only need to implement _retrieve_article().

        Args:
            query_or_queries: Single query string or list of query strings
            max_results: Maximum number of results to return
            topic: Optional topic to filter results
            deduplicate: Whether to remove duplicate results
            **kwargs: Additional retrieval parameters

        Returns:
            List of passages (format_type="rag") or structured results (format_type="storm")
        """
        # Handle various calling conventions
        logger.debug(
            f"{self.__class__.__name__}.search called with: args={args}, queries={query_or_queries}, kwargs={kwargs}"
        )

        if query_or_queries is None:
            if args:
                # Positional arguments: search("query", 5) or search("query")
                query_or_queries = args[0] if len(args) > 0 else None
                if max_results is None and len(args) > 1:
                    max_results = args[1]
            elif (
                "query_or_queries" in kwargs and kwargs["query_or_queries"] is not None
            ):
                # Query in kwargs
                query_or_queries = kwargs.pop("query_or_queries", None)
            elif len(args) == 0 and not kwargs:
                # Called with no arguments - this might be a STORM initialization call
                logger.warning(
                    f"{self.__class__.__name__}.search called with no arguments - returning empty list"
                )
                return []
            else:
                raise ValueError(
                    "No valid query provided. Expected 'query_or_queries' parameter, or positional argument."
                )

        # Set defaults using config values
        max_results = (
            max_results if max_results is not None else self.config.results_per_query
        )

        # Normalize input
        if isinstance(query_or_queries, str):
            query_list = [query_or_queries]
        else:
            query_list = list(query_or_queries)

        # Check result cache first
        cache_key = None
        if self.cache_results:
            cache_key = self._generate_result_cache_key(
                query_list, max_results, self.format_type
            )
            if cache_key in self._result_cache:
                logger.debug("Returning cached search results")
                return self._result_cache[cache_key]

        # Collect all results
        search_results = []

        # 1. Get all candidate results for each query
        for i, query in enumerate(query_list):
            logger.info(f"Searching for query {i + 1}/{len(query_list)}: {query}")
            if not self.content_filter.validate_query(query):
                logger.warning(
                    f"{self.__class__.__name__}.search called with malformed query, skipping: '{query[:100]}...'"
                )
                continue

            # Transform STORM queries if needed
            if self.format_type == "storm":
                query = self.content_filter.transform_storm_query(query)

            query_results = self._retrieve_article(
                query, topic=topic, max_results=max_results
            )
            search_results.extend(query_results)

        # 2. Deduplicate if requested
        if deduplicate:
            search_results = self.content_filter.deduplicate_results(
                search_results, format_type="dict"
            )

        # 3. Apply semantic/topic relevance scoring
        if topic and self.config.semantic_filtering_enabled:
            # if len(search_results) > self.config.max_content_pieces:
            search_results = self.relevance_scorer.calculate_relevance(
                topic, search_results
            )

        # 4. Filter out excluded URLs (data leakage prevention)
        search_results = self.content_filter.filter_excluded_urls(search_results)

        # 5. Limit results to final passages count from config
        search_results = search_results[: self.config.final_passages]

        # Cache results
        if self.cache_results and cache_key:
            self._result_cache[cache_key] = search_results

        return search_results

    def search_concurrent(
        self,
        query_list: List[str],
        max_results: int = None,
        max_workers: int = 3,
        topic: str = None,
        deduplicate: bool = True,
        **kwargs,
    ) -> List:
        """
        Concurrent version of search() for multiple queries.
        Preserves existing search() signature for STORM compatibility.

        Args:
            query_list: List of query strings to search concurrently
            max_results: Maximum number of results per query
            max_workers: Maximum number of concurrent threads (default: 3)
            topic: Optional topic to filter results
            deduplicate: Whether to remove duplicate results
            **kwargs: Additional retrieval parameters

        Returns:
            Flattened list of all results from all queries
        """
        if not query_list:
            return []

        if len(query_list) == 1:
            # Single query - no concurrency overhead
            return self.search(
                query_or_queries=query_list[0],
                max_results=max_results,
                topic=topic,
                deduplicate=deduplicate,
                **kwargs,
            )

        # Multiple queries - concurrent execution
        import concurrent.futures

        logger.info(
            f"Executing {len(query_list)} queries concurrently with {min(max_workers, len(query_list))} workers"
        )

        all_results = []
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(max_workers, len(query_list))
        ) as executor:
            # Submit all queries
            future_to_query = {
                executor.submit(
                    self.search,
                    query_or_queries=query,
                    max_results=max_results,
                    topic=topic,
                    deduplicate=deduplicate,
                    **kwargs,
                ): query
                for query in query_list
            }

            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_query):
                query = future_to_query[future]
                try:
                    results = future.result()
                    all_results.extend(results)
                    logger.debug(
                        f"Concurrent search for '{query}' returned {len(results)} results"
                    )
                except Exception as exc:
                    logger.error(f"Concurrent search query '{query}' failed: {exc}")

        logger.info(
            f"Concurrent search completed: {len(all_results)} total results from {len(query_list)} queries"
        )
        return all_results

    def _generate_description(self, content: str, title: str = "") -> str:
        """
        Extract substantive content from chunks, skipping headers and fluff.
        Optimized for individual chunks rather than full articles.

        Args:
            content: The chunk content text to generate description from
            title: Optional title for fallback if content is empty

        Returns:
            A meaningful description string (usually 150-250 characters)
        """
        if not content:
            return title if title else "No content available"

        lines = content.split("\n")
        content_lines = []

        # Filter out headers, short lines, and navigation text
        for line in lines:
            line = line.strip()
            if (
                len(line) > 30
                and not line.startswith("#")
                and not line.startswith("=")
                and not line.lower().startswith("see also")
                and not line.lower().startswith("references")
                and not line.lower().startswith("external links")
                and not line.lower().startswith("category:")
                and not line.lower().startswith("file:")
                and not "{{" in line
            ):  # Skip wiki templates
                content_lines.append(line)

        if not content_lines:
            # Fallback: use original content but clean it
            clean_content = content.replace("\n", " ").strip()
            return (
                clean_content[:200] + "..."
                if len(clean_content) > 200
                else clean_content
            )

        # Take the first substantial content line
        first_content = content_lines[0]

        # If it's very long, trim to sentence boundary
        if len(first_content) > 200:
            import re

            sentences = re.split(r"[.!?]+", first_content)
            first_content = sentences[0].strip()
            if len(first_content) < 100 and len(sentences) > 1:
                second_sentence = sentences[1].strip()
                if second_sentence:
                    first_content += ". " + second_sentence

        # Final cleanup and length check
        first_content = first_content.strip()
        if len(first_content) > 250:
            first_content = first_content[:247] + "..."

        return first_content if first_content else (title or "No content available")

    @abstractmethod
    def _retrieve_article(
        self, query: str, topic: str = None, max_results: int = None
    ) -> List[Dict]:
        """
        Retrieve articles for a single query. Must be implemented by subclasses.

        Args:
            query: Single query string
            topic: Optional topic for filtering
            max_results: Maximum results to return (uses config.results_per_query if None)

        Returns:
            List of result dictionaries with 'snippets', 'title', 'url' keys
        """
