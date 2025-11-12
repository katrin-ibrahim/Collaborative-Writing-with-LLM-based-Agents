"""
Base retriever interface for unified retrieval architecture.
Defines the essential contract for all retrieval sources.
"""

from abc import ABC, abstractmethod

import logging
from typing import Any, Dict, List, Optional

from src.config.retrieval_config import RetrievalConfig
from src.retrieval.utils.cache_manager import CacheManager
from src.retrieval.utils.filter import ContentFilter
from src.retrieval.utils.scorer import RelevanceScorer
from src.utils.data.models import ResearchChunk
from src.utils.experiment.experiment_setup import find_project_root

logger = logging.getLogger(__name__)


class BaseRetriever(ABC):
    """
    Abstract base class for all retrieval systems.
    Defines the standard interface and implements shared caching/processing logic.
    """

    def __init__(
        self,
        cache_dir: str = "data/wiki_cache",
        cache_results: bool = True,
        config: Optional[RetrievalConfig] = None,
        format_type: str = "rag",
    ):
        """Initialize base retriever with caching and core utilities."""
        self.config = config or RetrievalConfig()

        # Initialize Cache Manager, which handles all caching concerns
        self.cache_manager = CacheManager(
            cache_dir=cache_dir,
            cache_enabled=cache_results,
            retriever_class_name=self.__class__.__name__,
        )

        # Expose the semantic cache reference from the manager for semantic filtering access
        self.semantic_cache: Dict[str, Any] = self.cache_manager.semantic_cache
        self.cache_results = self.cache_manager.cache_enabled

        # Initialize content processing utilities (critical path dependencies)
        project_root = find_project_root()
        self.content_filter = ContentFilter(project_root)
        self.relevance_scorer = RelevanceScorer()
        self.format_type = format_type

    def __call__(self, k: Optional[int] = None, **kwargs) -> List[Dict[str, Any]]:
        """
        Allow direct call to search method for STORM compatibility.
        The final output is converted back to a list of dictionaries for the consumer framework.
        """
        # Handle flexible calling conventions for STORM compatibility
        actual_query = None
        logger.info(f"kwargs received in __call__: {kwargs}")

        if kwargs and len(kwargs) > 0:
            # STORM calls with positional arguments: retriever(query)
            actual_query = kwargs["query_or_queries"]

        # Handle empty or invalid queries gracefully
        if not actual_query:
            logger.warning("__call__ invoked with empty/invalid query")
            return []

        max_results = (
            k
            if k is not None
            else kwargs.get("max_results", self.config.results_per_query)
        )

        try:
            # Pass the single query positionally to search, which now returns List[ResearchChunk]
            chunk_results: List[ResearchChunk] = self.search(
                query_list=actual_query, max_results=max_results  # Passed positionally
            )

            # Convert List[ResearchChunk] back to List[Dict[str, Any]] (as requested)
            dict_results = []
            for chunk in chunk_results:
                result_dict = chunk.__dict__.copy()

                # Add STORM-compatible fields if they don't exist
                if "snippets" not in result_dict and "content" in result_dict:
                    result_dict["snippets"] = [result_dict["content"]]
                if "text" not in result_dict and "content" in result_dict:
                    result_dict["text"] = result_dict["content"]
                if "title" not in result_dict:
                    # Try to extract title from metadata or use a default
                    metadata = result_dict.get("metadata", {})
                    result_dict["title"] = metadata.get("article_title", "Unknown")

                dict_results.append(result_dict)

            return dict_results
        except Exception as e:
            logger.error(f"Retrieval failed for query '{actual_query}': {e}")
            return []

    # =================== Cache Methods (Delegated to CacheManager) ===================

    def _generate_result_cache_key(self, query: str, max_results: int) -> str:
        """Delegate cache key generation to the manager."""
        return self.cache_manager.generate_result_cache_key(query, max_results)

    def _load_from_cache(self, cache_key: str) -> Optional[List[Dict[str, Any]]]:
        """Delegate cache loading to the manager."""
        return self.cache_manager.load_from_cache(cache_key)

    def _save_to_cache(self, cache_key: str, results: List[Dict[str, Any]]) -> None:
        """Delegate cache saving to the manager."""
        self.cache_manager.save_to_cache(cache_key, results)

    def clear_cache(self) -> None:
        """Clear all cached results (memory and disk) via the manager."""
        self.cache_manager.clear_cache()

    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics via the manager."""
        return self.cache_manager.get_cache_stats()

    # =================== Core Search Logic ===================

    def search(
        self,  # For positional argument support
        max_results: Optional[int] = None,
        topic: Optional[str] = None,
        deduplicate: bool = True,
        query_list: Optional[List[str]] = None,
    ) -> List[ResearchChunk]:
        """
        Unified search method that handles query processing, caching, and result formatting.
        Returns a list of ResearchChunk objects.

        Args:
            max_results: Maximum number of results to retrieve PER query.
            topic: Optional topic string for post-retrieval semantic filtering/scoring.
            deduplicate: Whether to remove duplicate results across queries.
            query_list: Our internal argument for a list of queries.

        Returns:
            List of ResearchChunk objects.
        """
        logger.debug(
            f"{self.__class__.__name__}.search called with:  list={query_list},"
        )

        # --- Robust Query Normalization ---
        if query_list is not None:
            raw_queries = query_list
        # Check for positional arguments (This is the path hit when __call__ is used)
        else:
            logger.warning(
                f"{self.__class__.__name__}.search called with no valid query argument - returning empty list"
            )
            return []

        # Convert raw queries (string or list) into a definitive List[str]
        if isinstance(raw_queries, str):
            query_list_final = [raw_queries]
        elif raw_queries is not None:
            query_list_final = list(raw_queries)
        else:
            return []  # Should not happen if the checks above are correct

        # Set defaults using config values
        max_results = (
            max_results if max_results is not None else self.config.results_per_query
        )

        # 1. Check result cache

        # Mark which queries we found cached results for, if for all queries we have cached results, we can return early
        # if only some queries have cached results, we need to run the retrieval for the missing ones
        cached_queries = set()
        cache_key = None
        for query in query_list_final:
            if self.cache_results:
                cache_key = self._generate_result_cache_key(query, max_results)
                cached_results: Optional[List[Dict[str, Any]]] = self._load_from_cache(
                    cache_key
                )
                if cached_results is not None:
                    cached_queries.add(query)
        if self.cache_results and len(cached_queries) == len(query_list_final):
            all_cached_results: List[ResearchChunk] = []
            for query in query_list_final:
                cache_key = self._generate_result_cache_key(query, max_results)
                cached_results: Optional[List[Dict[str, Any]]] = self._load_from_cache(
                    cache_key
                )
                if cached_results is not None:
                    # Convert cached dicts to ResearchChunks before returning (as requested)
                    try:
                        all_cached_results.extend(
                            [ResearchChunk(**d) for d in cached_results]
                        )
                    except TypeError as e:
                        logger.error(
                            f"Failed to convert cached results to ResearchChunk: {e}. Re-running search."
                        )
                        # Fall through to perform actual retrieval if conversion fails
            return all_cached_results

        # If not all queries were cached, we need to run the retrieval for the missing ones
        missing_queries = set(query_list_final) - cached_queries
        query_list_final = list(missing_queries)
        # 2. Collect all results
        search_results: List[Dict[str, Any]] = []

        for i, query in enumerate(query_list_final):
            if not self.content_filter.validate_query(query):
                logger.warning(
                    f"{self.__class__.__name__}.search called with malformed query, skipping: '{query[:100]}...'"
                )
                continue

            # Transform queries (e.g., STORM-specific query reformatting)
            processed_query = (
                self.content_filter.transform_storm_query(query)
                if self.format_type == "storm"
                else query
            )

            # --- Abstract Retrieval Step ---
            query_results = self._retrieve_article(
                processed_query, max_results=max_results
            )

            query_results_dicts = {
                res.chunk_id: res.model_dump() for res in query_results
            }
            search_results.extend(query_results_dicts.values())

        # 3. Deduplicate
        if deduplicate:
            # Assumes content_filter works on the dict format
            search_results = self.content_filter.deduplicate_results(search_results)

        # 4. Apply semantic/topic relevance scoring (re-ranking)
        if topic and self.config.semantic_filtering_enabled:
            search_results = self.relevance_scorer.calculate_relevance(
                topic, search_results
            )

        # 5. Filter excluded URLs (data leakage prevention)
        search_results = self.content_filter.filter_excluded_urls(search_results)

        # 6. Limit final result set
        search_results = search_results[: self.config.final_passages]

        # 7. Save to cache (still saving dicts for simple I/O)
        if self.cache_results and cache_key:
            self._save_to_cache(cache_key, search_results)

        # 8. Convert final dict list to ResearchChunk list (as requested for search return)
        try:
            chunk_results = [ResearchChunk(**d) for d in search_results]
        except TypeError as e:
            logger.error(
                f"Failed to convert final search results to ResearchChunk: {e}. Returning raw list of dicts."
            )
            # Fallback for safety, though it violates the type hint
            return search_results  # type: ignore

        return chunk_results

    def search_concurrent(
        self,
        query_list: List[str],
        max_results: Optional[int] = None,
        max_workers: int = 3,
        topic: Optional[str] = None,
        deduplicate: bool = True,
        **kwargs,
    ) -> List[ResearchChunk]:
        """
        Concurrent version of search() for multiple queries.
        Returns a flattened and processed list of all results as ResearchChunk objects.

        Args:
            query_list: List of query strings to search concurrently
            max_results: Maximum number of results per query
            max_workers: Maximum number of concurrent threads
            topic: Optional topic to filter results
            deduplicate: Whether to remove duplicate results
            **kwargs: Additional retrieval parameters

        Returns:
            Flattened and processed list of all results as ResearchChunk objects.
        """
        if not query_list:
            return []

        # If max_workers is not defined, calculate a safe default
        max_workers = min(max_workers, len(query_list))

        # Use concurrent.futures for simple thread pooling
        import concurrent.futures

        logger.info(
            f"Executing {len(query_list)} queries concurrently with {max_workers} workers"
        )

        all_results: List[ResearchChunk] = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit each query as a separate search call
            future_to_query = {
                executor.submit(
                    # Call the standard search method, which now returns List[ResearchChunk]
                    self.search,
                    query_list=[query],  # Passed as a list
                    max_results=max_results,
                    topic=topic,
                    # Disable deduplication inside the concurrent call
                    deduplicate=True,
                ): query
                for i, query in enumerate(query_list)
            }

            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_query):
                query = future_to_query[future]
                try:
                    # Results are List[ResearchChunk]
                    results: List[ResearchChunk] = future.result()
                    all_results.extend(results)
                    logger.debug(
                        f"Concurrent search for '{query}' returned {len(results)} results"
                    )
                except Exception as exc:
                    logger.error(f"Concurrent search query '{query}' failed: {exc}")

        logger.info(f"Concurrent search completed: {len(all_results)} raw results.")

        # Perform final steps on the aggregated list
        if deduplicate:
            # Convert chunks to dicts for content_filter (which assumes dicts)
            dict_results = [chunk.__dict__ for chunk in all_results]
            deduped_dict_results = self.content_filter.deduplicate_results(dict_results)
            # Convert back to chunks
            all_results = [ResearchChunk(**d) for d in deduped_dict_results]

        # Re-apply final result limit (necessary because each thread might have returned max_results)
        all_results = all_results[: self.config.final_passages]

        return all_results

    # =================== Abstract Method (Contract) ===================

    @abstractmethod
    def _retrieve_article(
        self, query: str, max_results: Optional[int] = None
    ) -> List[ResearchChunk]:
        """
        Retrieve articles for a single query. **MUST be implemented by subclasses.**
        The raw results are expected to be returned as a list of dictionaries.

        Args:
            query: Single query string
            max_results: Maximum results to return

        Returns:
            List of result dictionaries. Each dict should conform to a standard
            structure (e.g., must contain 'snippets', 'title', 'url' keys).
        """
