"""
Utility class for handling all caching logic (in-memory and disk)
for the BaseRetriever.
"""

import hashlib

import json
import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class CacheManager:
    """Manages search result caching across memory and disk."""

    def __init__(
        self,
        cache_dir: str = "data/wiki_cache",
        cache_enabled: bool = True,
        retriever_class_name: str = "BaseRetriever",
    ):
        """
        Initialize the CacheManager.

        Args:
            cache_dir: Directory for disk cache files.
            cache_enabled: Flag to enable or disable all caching.
            retriever_class_name: The name of the retriever class using this manager
                                  to ensure unique cache keys.
        """
        self.cache_dir = cache_dir
        self.cache_enabled = cache_enabled
        self.retriever_class_name = retriever_class_name

        # In-memory caches, exposed for complex logic (e.g., semantic filtering checks)
        self._result_cache: Dict[str, List[Dict[str, Any]]] = {}
        self.semantic_cache: Dict[str, Any] = {}

        # Create cache directory if disk caching is enabled
        if cache_enabled:
            os.makedirs(cache_dir, exist_ok=True)

    def generate_result_cache_key(self, query: str, max_results: int) -> str:
        """Generate a unique cache key for search results."""
        # The cache key includes the queries, result limit, and the specific retriever class name
        content = f"{query}_{max_results}_{self.retriever_class_name}"
        return hashlib.md5(content.encode()).hexdigest()

    def load_from_cache(self, cache_key: str) -> Optional[List[Dict[str, Any]]]:
        """Load results from in-memory cache, falling back to disk cache."""
        if not self.cache_enabled:
            return None

        # 1. Check in-memory cache first
        if cache_key in self._result_cache:
            logger.debug(f"Loaded from in-memory cache for key: {cache_key}")
            return self._result_cache[cache_key]

        # 2. Load from file cache
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    cached_data = json.load(f)
                    self._result_cache[cache_key] = cached_data  # Populate memory cache
                    logger.debug(f"Loaded from file cache for key: {cache_key}")
                    return cached_data
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to load cache file {cache_file}: {e}")
                # Clean up corrupted cache file
                try:
                    os.remove(cache_file)
                except OSError:
                    pass

        return None

    def save_to_cache(self, cache_key: str, results: List[Dict[str, Any]]) -> None:
        """Save results to in-memory cache and disk cache."""
        if not self.cache_enabled or not results:
            return

        # 1. Save to in-memory cache
        self._result_cache[cache_key] = results

        # 2. Save to file cache
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
                logger.debug(f"Saved {len(results)} results to file cache: {cache_key}")
        except (IOError, TypeError) as e:
            logger.warning(f"Failed to save cache file {cache_file}: {e}")

    def clear_cache(self) -> None:
        """Clear all cached results (memory and disk)."""
        self._result_cache.clear()
        self.semantic_cache.clear()

        # Clear file cache
        if self.cache_enabled and os.path.exists(self.cache_dir):
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
            "memory_cache_size": len(self._result_cache),
            "semantic_cache_size": len(self.semantic_cache),
            "file_cache_size": 0,
        }
        # Count files on disk
        if self.cache_enabled and os.path.exists(self.cache_dir):
            try:
                cache_files = [
                    f for f in os.listdir(self.cache_dir) if f.endswith(".json")
                ]
                stats["file_cache_size"] = len(cache_files)
            except OSError:
                pass

        return stats
