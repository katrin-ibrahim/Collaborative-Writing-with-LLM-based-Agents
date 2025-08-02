"""
Clean, simplified Wikipedia retriever.
Eliminates the wrapper layers and provides direct, efficient Wikipedia access.
"""

import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed

import json
import logging
import os
import wikipedia
from typing import Dict, List, Union

from src.retrieval.base_retriever import BaseRetriever

logger = logging.getLogger(__name__)


class WikiRM(BaseRetriever):
    """
    Clean, simplified Wikipedia retriever that eliminates the wrapper hell.

    Single implementation that can output both RAG and STORM formats
    without multiple conversion layers.
    """

    def __init__(
        self,
        max_articles: int = 3,
        max_sections: int = 3,
        cache_dir: str = "data/wiki_cache",
    ):
        self.max_articles = max_articles
        self.max_sections = max_sections
        self.cache_dir = cache_dir

        # Setup Wikipedia client
        os.makedirs(cache_dir, exist_ok=True)
        wikipedia.set_rate_limiting(True)
        wikipedia.set_lang("en")

        logger.info(
            f"WikiRM initialized (articles={max_articles}, sections={max_sections})"
        )

    def search(
        self,
        queries: Union[str, List[str]],
        max_results: int = 8,
        format_type: str = "rag",
        **kwargs,
    ) -> List:
        """
        Single search method that handles both RAG and STORM formats.
        Eliminates the need for separate search() and retrieve() methods.
        """
        # Normalize input
        if isinstance(queries, str):
            query_list = [queries]
        else:
            query_list = list(queries)

        # Collect all results
        all_results = []

        for query in query_list:
            query_results = self._search_single_query(query, format_type)
            all_results.extend(query_results)

            # Early termination if we have enough results
            if len(all_results) >= max_results:
                break

        # Return in requested format
        if format_type == "rag":
            # For RAG: return list of content strings
            passages = []
            for result in all_results[:max_results]:
                if isinstance(result, dict) and "content" in result:
                    passages.append(result["content"])
                elif isinstance(result, str):
                    passages.append(result)
            return passages

        elif format_type == "storm":
            # For STORM: return list of structured dicts
            return all_results[:max_results]

        else:
            raise ValueError(f"Unknown format_type: {format_type}")

    def is_available(self) -> bool:
        """Check if Wikipedia is accessible - fail clearly if not."""
        try:
            # Simple test search
            test_results = wikipedia.search("test", results=1)
            return len(test_results) > 0
        except Exception as e:
            logger.error(f"Wikipedia not available: {e}")
            return False

    def _search_single_query(self, query: str, format_type: str) -> List[Dict]:
        """Search for a single query and return structured results."""
        # Check cache first
        cache_key = self._get_cache_key(query)
        cached_results = self._load_from_cache(cache_key)
        if cached_results:
            return cached_results

        results = []

        try:
            # Search Wikipedia
            search_results = wikipedia.search(query, results=self.max_articles)

            if not search_results:
                raise ValueError(f"No Wikipedia results found for query: '{query}'")

            # Process pages in parallel for speed
            with ThreadPoolExecutor(max_workers=3) as executor:
                future_to_title = {
                    executor.submit(self._extract_page_content, title, query): title
                    for title in search_results[: self.max_articles]
                }

                for future in as_completed(future_to_title):
                    page_results = future.result()
                    if page_results:
                        results.extend(page_results)

        except Exception as e:
            logger.error(f"Wikipedia search failed for '{query}': {e}")
            raise RuntimeError(
                f"Wikipedia search failed for query '{query}': {e}"
            ) from e

        # Cache the results
        self._save_to_cache(cache_key, results)

        return results

    def _extract_page_content(self, page_title: str, original_query: str) -> List[Dict]:
        """Extract content from a single Wikipedia page."""
        results = []

        try:
            page = wikipedia.page(page_title, auto_suggest=False)

            # Add summary as first result
            if page.summary:
                results.append(
                    {
                        "title": page.title,
                        "section": "Summary",
                        "content": page.summary.strip(),
                        "url": page.url,
                        "relevance": self._calculate_relevance(
                            page.title, original_query
                        ),
                        "source": "wikipedia",
                    }
                )

            # Add top sections
            if hasattr(page, "sections") and page.sections:
                top_sections = self._get_relevant_sections(page, original_query)

                for section_title in top_sections[: self.max_sections]:
                    try:
                        section_content = page.section(section_title)
                        if section_content and len(section_content.strip()) > 100:
                            results.append(
                                {
                                    "title": page.title,
                                    "section": section_title,
                                    "content": section_content.strip()[
                                        :2000
                                    ],  # Limit length
                                    "url": page.url,
                                    "relevance": self._calculate_relevance(
                                        section_title, original_query
                                    ),
                                    "source": "wikipedia",
                                }
                            )
                    except Exception:
                        continue  # Skip problematic sections

        except wikipedia.exceptions.DisambiguationError as e:
            # Try first disambiguation option
            if e.options:
                try:
                    return self._extract_page_content(e.options[0], original_query)
                except Exception:
                    pass

        except wikipedia.exceptions.PageError:
            logger.debug(f"Wikipedia page not found: {page_title}")
        except Exception as e:
            logger.warning(f"Error extracting content from '{page_title}': {e}")

        return results

    def _get_relevant_sections(self, page, query: str) -> List[str]:
        """Get the most relevant sections based on the query."""
        if not hasattr(page, "sections"):
            return []

        sections = page.sections
        query_words = set(query.lower().split())

        # Score sections by relevance
        scored_sections = []
        for section in sections:
            # Skip very short sections or common generic ones
            if len(section) < 3 or section.lower() in {
                "see also",
                "references",
                "external links",
                "notes",
            }:
                continue

            # Calculate relevance score
            section_words = set(section.lower().split())
            relevance = len(query_words.intersection(section_words))
            scored_sections.append((relevance, section))

        # Sort by relevance and return top sections
        scored_sections.sort(reverse=True, key=lambda x: x[0])
        return [section for _, section in scored_sections]

    def _calculate_relevance(self, text: str, query: str) -> float:
        """Calculate relevance score between text and query."""
        if not text or not query:
            return 0.0

        text_words = set(text.lower().split())
        query_words = set(query.lower().split())

        if not query_words:
            return 0.0

        # Jaccard similarity
        intersection = len(text_words.intersection(query_words))
        union = len(text_words.union(query_words))

        return intersection / union if union > 0 else 0.0

    def _get_cache_key(self, query: str) -> str:
        """Generate cache key for query."""
        content = f"{query}_{self.max_articles}_{self.max_sections}"
        return hashlib.md5(content.encode()).hexdigest()

    def _load_from_cache(self, cache_key: str) -> List[Dict]:
        """Load results from cache."""
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")

        if os.path.exists(cache_file):
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass  # Cache corrupted, will regenerate

        return None

    def _save_to_cache(self, cache_key: str, results: List[Dict]):
        """Save results to cache."""
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")

        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"Failed to cache results: {e}")
