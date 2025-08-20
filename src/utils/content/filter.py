"""
Content filtering utilities for URL exclusion and result deduplication.
"""

import csv
from pathlib import Path

import logging
from typing import Any, Dict, List, Union

logger = logging.getLogger(__name__)


class ContentFilter:
    """Handles content filtering, deduplication, and URL exclusion."""

    def __init__(self, project_root: str = None):
        """
        Initialize content filter.

        Args:
            project_root: Root directory of the project for loading exclusion lists
        """
        self.project_root = project_root or Path.cwd()
        self.exclude_urls = set()
        self._load_exclude_urls()

    def _load_exclude_urls(self) -> None:
        """Load URLs from freshwiki dataset to exclude (prevent data leakage)."""
        freshwiki_csv = (
            Path(self.project_root) / "data" / "fw" / "topic_list_with_categories.csv"
        )

        if not freshwiki_csv.exists():
            logger.warning(f"FreshWiki CSV not found at {freshwiki_csv}")
            return

        try:
            with open(freshwiki_csv, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if "url" in row and row["url"]:
                        self.exclude_urls.add(row["url"])
                        # Also add the title-based URL format
                        if "topic" in row and row["topic"]:
                            topic_url = f"https://en.wikipedia.org/wiki/{row['topic'].replace(' ', '_')}"
                            self.exclude_urls.add(topic_url)

            logger.info(
                f"Loaded {len(self.exclude_urls)} URLs to exclude from FreshWiki dataset"
            )
        except Exception as e:
            logger.warning(f"Failed to load FreshWiki exclude URLs: {e}")
            self.exclude_urls = set()

    def filter_excluded_urls(
        self, results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Filter out results with excluded URLs to prevent data leakage.

        Args:
            results: List of search result dictionaries

        Returns:
            Filtered results with excluded URLs removed
        """
        if not self.exclude_urls:
            return results

        filtered_results = []
        for result in results:
            url = result.get("url", "")
            if url and url not in self.exclude_urls:
                filtered_results.append(result)
            elif not url:
                # Keep results without URLs
                filtered_results.append(result)

        if len(filtered_results) != len(results):
            logger.info(
                f"Excluded {len(results) - len(filtered_results)} results due to URL filtering"
            )

        return filtered_results

    def deduplicate_results(
        self, results: List[Union[str, Dict[str, Any]]], format_type: str = "dict"
    ) -> List[Union[str, Dict[str, Any]]]:
        """
        Remove duplicate results based on content or URL.

        Args:
            results: List of results to deduplicate
            format_type: Type of results ("dict", "string", or "auto")

        Returns:
            Deduplicated results
        """
        if not results:
            return results

        if format_type == "auto":
            format_type = "dict" if isinstance(results[0], dict) else "string"

        if format_type == "dict":
            return self._deduplicate_dict_results(results)
        else:
            return self._deduplicate_string_results(results)

    def _deduplicate_dict_results(
        self, results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Deduplicate dictionary-based results."""
        seen_urls = set()
        seen_content_hashes = set()
        unique_results = []

        for result in results:
            url = result.get("url", "")
            content = result.get("snippets", "") or result.get("content", "")

            # Use URL as primary identifier
            if url:
                if url not in seen_urls:
                    seen_urls.add(url)
                    unique_results.append(result)
                    continue

            # Fall back to content hash if no URL
            if content:
                # Handle both string and list content
                if isinstance(content, list):
                    # For lists, join all snippets and use that for hashing
                    content_str = " ".join(
                        str(snippet) for snippet in content if snippet
                    )
                else:
                    content_str = str(content)

                if content_str:
                    content_hash = hash(content_str.strip().lower())
                    if content_hash not in seen_content_hashes:
                        seen_content_hashes.add(content_hash)
                        unique_results.append(result)
            elif not url:  # Empty result, but keep it
                unique_results.append(result)

        logger.debug(
            f"Deduplication: {len(results)} → {len(unique_results)} dict results"
        )
        return unique_results

    def _deduplicate_string_results(self, results: List[str]) -> List[str]:
        """Deduplicate string-based results."""
        seen_hashes = set()
        unique_results = []

        for result in results:
            if isinstance(result, str):
                content_hash = hash(result.strip().lower())
                if content_hash not in seen_hashes:
                    seen_hashes.add(content_hash)
                    unique_results.append(result)
            else:
                # Non-string result, keep it
                unique_results.append(result)

        logger.debug(
            f"Deduplication: {len(results)} → {len(unique_results)} string results"
        )
        return unique_results

    def validate_query(self, query: str) -> bool:
        """
        Validate if a query is a proper search query and not malformed template text.

        Args:
            query: The query string to validate

        Returns:
            True if query appears valid, False if it's malformed template text
        """
        if not isinstance(query, str) or not query.strip():
            return False

        query_lower = query.lower().strip()

        # Detect common malformed template patterns from STORM
        malformed_patterns = [
            "here are the queries",
            "i would use",
            "search:",
            "query:",
            "queries:",
            "would search for",
            "you could search",
            "try searching",
            "search terms:",
            "google search:",
            "wikipedia search:",
            "search queries:",
            "relevant searches:",
            "suggested queries:",
            "recommended searches:",
            "useful searches:",
            "potential queries:",
            "search suggestions:",
            "you might search",
            "one could search",
            "searches include:",
            "good searches:",
            "effective queries:",
            "google",
        ]

        # Check if query starts with malformed patterns
        for pattern in malformed_patterns:
            if query_lower.startswith(pattern):
                return False

        # Detect queries that are mostly template language rather than search terms
        template_phrases = [
            "here are",
            "i would",
            "you could",
            "one could",
            "try to",
            "might be",
            "would be",
            "could be",
            "search for information about",
            "look up information",
            "find information",
            "research about",
            "information on",
        ]

        # If query contains mostly template language, it's likely malformed
        template_word_count = 0
        query_words = query_lower.split()
        total_words = len(query_words)

        for phrase in template_phrases:
            if phrase in query_lower:
                template_word_count += len(phrase.split())

        # If more than 40% of words are template language, consider it malformed
        if total_words > 0 and (template_word_count / total_words) > 0.4:
            return False

        # Detect very long queries that are likely explanatory text rather than search queries
        if len(query) > 200:  # Real search queries are typically much shorter
            return False

        return True

    def transform_storm_query(self, query: str) -> str:
        """
        Transform STORM-style query to keywords for better search results.

        Args:
            query: Original query string

        Returns:
            Transformed query string
        """
        import re

        # Extract quoted phrases
        quoted = re.findall(r'"([^"]*)"', query)

        # Remove question words and punctuation
        clean = re.sub(
            r"\b(what|how|when|where|why|who|which|does|is|are|mean|means)\b",
            "",
            query,
            flags=re.IGNORECASE,
        )
        clean = re.sub(r'[?!"]', "", clean)
        clean = re.sub(r"\s+", " ", clean).strip()

        # Combine quoted phrases with cleaned query
        if quoted:
            result = f'"{quoted[0]}" {clean}'
        else:
            result = clean

        return result if len(result) > 3 else query
