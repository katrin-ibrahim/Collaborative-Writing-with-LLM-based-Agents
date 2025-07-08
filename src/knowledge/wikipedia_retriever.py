import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed

import json
import logging
import os
import re
import wikipedia
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class WikipediaRetriever:
    """High-performance Wikipedia retriever optimized for speed and efficiency."""

    def __init__(self, cache_dir: str = "data/wiki_cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        wikipedia.set_rate_limiting(True)  # Use library's built-in rate limiting only
        wikipedia.set_lang("en")
        self.request_timeout = 10  # 10 second timeout per request
        logger.info(
            f"Optimized WikipediaRetriever initialized with cache at: {cache_dir}"
        )

    def _clean_query(self, query: str) -> str:
        """Clean and normalize search queries."""
        # Remove STORM query prefixes like "query 1:", "Query 2:", etc.
        cleaned = re.sub(r"^[Qq]uery\s*\d*\s*:\s*", "", query.strip())

        # Remove quotes that might interfere with search
        cleaned = cleaned.replace('"', "").replace("'", "")

        # Fix common OCR/transcription errors
        replacements = {
            "bob dan": "Bob Dylan",
            "ted buddy": "Ted Bundy",
            "car grant": "Cary Grant",
        }

        cleaned_lower = cleaned.lower()
        for wrong, correct in replacements.items():
            if wrong in cleaned_lower:
                cleaned = re.sub(
                    re.escape(wrong), correct, cleaned, flags=re.IGNORECASE
                )

        # Remove extra whitespace
        cleaned = " ".join(cleaned.split())

        logger.debug(f"Cleaned query: '{query}' -> '{cleaned}'")
        return cleaned

    def get_wiki_content(
        self, topic: str, max_articles: int = 3, max_sections: int = 3
    ) -> List[Dict[str, Any]]:
        """Retrieve Wikipedia content with optimized performance."""
        cleaned_topic = self._clean_query(topic)

        cache_key = hashlib.md5(
            f"{cleaned_topic}_{max_articles}_{max_sections}_optimized_v3".encode()
        ).hexdigest()
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")

        if os.path.exists(cache_file):
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    cached_data = json.load(f)
                logger.info(f"Loaded cached Wikipedia data for '{cleaned_topic}'")
                return cached_data
            except (json.JSONDecodeError, FileNotFoundError):
                logger.warning(
                    f"Cache file corrupted for '{cleaned_topic}', retrieving fresh data"
                )

        snippets = self._retrieve_with_optimized_search(
            cleaned_topic, max_articles, max_sections
        )

        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(snippets, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"Failed to cache Wikipedia data: {e}")

        return snippets

    def _retrieve_with_optimized_search(
        self, topic: str, max_articles: int, max_sections: int
    ) -> List[Dict[str, Any]]:
        """Optimized retrieval with minimal API calls and parallel processing."""
        if not topic or len(topic.strip()) < 2:
            logger.warning(f"Topic too short or empty: '{topic}'")
            return []

        # Single search with the main topic
        try:
            logger.debug(f"Searching Wikipedia for: '{topic}'")
            search_results = wikipedia.search(topic, results=max_articles)

            if not search_results:
                logger.warning(f"No Wikipedia pages found for '{topic}'")
                return []

            # Process pages in parallel for speed
            return self._extract_content_parallel(
                search_results[:max_articles], max_sections, topic
            )

        except Exception as e:
            logger.warning(f"Search failed for topic '{topic}': {e}")
            return []

    def _extract_content_parallel(
        self, page_titles: List[str], max_sections: int, original_topic: str
    ) -> List[Dict[str, Any]]:
        """Extract content from multiple pages in parallel."""
        snippets = []

        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=3) as executor:
            # Submit all page extraction tasks
            future_to_title = {
                executor.submit(
                    self._extract_single_page, title, max_sections, original_topic
                ): title
                for title in page_titles
            }

            # Collect results as they complete
            for future in as_completed(future_to_title, timeout=30):
                title = future_to_title[future]
                try:
                    page_snippets = future.result(timeout=10)
                    snippets.extend(page_snippets)

                    # Early termination if we have enough content
                    if len(snippets) >= 10:  # Stop at 10 good snippets
                        logger.debug(
                            f"Early termination: collected {len(snippets)} snippets"
                        )
                        break

                except Exception as e:
                    logger.warning(f"Failed to extract content from '{title}': {e}")
                    continue

        # Sort by relevance and return best results
        snippets.sort(key=lambda x: x.get("relevance", 0), reverse=True)
        logger.info(
            f"Retrieved {len(snippets)} Wikipedia snippets for '{original_topic}'"
        )
        return snippets[:15]  # Limit to top 15 snippets

    def _extract_single_page(
        self, page_title: str, max_sections: int, original_topic: str
    ) -> List[Dict[str, Any]]:
        """Extract content from a single Wikipedia page with timeout protection."""
        snippets = []
        source_id = hash(page_title) % 10000  # Generate consistent source_id

        try:
            logger.debug(f"Extracting content from page: '{page_title}'")
            page = wikipedia.page(page_title, auto_suggest=True)

            # Always add summary first (most important content)
            if page.summary and len(page.summary.strip()) > 50:
                snippets.append(
                    {
                        "title": page.title,
                        "section": "Summary",
                        "content": page.summary.strip(),
                        "url": page.url,
                        "source_id": source_id,
                        "retrieval_method": "wikipedia_summary",
                        "relevance": self._calculate_relevance(
                            page.title, original_topic
                        ),
                    }
                )

            # Extract limited sections for speed
            if hasattr(page, "sections") and page.sections:
                relevant_sections = self._get_top_sections(
                    page, original_topic, max_sections
                )

                for section_title, content in relevant_sections:
                    if content and len(content.strip()) > 100:
                        snippets.append(
                            {
                                "title": page.title,
                                "section": section_title,
                                "content": content.strip()[
                                    :2000
                                ],  # Limit content length
                                "url": page.url,
                                "source_id": source_id,
                                "retrieval_method": "wikipedia_section",
                                "relevance": self._calculate_relevance(
                                    section_title, original_topic
                                ),
                            }
                        )

        except wikipedia.exceptions.DisambiguationError as e:
            # Only try first disambiguation option for speed
            if e.options and len(e.options) > 0:
                try:
                    page = wikipedia.page(e.options[0], auto_suggest=False)
                    if page.summary and len(page.summary.strip()) > 50:
                        snippets.append(
                            {
                                "title": page.title,
                                "section": "Summary (Disambiguated)",
                                "content": page.summary.strip(),
                                "url": page.url,
                                "source_id": source_id,
                                "retrieval_method": "wikipedia_disambiguation",
                                "relevance": self._calculate_relevance(
                                    page.title, original_topic
                                ),
                            }
                        )
                except Exception:
                    pass  # Skip if disambiguation fails

        except wikipedia.exceptions.PageError:
            logger.debug(f"Wikipedia page not found: '{page_title}'")
        except Exception as e:
            logger.warning(f"Error retrieving page '{page_title}': {e}")

        return snippets

    def _get_top_sections(self, page, topic: str, max_sections: int) -> List[tuple]:
        """Get most relevant sections quickly without extensive processing."""
        relevant_sections = []
        topic_words = set(topic.lower().replace("_", " ").split())

        # Only check first few sections for speed
        sections_to_check = (
            page.sections[: max_sections * 2] if hasattr(page, "sections") else []
        )

        for section_title in sections_to_check:
            try:
                content = page.section(section_title)
                if not content or len(content.strip()) < 100:
                    continue

                # Quick relevance check
                relevance = self._quick_relevance_score(section_title, topic_words)
                relevant_sections.append((section_title, content.strip(), relevance))

                # Stop early if we have enough good sections
                if len(relevant_sections) >= max_sections:
                    break

            except Exception:
                continue  # Skip problematic sections

        # Sort by relevance and return top sections
        relevant_sections.sort(key=lambda x: x[2], reverse=True)
        return [
            (title, content) for title, content, _ in relevant_sections[:max_sections]
        ]

    def _quick_relevance_score(self, text: str, topic_words: set) -> float:
        """Fast relevance calculation."""
        if not text or not topic_words:
            return 0.0

        text_words = set(text.lower().split())
        intersection = len(text_words.intersection(topic_words))
        return intersection / max(len(topic_words), 1)

    def _calculate_relevance(self, text: str, topic: str) -> float:
        """Calculate relevance score between text and topic."""
        if not text or not topic:
            return 0.0

        text_words = set(text.lower().replace("_", " ").split())
        topic_words = set(topic.lower().replace("_", " ").split())

        if not topic_words:
            return 0.5

        # Jaccard similarity
        intersection = len(text_words.intersection(topic_words))
        union = len(text_words.union(topic_words))

        return intersection / union if union > 0 else 0.0
