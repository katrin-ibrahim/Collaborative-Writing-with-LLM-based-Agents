import hashlib
import time

import json
import logging
import os
import re
import wikipedia
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class WikipediaRetriever:
    """Enhanced Wikipedia retriever with better search strategies and content extraction."""

    def __init__(self, cache_dir: str = "data/wiki_cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        wikipedia.set_rate_limiting(True)
        wikipedia.set_lang("en")
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 100ms between requests
        logger.info(
            f"Enhanced WikipediaRetriever initialized with cache at: {cache_dir}"
        )

    def _rate_limit(self):
        """Simple rate limiting to avoid overwhelming Wikipedia API."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last)
        self.last_request_time = time.time()

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
        self, topic: str, max_articles: int = 5, max_sections: int = 8
    ) -> List[Dict[str, Any]]:
        """Retrieve Wikipedia content with enhanced search strategies."""
        cleaned_topic = self._clean_query(topic)

        cache_key = hashlib.md5(
            f"{cleaned_topic}_{max_articles}_{max_sections}_enhanced_v2".encode()
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

        snippets = self._retrieve_with_enhanced_search(
            cleaned_topic, max_articles, max_sections
        )

        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(snippets, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"Failed to cache Wikipedia data: {e}")

        return snippets

    def _retrieve_with_enhanced_search(
        self, topic: str, max_articles: int, max_sections: int
    ) -> List[Dict[str, Any]]:
        """Enhanced retrieval with better search strategies."""
        if not topic or len(topic.strip()) < 2:
            logger.warning(f"Topic too short or empty: '{topic}'")
            return []

        search_queries = self._generate_search_queries(topic)
        all_pages = []

        for query in search_queries:
            try:
                self._rate_limit()
                logger.debug(f"Searching Wikipedia for: '{query}'")
                results = wikipedia.search(query, results=max_articles)

                for result in results:
                    if result not in all_pages:
                        all_pages.append(result)
                        if len(all_pages) >= max_articles * 2:
                            break

                if len(all_pages) >= max_articles:
                    break

            except Exception as e:
                logger.warning(f"Search failed for query '{query}': {e}")
                continue

        if not all_pages:
            logger.warning(f"No Wikipedia pages found for '{topic}'")
            return []

        return self._extract_content_from_pages(
            all_pages[:max_articles], max_sections, topic
        )

    def _generate_search_queries(self, topic: str) -> List[str]:
        """Generate multiple search query variations for better coverage."""
        queries = [topic]

        # Clean topic variations
        clean_topic = topic.replace("_", " ").replace("-", " ").strip()
        if clean_topic != topic and clean_topic:
            queries.append(clean_topic)

        # Extract main subject from compound topics
        words = clean_topic.split()
        if len(words) > 1:
            # Try first word (often the main subject)
            if len(words[0]) > 2:
                queries.append(words[0])

            # Try first two words
            if len(words) > 2:
                queries.append(" ".join(words[:2]))

        # Remove duplicates while preserving order
        unique_queries = []
        for q in queries:
            if q and q not in unique_queries and len(q.strip()) > 1:
                unique_queries.append(q)

        return unique_queries[:3]  # Limit to avoid too many API calls

    def _extract_content_from_pages(
        self, page_titles: List[str], max_sections: int, original_topic: str
    ) -> List[Dict[str, Any]]:
        """Extract content from Wikipedia pages with better error handling."""
        snippets = []
        source_id = 0

        for page_title in page_titles:
            try:
                self._rate_limit()
                logger.debug(f"Extracting content from page: '{page_title}'")

                page = wikipedia.page(page_title, auto_suggest=True)

                # Add summary (always valuable)
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
                    source_id += 1

                # Extract relevant sections
                if hasattr(page, "sections") and page.sections:
                    relevant_sections = self._get_relevant_sections(
                        page, original_topic, max_sections
                    )
                    for section_title, content in relevant_sections:
                        if content and len(content.strip()) > 100:
                            snippets.append(
                                {
                                    "title": page.title,
                                    "section": section_title,
                                    "content": content.strip(),
                                    "url": page.url,
                                    "source_id": source_id,
                                    "retrieval_method": "wikipedia_section",
                                    "relevance": self._calculate_relevance(
                                        section_title, original_topic
                                    ),
                                }
                            )
                            source_id += 1

            except wikipedia.exceptions.DisambiguationError as e:
                # Try first disambiguation option
                if e.options and len(e.options) > 0:
                    try:
                        self._rate_limit()
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
                            source_id += 1
                    except Exception as inner_e:
                        logger.warning(
                            f"Failed to resolve disambiguation for '{page_title}': {inner_e}"
                        )
            except wikipedia.exceptions.PageError:
                logger.warning(f"Wikipedia page not found: '{page_title}'")
            except Exception as e:
                logger.warning(f"Error retrieving page '{page_title}': {e}")

        # Sort by relevance and return best results
        snippets.sort(key=lambda x: x.get("relevance", 0), reverse=True)
        logger.info(
            f"Retrieved {len(snippets)} Wikipedia snippets for '{original_topic}'"
        )
        return snippets

    def _get_relevant_sections(
        self, page, topic: str, max_sections: int
    ) -> List[tuple]:
        """Get most relevant sections from a Wikipedia page."""
        relevant_sections = []
        topic_words = set(topic.lower().replace("_", " ").split())

        sections_to_check = (
            page.sections[: max_sections * 3] if hasattr(page, "sections") else []
        )

        for section_title in sections_to_check:
            try:
                self._rate_limit()
                content = page.section(section_title)
                if not content or len(content.strip()) < 100:
                    continue

                relevance = self._calculate_section_relevance(
                    section_title, content, topic_words
                )
                relevant_sections.append((section_title, content.strip(), relevance))

            except Exception as e:
                logger.debug(f"Error extracting section '{section_title}': {e}")
                continue

        # Sort by relevance and return top sections
        relevant_sections.sort(key=lambda x: x[2], reverse=True)
        return [
            (title, content) for title, content, _ in relevant_sections[:max_sections]
        ]

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

    def _calculate_section_relevance(
        self, section_title: str, content: str, topic_words: set
    ) -> float:
        """Calculate relevance of a section to the topic."""
        if not section_title or not content or not topic_words:
            return 0.0

        # Title relevance (weighted more heavily)
        title_words = set(section_title.lower().split())
        title_relevance = len(title_words.intersection(topic_words)) / max(
            len(topic_words), 1
        )

        # Content relevance (check first 300 chars for efficiency)
        content_sample = content[:300].lower()
        content_words = set(content_sample.split())
        content_relevance = len(content_words.intersection(topic_words)) / max(
            len(topic_words), 1
        )

        # Weighted combination
        return (title_relevance * 0.7) + (content_relevance * 0.3)
