import wikipedia
import hashlib
import json
import os
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class WikipediaRetriever:
    """Enhanced Wikipedia retriever with better search strategies and content extraction."""
    
    def __init__(self, cache_dir: str = "data/wiki_cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        wikipedia.set_rate_limiting(True)
        logger.info(f"Enhanced WikipediaRetriever initialized with cache at: {cache_dir}")

    def get_wiki_content(self, topic: str, max_articles: int = 5, max_sections: int = 8) -> List[Dict[str, Any]]:
        """Retrieve Wikipedia content with enhanced search strategies."""
        cache_key = hashlib.md5(f"{topic}_{max_articles}_{max_sections}_enhanced".encode()).hexdigest()
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")

        if os.path.exists(cache_file):
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    cached_data = json.load(f)
                logger.info(f"Loaded cached Wikipedia data for '{topic}'")
                return cached_data
            except (json.JSONDecodeError, FileNotFoundError):
                logger.warning(f"Cache file corrupted for '{topic}', retrieving fresh data")

        snippets = self._retrieve_with_enhanced_search(topic, max_articles, max_sections)

        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(snippets, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"Failed to cache Wikipedia data: {e}")

        return snippets

    def _retrieve_with_enhanced_search(self, topic: str, max_articles: int, max_sections: int) -> List[Dict[str, Any]]:
        """Enhanced retrieval with better search strategies."""
        search_queries = self._generate_search_queries(topic)
        all_pages = []
        
        for query in search_queries:
            try:
                results = wikipedia.search(query, results=max_articles)
                for result in results:
                    if result not in all_pages:
                        all_pages.append(result)
                        if len(all_pages) >= max_articles * 2:  # Get extra candidates
                            break
                if all_pages:  # If we found good results, don't try more variations
                    break
            except Exception as e:
                logger.warning(f"Search failed for query '{query}': {e}")
                continue

        if not all_pages:
            logger.warning(f"No Wikipedia pages found for '{topic}'")
            return []

        # Extract content from best pages
        return self._extract_content_from_pages(all_pages[:max_articles], max_sections, topic)

    def _generate_search_queries(self, topic: str) -> List[str]:
        """Generate multiple search query variations for better coverage."""
        queries = [topic]
        
        # Remove common prefixes/suffixes that might limit results
        clean_topic = topic.replace("_", " ").replace("-", " ")
        if clean_topic != topic:
            queries.append(clean_topic)
        
        # Try variations for compound topics
        words = clean_topic.split()
        if len(words) > 1:
            # Try just the main subject (first part)
            queries.append(words[0])
            # Try without last word (often descriptive)
            queries.append(" ".join(words[:-1]))
            # Try key terms
            if len(words) > 2:
                queries.append(" ".join(words[:2]))
        
        # Remove duplicates while preserving order
        unique_queries = []
        for q in queries:
            if q not in unique_queries:
                unique_queries.append(q)
        
        return unique_queries[:4]  # Limit to avoid too many API calls

    def _extract_content_from_pages(self, page_titles: List[str], max_sections: int, original_topic: str) -> List[Dict[str, Any]]:
        """Extract content from Wikipedia pages with better filtering."""
        snippets = []
        source_id = 0

        for page_title in page_titles:
            try:
                page = wikipedia.page(page_title, auto_suggest=True)
                
                # Add summary (always valuable)
                if page.summary and len(page.summary) > 50:
                    snippets.append({
                        "title": page.title,
                        "section": "Summary",
                        "content": page.summary,
                        "url": page.url,
                        "source_id": source_id,
                        "retrieval_method": "wikipedia_summary",
                        "relevance": self._calculate_relevance(page.title, original_topic)
                    })
                    source_id += 1

                # Extract most relevant sections
                relevant_sections = self._get_relevant_sections(page, original_topic, max_sections)
                for section_title, content in relevant_sections:
                    snippets.append({
                        "title": page.title,
                        "section": section_title,
                        "content": content,
                        "url": page.url,
                        "source_id": source_id,
                        "retrieval_method": "wikipedia_section",
                        "relevance": self._calculate_relevance(section_title, original_topic)
                    })
                    source_id += 1

            except wikipedia.exceptions.DisambiguationError as e:
                # Try first disambiguation option
                if e.options:
                    try:
                        page = wikipedia.page(e.options[0], auto_suggest=False)
                        if page.summary and len(page.summary) > 50:
                            snippets.append({
                                "title": page.title,
                                "section": "Summary (Disambiguated)",
                                "content": page.summary,
                                "url": page.url,
                                "source_id": source_id,
                                "retrieval_method": "wikipedia_disambiguation",
                                "relevance": self._calculate_relevance(page.title, original_topic)
                            })
                            source_id += 1
                    except Exception:
                        logger.warning(f"Failed to resolve disambiguation for '{page_title}'")
            except Exception as e:
                logger.warning(f"Error retrieving page '{page_title}': {e}")

        # Sort by relevance and return best results
        snippets.sort(key=lambda x: x.get("relevance", 0), reverse=True)
        logger.info(f"Retrieved {len(snippets)} Wikipedia snippets for '{original_topic}'")
        return snippets

    def _get_relevant_sections(self, page, topic: str, max_sections: int) -> List[tuple]:
        """Get most relevant sections from a Wikipedia page."""
        relevant_sections = []
        topic_words = set(topic.lower().replace("_", " ").split())
        
        for section_title in page.sections[:max_sections * 2]:  # Check more than we need
            try:
                content = page.section(section_title)
                if not content or len(content) < 100:  # Skip short sections
                    continue
                
                # Calculate relevance based on title and content
                relevance = self._calculate_section_relevance(section_title, content, topic_words)
                relevant_sections.append((section_title, content, relevance))
                
            except Exception as e:
                logger.debug(f"Error extracting section '{section_title}': {e}")
                continue
        
        # Sort by relevance and return top sections
        relevant_sections.sort(key=lambda x: x[2], reverse=True)
        return [(title, content) for title, content, _ in relevant_sections[:max_sections]]

    def _calculate_relevance(self, text: str, topic: str) -> float:
        """Calculate relevance score between text and topic."""
        text_words = set(text.lower().replace("_", " ").split())
        topic_words = set(topic.lower().replace("_", " ").split())
        
        if not topic_words:
            return 0.5
        
        # Jaccard similarity
        intersection = len(text_words.intersection(topic_words))
        union = len(text_words.union(topic_words))
        
        return intersection / union if union > 0 else 0.0

    def _calculate_section_relevance(self, section_title: str, content: str, topic_words: set) -> float:
        """Calculate relevance of a section to the topic."""
        # Title relevance (weighted more heavily)
        title_words = set(section_title.lower().split())
        title_relevance = len(title_words.intersection(topic_words)) / max(len(topic_words), 1)
        
        # Content relevance (check first 300 chars for efficiency)
        content_sample = content[:300].lower()
        content_words = set(content_sample.split())
        content_relevance = len(content_words.intersection(topic_words)) / max(len(topic_words), 1)
        
        # Weighted combination
        return (title_relevance * 0.7) + (content_relevance * 0.3)