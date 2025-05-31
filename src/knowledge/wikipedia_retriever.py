import wikipedia
import hashlib
import json
import os
from typing import List, Dict, Any, Optional

class WikipediaRetriever:
    """
    Specialized retriever for Wikipedia content.
    
    This class handles the complexities of Wikipedia API interaction, including
    caching, error handling, and content extraction. It serves as both a standalone
    component and a building block for other retrieval systems.
    
    Teaching note: This class demonstrates the single responsibility principle -
    it has one job (Wikipedia retrieval) and does it well.
    """
    
    def __init__(self, cache_dir: str = "data/wiki_cache"):
        """Initialize Wikipedia retriever with intelligent caching."""
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        # Configure Wikipedia API for respectful usage
        wikipedia.set_rate_limiting(True)
        
        print(f"WikipediaRetriever initialized with cache at: {cache_dir}")

    def get_wiki_content(
        self, topic: str, max_articles: int = 3, max_sections: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Retrieve Wikipedia content with intelligent caching.
        
        This method demonstrates how to build robust, cacheable retrieval systems.
        The caching strategy improves performance and reduces API load while maintaining
        fresh content when needed.
        
        Args:
            topic: The topic to search for
            max_articles: Maximum number of Wikipedia articles to retrieve
            max_sections: Maximum sections to extract per article
            
        Returns:
            List of content snippets with metadata
        """
        # Create deterministic cache key based on parameters
        # This ensures we get the same cached result for identical requests
        cache_key = hashlib.md5(
            f"{topic}_{max_articles}_{max_sections}".encode()
        ).hexdigest()
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")

        # Check cache first - this is a common optimization pattern
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    cached_data = json.load(f)
                print(f"Loaded cached Wikipedia data for '{topic}'")
                return cached_data
            except (json.JSONDecodeError, FileNotFoundError):
                print(f"Cache file corrupted for '{topic}', retrieving fresh data")

        # Retrieve fresh data if not in cache
        snippets = self._retrieve_from_wiki(topic, max_articles, max_sections)

        # Cache the results for future use
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(snippets, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Warning: Failed to cache Wikipedia data: {e}")

        return snippets

    def _retrieve_from_wiki(
        self, topic: str, max_articles: int = 3, max_sections: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Core Wikipedia retrieval logic with robust error handling.
        
        This method demonstrates how to handle the complexities of external API
        interaction, including disambiguation, search variations, and graceful
        error recovery.
        """
        try:
            # Generate search variations to improve hit rate
            # This technique helps handle different ways people might phrase topics
            search_variations = [
                topic,
                topic.replace("-", " "),  # "machine-learning" → "machine learning"
                topic.replace(" ", "-"),  # "machine learning" → "machine-learning"
                " ".join(topic.split()[:-1]) if len(topic.split()) > 1 else topic,  # Remove last word
                topic.split()[0] if len(topic.split()) > 1 else topic,  # Just first word
            ]

            # Try each variation until we find results
            all_search_results = []
            for variation in search_variations:
                try:
                    results = wikipedia.search(variation, results=max_articles)
                    all_search_results.extend(results)
                    if results:  # If we found something, we can stop trying variations
                        break
                except Exception as e:
                    print(f"Search failed for variation '{variation}': {e}")
                    continue

            # Remove duplicates while preserving order
            search_results = []
            for result in all_search_results:
                if result not in search_results:
                    search_results.append(result)

            search_results = search_results[:max_articles]
            
            if not search_results:
                print(f"Warning: Could not find any Wikipedia pages for '{topic}'")
                return []

            # Extract content from found pages
            snippets = []
            source_id = 0

            for i, page_title in enumerate(search_results):
                if i >= max_articles:
                    break

                try:
                    page = wikipedia.page(page_title, auto_suggest=True)

                    # Add page summary - this is usually the most important content
                    snippets.append({
                        "title": page.title,
                        "section": "Summary",
                        "content": page.summary,
                        "url": page.url,
                        "source_id": source_id,
                        "retrieval_method": "wikipedia_summary"
                    })
                    source_id += 1

                    # Extract detailed sections
                    section_count = 0
                    for section_title in page.sections:
                        if section_count >= max_sections:
                            break

                        try:
                            content = page.section(section_title)
                            # Filter out very short sections that don't provide value
                            if content and len(content) > 100:
                                snippets.append({
                                    "title": page.title,
                                    "section": section_title,
                                    "content": content,
                                    "url": page.url,
                                    "source_id": source_id,
                                    "retrieval_method": "wikipedia_section"
                                })
                                section_count += 1
                                source_id += 1
                        except Exception as e:
                            print(f"Error extracting section '{section_title}': {e}")

                except wikipedia.exceptions.DisambiguationError as e:
                    # Handle disambiguation gracefully by trying the first option
                    if e.options and len(e.options) > 0:
                        try:
                            page = wikipedia.page(e.options[0], auto_suggest=False)
                            snippets.append({
                                "title": page.title,
                                "section": "Summary",
                                "content": page.summary,
                                "url": page.url,
                                "source_id": source_id,
                                "retrieval_method": "wikipedia_disambiguation"
                            })
                            source_id += 1
                        except Exception as inner_e:
                            print(f"Failed to get disambiguation option for '{page_title}': {inner_e}")
                except Exception as e:
                    print(f"Error retrieving page '{page_title}': {e}")

            print(f"Retrieved {len(snippets)} Wikipedia snippets for '{topic}'")
            return snippets

        except Exception as e:
            print(f"Wikipedia retrieval failed for '{topic}': {e}")
            return []
