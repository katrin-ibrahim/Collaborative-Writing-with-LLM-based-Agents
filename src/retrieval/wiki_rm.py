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

try:
    import numpy as np
    from sentence_transformers import SentenceTransformer

    SEMANTIC_AVAILABLE = True
except ImportError:
    SEMANTIC_AVAILABLE = False

from src.config.retrieval_config import DEFAULT_RETRIEVAL_CONFIG

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

        # Add semantic filtering setup
        self.semantic_enabled = (
            DEFAULT_RETRIEVAL_CONFIG.semantic_filtering_enabled and SEMANTIC_AVAILABLE
        )
        self.embedding_model = None
        self.semantic_cache = {}

        if self.semantic_enabled:
            try:
                self.embedding_model = SentenceTransformer(
                    DEFAULT_RETRIEVAL_CONFIG.embedding_model
                )
                logger.info(
                    f"Semantic filtering enabled with {DEFAULT_RETRIEVAL_CONFIG.embedding_model}"
                )
            except Exception as e:
                logger.warning(
                    f"Failed to load embedding model: {e}. Disabling semantic filtering."
                )
                self.semantic_enabled = False

        # Setup Wikipedia client
        os.makedirs(cache_dir, exist_ok=True)
        wikipedia.set_rate_limiting(True)
        wikipedia.set_lang("en")

        logger.info(
            f"WikiRM initialized (articles={max_articles}, sections={max_sections}, semantic={self.semantic_enabled})"
        )

    def __call__(self, *args, **kwargs):
        """
        Allow direct call to search method for convenience.
        """
        return self.search(*args, **kwargs)

    def search(
        self,
        queries: Union[str, List[str]],
        max_results: int = 8,
        format_type: str = "rag",
        topic: str = None,
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
            query_results = self._search_single_query(topic, query, format_type)
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

    def _search_single_query(
        self, topic: str, query: str, format_type: str
    ) -> List[Dict]:
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
                    executor.submit(self._extract_page_content, title, topic): title
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

            # Get all available content items
            if hasattr(page, "sections") and page.sections:
                content_items = page.sections
                content_type = "sections"
            elif hasattr(page, "content") and page.content:
                # Create chunks from full content
                content_items = self._create_content_chunks(page.content)
                content_type = "chunks"
            else:
                return results

            # Apply unified semantic filtering
            relevant_items = self._get_relevant_content(content_items, original_query)

            # Process the filtered items
            for i, item in enumerate(relevant_items):
                try:
                    if content_type == "sections":
                        content = page.section(item)
                        section_name = item
                    else:  # chunks
                        content = item  # item is already the content chunk
                        section_name = f"Content Part {i + 1}"

                    if content and len(content.strip()) > 100:
                        results.append(
                            {
                                "title": page.title,
                                "section": section_name,
                                "content": content.strip()[:2000],
                                "url": page.url,
                                "source": "wikipedia",
                            }
                        )
                except Exception:
                    continue  # Skip problematic sections/chunks

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

    def _get_relevant_content(self, content_items: List[str], query: str) -> List[str]:
        """Get the most relevant content items based on semantic similarity to query."""
        if not content_items:
            return []

        # Calculate similarities directly with content items
        similarities = self._calculate_section_similarities(query, content_items)

        # Filter by threshold and take top items
        relevant_items = [
            item
            for item, score in similarities
            if score >= DEFAULT_RETRIEVAL_CONFIG.similarity_threshold
        ]

        # Take top items (with minimum guarantee)
        if len(relevant_items) < 2:
            relevant_items = [s[0] for s in similarities[: self.max_sections]]

        return relevant_items[: self.max_sections]

    def _create_content_chunks(self, content: str) -> List[str]:
        """Create meaningful chunks from full page content."""
        # Split by paragraphs first
        paragraphs = [p.strip() for p in content.split("\n\n") if len(p.strip()) > 50]

        if not paragraphs:
            # Fallback: split by sentences if no paragraphs
            sentences = [
                s.strip() + "." for s in content.split(".") if len(s.strip()) > 20
            ]
            if sentences:
                paragraphs = sentences
            else:
                return [content[:2000]]

        # Combine paragraphs into ~1000-1500 character chunks
        chunks = []
        current_chunk = ""

        for para in paragraphs:
            if len(current_chunk + para) > 1500 and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = para
            else:
                current_chunk += "\n\n" + para if current_chunk else para

        # Add remaining content
        if current_chunk:
            chunks.append(current_chunk.strip())

        # Ensure we don't have too many tiny chunks
        if len(chunks) > 10:
            # Merge small chunks
            merged_chunks = []
            temp_chunk = ""
            for chunk in chunks:
                if len(temp_chunk + chunk) <= 2000:
                    temp_chunk += "\n\n" + chunk if temp_chunk else chunk
                else:
                    if temp_chunk:
                        merged_chunks.append(temp_chunk)
                    temp_chunk = chunk
            if temp_chunk:
                merged_chunks.append(temp_chunk)
            chunks = merged_chunks

        return chunks

    def _calculate_section_similarities(
        self, query: str, content_items: List[str]
    ) -> List[tuple]:
        """Calculate semantic similarities between query and content items."""
        from numpy.linalg import norm

        if not content_items:
            return []

        # Check cache for query embedding
        cache_key = hash(query)
        if cache_key in self.semantic_cache:
            query_embedding = self.semantic_cache[cache_key]
        else:
            query_embedding = self.embedding_model.encode([query])[0]

            # Cache management
            if len(self.semantic_cache) >= DEFAULT_RETRIEVAL_CONFIG.semantic_cache_size:
                oldest_key = next(iter(self.semantic_cache))
                del self.semantic_cache[oldest_key]

            self.semantic_cache[cache_key] = query_embedding

        # Encode all content items directly
        item_embeddings = self.embedding_model.encode(content_items)

        # Calculate similarities
        similarities = []
        for item, embedding in zip(content_items, item_embeddings):
            # Cosine similarity
            similarity = np.dot(query_embedding, embedding) / (
                norm(query_embedding) * norm(embedding)
            )
            similarities.append((item, float(similarity)))

        # Sort by similarity score (highest first)
        return sorted(similarities, key=lambda x: x[1], reverse=True)

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
