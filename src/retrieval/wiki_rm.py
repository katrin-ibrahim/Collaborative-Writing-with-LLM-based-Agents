"""
Clean, simplified Wikipedia retriever.
Eliminates the wrapper layers and provides direct, efficient Wikipedia access.
"""

import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed

import json
import logging
import numpy as np
import os
import wikipedia
from sentence_transformers import SentenceTransformer
from typing import Dict, List, Optional

from src.config.retrieval_config import DEFAULT_RETRIEVAL_CONFIG, RetrievalConfig
from src.retrieval.base_retriever import BaseRetriever

logger = logging.getLogger(__name__)


class WikiRM(BaseRetriever):
    """
    Unified Wikipedia Retrieval Manager.

    Clean, simplified Wikipedia retriever that eliminates the wrapper hell.
    Single implementation that can output both RAG and STORM formats
    without multiple conversion layers.

    This class serves as both the concrete Wikipedia implementation and
    the unified retrieval interface, eliminating the need for separate
    RM wrapper classes.
    """

    def __init__(
        self,
        max_articles: Optional[int] = None,
        max_sections: Optional[int] = None,
        format_type: str = "rag",
        cache_dir: str = "data/wiki_cache",
        cache_results: bool = True,
        config: Optional["RetrievalConfig"] = None,
    ):
        # Use passed config or default
        retrieval_config = config or DEFAULT_RETRIEVAL_CONFIG

        self.max_articles = (
            max_articles
            if max_articles is not None
            else retrieval_config.results_per_query
        )
        self.max_sections = (
            max_sections
            if max_sections is not None
            else retrieval_config.max_content_pieces
        )
        self.format_type = format_type
        self.cache_dir = cache_dir
        self.cache_results = cache_results
        self._result_cache = {} if cache_results else None

        # Add semantic filtering setup
        self.semantic_enabled = retrieval_config.semantic_filtering_enabled
        self.embedding_model = None
        self.semantic_cache = {}

        if self.semantic_enabled:
            try:
                self.embedding_model = SentenceTransformer(
                    retrieval_config.embedding_model
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

    def __call__(self, query=None, k=None, *args, **kwargs):
        """
        Allow direct call to search method for convenience.
        Handles STORM's calling convention: search_rm(query, k=5)
        """
        if query is not None:
            # STORM-style call: search_rm(query, k=5)
            max_results = k if k is not None else kwargs.get("max_results", None)
            return self.search(queries=query, max_results=max_results, **kwargs)
        else:
            # Standard call: search_rm(queries=..., max_results=...)
            return self.search(*args, **kwargs)

    def search(
        self,
        *args,
        max_results: int = None,
        format_type: str = None,
        topic: str = None,
        deduplicate: bool = True,
        query: str = None,
        **kwargs,
    ) -> List:
        """
        Single search method that handles both RAG and STORM formats.
        Eliminates the need for separate search() and retrieve() methods.
        """
        # Debug logging to understand STORM's calling pattern
        logger.debug(
            f"WikiRM.search called with: args={args}, query={query}, kwargs={kwargs}"
        )

        # Handle various STORM calling conventions
        if query is not None:
            # Keyword argument: search(query="...")
            query = query
        if args:
            # Positional arguments: search("query", 5) or search("query")
            query = args[0] if len(args) > 0 else None
            if max_results is None and len(args) > 1:
                max_results = args[1]
        elif "query_or_queries" in kwargs and kwargs["query_or_queries"] is not None:
            # Query in kwargs
            query = kwargs.pop("query_or_queries", None)
        elif len(args) == 0 and not kwargs:
            # Called with no arguments - this might be a STORM initialization call
            logger.warning(
                "WikiRM.search called with no arguments - returning empty list"
            )
            return []

        if query is None:
            logger.error(
                f"WikiRM.search called with no valid query. args={args}, kwargs={kwargs}"
            )
            raise ValueError(
                "No valid query provided. Expected 'query', 'query' parameter, or positional argument."
            )

        max_results = (
            max_results
            if max_results is not None
            else DEFAULT_RETRIEVAL_CONFIG.results_per_query
        )
        format_type = format_type if format_type else self.format_type

        # Normalize input
        if isinstance(query, str):
            query_list = [query]
        else:
            query_list = list(query)

        # Check result cache first
        cache_key = None
        if self.cache_results:
            cache_key = self._generate_result_cache_key(
                query_list, max_results, format_type
            )
            if cache_key in self._result_cache:
                logger.debug("Returning cached search results")
                return self._result_cache[cache_key]

        # Collect all results
        all_results = []

        for query in query_list:
            query_results = self._search_single_query(topic, query, format_type)
            all_results.extend(query_results)

            # Early termination if we have enough results
            if len(all_results) >= max_results:
                break

        # Deduplicate if requested
        if deduplicate:
            all_results = self._deduplicate_results(all_results, format_type)

        # Return in requested format
        if format_type == "rag":
            # For RAG: return list of content strings
            passages = []
            for result in all_results[:max_results]:
                if isinstance(result, dict) and "snippets" in result:
                    passages.append(result["snippets"])
                elif isinstance(result, str):
                    passages.append(result)
            final_results = passages
        elif format_type == "storm":
            # For STORM: return dict with 'snippets' key
            storm_results = []
            for result in all_results[:max_results]:
                if isinstance(result, dict):
                    storm_result = {
                        "url": result.get("url", ""),
                        "title": result.get("title", ""),
                        "description": result.get("snippets", "")[:200],
                        "snippets": [result.get("snippets", "")],  # Content as LIST
                    }
                    storm_results.append(storm_result)
            final_results = storm_results
        else:
            raise ValueError(f"Unknown format_type: {format_type}")

        # Cache results
        if self.cache_results and cache_key:
            self._result_cache[cache_key] = final_results

        return final_results

    @staticmethod
    def transform_storm_query_to_keywords(query: str, topic: str = None) -> str:
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

    def _search_single_query(
        self, topic: str, query: str, format_type: str
    ) -> List[Dict]:
        """Search for a single query and return structured results."""
        # Check cache first
        if self.format_type == "storm":
            query = self.transform_storm_query_to_keywords(query, topic)

        cache_key = self._get_cache_key(query)
        cached_results = self._load_from_cache(cache_key)
        if cached_results:
            return cached_results

        results = []

        try:
            # Search Wikipedia
            search_results = wikipedia.search(
                query, results=self.max_articles, suggestion=False
            )

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
            return []

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

            if not relevant_items:
                # return summary if no relevant items found
                summary = page.summary.strip()
                if summary and len(summary) > 100:
                    results.append(
                        {
                            "title": page.title,
                            "section": "Content",
                            "snippets": summary[:2000],
                            "url": page.url,
                            "source": "wikipedia",
                        }
                    )
            else:
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
                            result_item = {
                                "title": page.title,
                                "section": section_name,
                                "snippets": content.strip()[:2000],
                                "url": page.url,
                                "source": "wikipedia",
                            }
                            results.append(result_item)
                            logger.debug(f"Added result: {page.title} - {section_name}")
                        else:
                            logger.debug(
                                f"Skipped item {i}: content too short ({len(content) if content else 0} chars)"
                            )
                    except Exception as e:
                        logger.warning(
                            f"Error processing item {i} for page '{page_title}': {e}"
                        )
                        continue  # Skip problematic sections/chunks

                # logger.info(
                #     f"Page '{page_title}' produced {len(results)} final results"
                # )

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

        if query is None or not query.strip():
            # logger.warning("Empty query provided for semantic similarity calculation.")
            return []

        try:
            # Check cache for query embedding
            cache_key = hash(query)
            if cache_key in self.semantic_cache:
                query_embedding = self.semantic_cache[cache_key]
            else:
                query_embedding = self.embedding_model.encode([query])[0]
                if (
                    len(self.semantic_cache)
                    >= DEFAULT_RETRIEVAL_CONFIG.semantic_cache_size
                ):
                    oldest_key = next(iter(self.semantic_cache))
                    del self.semantic_cache[oldest_key]

                self.semantic_cache[cache_key] = query_embedding
        except Exception as e:
            logger.error(f"Failed to encode query '{query}': {e}")
            return []

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

    def clear_cache(self):
        """Clear both result cache and file cache."""
        if self.cache_results:
            self._result_cache.clear()
            logger.info("Result cache cleared")

        # Clear semantic cache
        self.semantic_cache.clear()
        logger.info("Semantic cache cleared")

    def _generate_result_cache_key(
        self, queries: List[str], max_results: int, format_type: str
    ) -> str:
        """Generate a cache key for search results."""
        key_data = f"{';'.join(sorted(queries))}:{max_results}:{format_type}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def _deduplicate_results(self, results: List, format_type: str) -> List:
        """Remove duplicate results based on format type."""
        # At this stage, results are always dictionaries from Wikipedia
        # regardless of the final format_type, so we deduplicate based on dict structure
        seen_urls = set()
        unique_results = []

        for result in results:
            if isinstance(result, dict):
                url = result.get("url", "")
                content = result.get("snippets", "")

                # Use URL if available, otherwise use content hash
                identifier = (
                    url
                    if url
                    else hash(content.strip().lower()) if content else hash(str(result))
                )

                if identifier not in seen_urls:
                    seen_urls.add(identifier)
                    unique_results.append(result)
            elif isinstance(result, str):
                # Handle string results (shouldn't happen at this stage, but just in case)
                content_hash = hash(result.strip().lower())
                if content_hash not in seen_urls:
                    seen_urls.add(content_hash)
                    unique_results.append(result)

        logger.info(f"Deduplication: {len(results)} → {len(unique_results)} results")
        return unique_results
