import logging
import wikipedia
from sentence_transformers import SentenceTransformer
from typing import Dict, List

from src.config.config_context import ConfigContext
from src.retrieval.rms.base_retriever import BaseRetriever

logger = logging.getLogger(__name__)


class WikiRM(BaseRetriever):

    def __init__(
        self,
        format_type: str = "rag",
        cache_dir: str = "data/wiki_cache",
        cache_results: bool = True,
    ):
        # Use passed config or default
        self.retrieval_config = ConfigContext.get_retrieval_config()

        # Initialize parent class with caching and config
        super().__init__(
            cache_dir=cache_dir,
            cache_results=cache_results,
            config=self.retrieval_config,
            format_type=format_type,
        )

        self.format_type = format_type
        self.results_per_query = self.retrieval_config.results_per_query

        # Add semantic filtering setup
        self.semantic_enabled = self.retrieval_config.semantic_filtering_enabled
        self.embedding_model = None
        self._extracted_pages = set()

        if self.semantic_enabled:
            try:
                self.embedding_model = SentenceTransformer(
                    self.retrieval_config.embedding_model,
                    device="cpu",  # Use "cuda" if available
                )
            except Exception as e:
                logger.warning(
                    f"Failed to load embedding model: {e}. Disabling semantic filtering."
                )
                self.semantic_enabled = False

        # Setup Wikipedia client
        wikipedia.set_rate_limiting(True)
        wikipedia.set_lang("en")

    def _retrieve_article(
        self, query: str, topic: str = None, max_results: int = None
    ) -> List[Dict]:
        """Search for a single query and return structured results."""
        # Use config default if max_results not provided
        if max_results is None:
            max_results = self.retrieval_config.results_per_query

        # Check cache first
        cache_key = self._get_cache_key(query)
        cached_results = self._load_from_cache(cache_key)
        if cached_results:
            return cached_results

        results = []

        try:
            # Search Wikipedia
            wiki_response = wikipedia.search(
                query, results=max_results, suggestion=True
            )

            if isinstance(wiki_response, tuple):
                search_results, suggestion = wiki_response
            else:
                search_results = wiki_response

            new_pages = [
                title for title in search_results if title not in self._extracted_pages
            ]
            self._extracted_pages.update(new_pages)

            if not search_results:
                logger.warning(f"No Wikipedia results found for query: '{query}'")
                return []

            # Process pages in parallel for speed
            results = []
            for title in new_pages:
                content = self._extract_page_content(title)
                results.extend(content)

        except Exception as e:
            logger.error(f"Wikipedia search failed for '{query}': {e}")
            return []

        # Cache the results
        self._save_to_cache(cache_key, results)

        return results

    def _extract_page_content(self, page_title: str) -> List[Dict]:
        """Extract content from a single Wikipedia page."""
        results = []

        try:
            page = wikipedia.page(page_title, auto_suggest=False)

            content_items = []
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

            # Process all content items - filtering happens later in BaseRetriever
            for i, item in enumerate(content_items):
                try:
                    if content_type == "sections":
                        content = page.section(item)
                        section_name = item
                    else:  # chunks
                        content = item  # item is already the content chunk
                        section_name = f"Content Part {i + 1}"

                    if (
                        content
                        and len(content.strip())
                        > self.retrieval_config.passage_min_length
                    ):
                        result_item = {
                            "title": page.title,
                            "section": section_name,
                            "snippets": [content.strip()],
                            "description": self._generate_description(
                                content.strip(), f"{page.title} - {section_name}"
                            ),
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

        except Exception as e:
            logger.error(f"Failed to extract content from page '{page_title}': {e}")

        return results

    def _create_content_chunks(self, content: str) -> List[str]:
        """Create content chunks from full page content."""
        from src.utils.content.chunker import ContentChunker

        # Chunk content for processing
        chunker = ContentChunker(
            chunk_size=self.retrieval_config.passage_max_length, overlap=100
        )
        return chunker.chunk_text(content)
