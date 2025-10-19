import hashlib
from pathlib import Path
from urllib.parse import quote

import logging
import wikipedia
from matplotlib.pyplot import title
from sentence_transformers import SentenceTransformer
from typing import List, Optional

from src.config.config_context import ConfigContext
from src.retrieval.rms.base_retriever import BaseRetriever
from src.retrieval.utils.chunker import ContentChunker
from src.retrieval.utils.description_generator import DescriptionGenerator
from src.utils.data.models import ResearchChunk
from src.utils.experiment.experiment_setup import find_project_root

logger = logging.getLogger(__name__)


class WikiRM(BaseRetriever):
    """Wikipedia-based retrieval manager using live Wikipedia API.
    Returns structured ResearchChunk objects (Pydantic model).
    """

    def __init__(
        self,
        format_type: str = "rag",
        cache_dir: str = "data/wiki_cache",
        cache_results: bool = True,
    ):
        # Use passed config or default
        self.retrieval_config = ConfigContext.get_retrieval_config()
        if self.retrieval_config is None:
            raise RuntimeError(
                "Retrieval config is None. Ensure ConfigContext is properly initialized before using FaissRM."
            )

        # Automatically resolve cache_dir to absolute path from project root
        project_root_path = Path(find_project_root())
        if not Path(cache_dir).is_absolute():
            cache_dir = str(project_root_path / cache_dir)

        logger.debug(f"WikiRM using cache_dir for results: {cache_dir}")

        self.embedding_model_name = self.retrieval_config.embedding_model

        # Store cache_dir for _load_index method
        self.cache_dir = cache_dir
        # Initialize parent class with absolute cache path
        super().__init__(
            cache_dir=cache_dir,
            cache_results=cache_results,
            config=self.retrieval_config,
        )

        self.format_type = format_type
        self.results_per_query = self.retrieval_config.results_per_query

        self.encoder = SentenceTransformer(self.embedding_model_name, device="cpu")
        self.chunker = ContentChunker()
        self.description_generator = DescriptionGenerator()

        self._extracted_pages = set()

        # Setup Wikipedia client
        wikipedia.set_rate_limiting(True)
        wikipedia.set_lang("en")

    def _retrieve_article(
        self,
        query: str,
        topic: Optional[str] = None,
        max_results: Optional[int] = None,
        **kwargs,
    ) -> List[ResearchChunk]:
        """Search for a single query and return structured results."""
        # Use config default if max_results not provided
        if max_results is None:
            if self.retrieval_config is None:
                raise RuntimeError(
                    "retrieval_config is None. Cannot determine results_per_query."
                )
            max_results = self.retrieval_config.results_per_query

        results: List[ResearchChunk] = []

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

        return results

    def _extract_page_content(self, page_title: str) -> List[ResearchChunk]:
        """Extract content from a single Wikipedia page."""
        results = []

        try:
            page = wikipedia.page(page_title, auto_suggest=False)

            # Get all available content items
            if hasattr(page, "content") and page.content:
                # Create chunks from full content
                chunks = self.chunker.chunk_text(page.content)
            else:
                raise ValueError("Page has no content")

            url = f"https://en.wikipedia.org/wiki/{quote(page.title.replace(' ', '_'))}"

            for idx, chunk in enumerate(chunks):
                description = self.description_generator.create_description(
                    content=chunk,
                    source_type="Wikipedia Simple English",
                    title=page.title,
                    chunk_idx=idx,
                    categories=page.categories if hasattr(page, "categories") else None,
                    include_categories=True,
                    max_preview_length=200,
                )

                research_chunk = ResearchChunk(
                    chunk_id=self.make_chunk_id(page.title, idx),
                    description=description,
                    content=chunk,
                    source="wikipedia_simple_english",
                    url=url,
                    metadata={
                        "article_title": title,
                        "article_id": page.pageid if hasattr(page, "pageid") else None,
                        "chunk_index": idx,
                        "total_chunks": len(chunks),
                        "chunk_size": len(chunk),
                        "overlap": self.chunker.overlap,
                        "embedding_model": self.embedding_model_name,
                        "categories": (
                            page.categories if hasattr(page, "categories") else None
                        ),
                        "language": "simple_english",
                    },
                )
                results.append(research_chunk)

        except Exception as e:
            logger.error(f"Failed to extract content from page '{page_title}': {e}")

        return results

    @staticmethod
    def make_chunk_id(title: str, chunk_idx: int) -> str:
        """Generate consistent chunk_id"""
        raw = f"wikipedia|{title}|{chunk_idx}"
        return hashlib.md5(raw.encode("utf-8")).hexdigest()
