import hashlib
from pathlib import Path
from urllib.parse import quote

import logging
import wikipedia
from typing import List, Optional

from src.config.config_context import ConfigContext
from src.retrieval.rms.base_retriever import BaseRetriever
from src.retrieval.utils.chunker import ContentChunker
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
        cache_results: bool = False,
    ):
        # Use passed config or default
        self.retrieval_config = ConfigContext.get_retrieval_config()
        if self.retrieval_config is None:
            raise RuntimeError(
                "Retrieval config is None. Ensure ConfigContext is properly initialized before using WikiRM."
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

        self.chunker = ContentChunker()

        # Setup Wikipedia client
        wikipedia.set_rate_limiting(True)
        wikipedia.set_lang("en")

    def _retrieve_article(
        self,
        query: str,
        max_results: Optional[int] = None,
    ) -> List[ResearchChunk]:
        """
        Retrieve article by direct page title access.
        Query should be an exact Wikipedia page title, not a search term.
        """
        if max_results is None:
            if self.retrieval_config is None:
                raise RuntimeError(
                    "retrieval_config is None. Cannot determine results_per_query."
                )
            max_results = self.retrieval_config.results_per_query
        if max_results is None:
            max_results = 5

        # Extract chunks from the resolved page
        chunks = self._extract_page_content(query)

        return chunks[:max_results] if chunks else []

    # --- make _extract_page_content robust if a disambiguation slips through ---

    def _extract_page_content(self, page_title: str) -> List[ResearchChunk]:
        """Extract content from a single Wikipedia page. NO disambiguation handling."""
        from wikipedia import DisambiguationError, PageError
        from wikipedia import page as wpage

        results: List[ResearchChunk] = []

        try:
            p = wpage(page_title, auto_suggest=False, redirect=True)
        except DisambiguationError:
            logger.debug(f"[WikiRM] Skipping disambiguation page: '{page_title}'")
            return []
        except PageError:
            logger.debug(f"[WikiRM] Page not found: '{page_title}'")
            return []
        except Exception as e:
            logger.error(f"[WikiRM] Error loading page '{page_title}': {e}")
            return []

        # content
        if not getattr(p, "content", None):
            logger.error(f"[WikiRM] Page has no content: '{p.title}'")
            return []

        logger.debug(
            f"[WikiRM] Extracting chunks from page: '{p.title}' (content length: {len(p.content)})"
        )

        chunks = self.chunker.chunk_text(p.content)
        url = f"https://en.wikipedia.org/wiki/{quote(p.title.replace(' ', '_'))}"

        logger.debug(f"[WikiRM] Created {len(chunks)} chunks from '{p.title}'")

        for idx, chunk in enumerate(chunks):

            description = f"Article: '{p.title}', Chunk excerpt: {chunk[:512]}..."
            research_chunk = ResearchChunk(
                chunk_id=self.make_chunk_id(p.title, idx),
                description=description,  # Use the content-based description, not just title
                content=chunk,
                source="wikipedia_en",
                url=url,
                metadata={
                    "article_title": p.title,  # Store title in metadata instead
                    "article_id": getattr(p, "pageid", None),
                    "chunk_index": idx,
                    "total_chunks": len(chunks),
                    "chunk_size": len(chunk),
                    "overlap": self.chunker.overlap,
                    "embedding_model": self.embedding_model_name,
                    "categories": getattr(p, "categories", None),
                    "language": "en",
                },
            )
            results.append(research_chunk)

        return results

    @staticmethod
    def make_chunk_id(title: str, chunk_idx: int) -> str:
        """Generate consistent chunk_id"""
        raw = f"{title}|{chunk_idx}"
        return hashlib.md5(raw.encode("utf-8")).hexdigest()
