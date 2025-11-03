import hashlib
from pathlib import Path
from urllib.parse import quote

import logging
import wikipedia
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

    def _resolve_title(self, title: str, query: Optional[str] = None) -> Optional[str]:
        """Return a concrete Wikipedia page title or None if not resolvable."""
        from wikipedia import DisambiguationError, PageError, page, suggest

        def _score(opt: str) -> int:
            s = 0
            # prefer options that match query tokens
            if query:
                for tok in query.lower().split():
                    if tok and tok in opt.lower():
                        s += 2
            # small bonus for having a year/parenthetical which often disambiguates
            if any(ch.isdigit() for ch in opt):
                s += 1
            if "(" in opt:
                s += 1
            # penalize generic/junk
            if any(
                x in opt.lower()
                for x in ["list of", "category:", "portal:", "index of"]
            ):
                s -= 5
            return s

        try:
            p = page(title, auto_suggest=False, redirect=True, preload=False)
            return p.title  # concrete page resolved by Wikipedia
        except DisambiguationError as e:
            # pick the best option, then recurse once
            candidates = [
                opt
                for opt in e.options
                if not any(
                    x in opt.lower()
                    for x in ["list of", "category:", "portal:", "index of"]
                )
            ]
            if not candidates:
                return None
            best = sorted(candidates, key=lambda o: (-_score(o), len(o)))[0]
            try:
                p = page(best, auto_suggest=False, redirect=True, preload=False)
                return p.title
            except (DisambiguationError, PageError):
                return None
        except PageError:
            # try a library suggestion
            sug = suggest(title)
            if sug:
                try:
                    p = page(sug, auto_suggest=False, redirect=True, preload=False)
                    return p.title
                except Exception:
                    return None
            return None
        except Exception:
            return None

    def _retrieve_article(
        self,
        query: str,
        max_results: Optional[int] = None,
    ) -> List[ResearchChunk]:
        """Wikipedia retriever with disambiguation resolution and noise filtering."""
        import concurrent.futures

        from wikipedia import search

        if max_results is None:
            if self.retrieval_config is None:
                raise RuntimeError(
                    "retrieval_config is None. Cannot determine results_per_query."
                )
            max_results = self.retrieval_config.results_per_query
        if max_results is None:
            max_results = 5  # fallback default to ensure int type

        try:
            titles = search(query, results=max_results)
        except Exception as e:
            logger.error(f"[WikiRM] Wikipedia search failed for '{query}': {e}")
            return []

        if not titles:
            logger.warning(f"[WikiRM] No results for query: '{query}'")
            return []

        clean = [
            t
            for t in titles
            if not any(
                x in t.lower() for x in ["list of", "category:", "portal:", "index of"]
            )
        ]
        if not clean:
            return []

        results: List[ResearchChunk] = []
        seen_titles = set()

        def process_title(raw_title: str) -> List[ResearchChunk]:
            # Resolve *before* extraction to avoid DisambiguationError later
            resolved = self._resolve_title(raw_title, query=query)
            if not resolved:
                logger.debug(f"[WikiRM] Could not resolve title: {raw_title!r}")
                return []
            if resolved in seen_titles:
                return []
            seen_titles.add(resolved)
            return self._extract_page_content(resolved)

        # keep threads modest; wikipedia lib has rate limiting on
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(process_title, t) for t in clean[: max_results * 2]
            ]
            for fut in concurrent.futures.as_completed(futures):
                try:
                    chunk_list = fut.result()
                    for c in chunk_list:
                        results.append(c)
                        if len(results) >= max_results:
                            break
                except Exception as e:
                    logger.debug(f"[WikiRM] worker error: {e}")
                if len(results) >= max_results:
                    break

        return results[:max_results]

    # --- make _extract_page_content robust if a disambiguation slips through ---

    def _extract_page_content(self, page_title: str) -> List[ResearchChunk]:
        """Extract content from a single (resolved) Wikipedia page."""
        from wikipedia import DisambiguationError, PageError
        from wikipedia import page as wpage

        results: List[ResearchChunk] = []

        try:
            p = wpage(page_title, auto_suggest=False, redirect=True)
        except DisambiguationError:
            # try one-shot resolve and retry
            resolved = self._resolve_title(page_title)
            if not resolved:
                logger.error(
                    f"[WikiRM] Disambiguation could not be resolved for '{page_title}'"
                )
                return []
            try:
                p = wpage(resolved, auto_suggest=False, redirect=True)
            except Exception as e:
                logger.error(f"[WikiRM] Failed to load resolved page '{resolved}': {e}")
                return []
        except PageError as e:
            logger.error(f"[WikiRM] PageError for '{page_title}': {e}")
            return []
        except Exception as e:
            logger.error(f"[WikiRM] Unexpected error for '{page_title}': {e}")
            return []

        # content
        if not getattr(p, "content", None):
            logger.error(f"[WikiRM] Page has no content: '{p.title}'")
            return []

        chunks = self.chunker.chunk_text(p.content)
        url = f"https://en.wikipedia.org/wiki/{quote(p.title.replace(' ', '_'))}"

        for idx, chunk in enumerate(chunks):
            description = self.description_generator.create_description(
                content=chunk,
                title=p.title,
                chunk_idx=idx,
            )

            research_chunk = ResearchChunk(
                chunk_id=self.make_chunk_id(p.title, idx),
                description=description,
                content=chunk,
                source="wikipedia_en",
                url=url,
                metadata={
                    "article_title": p.title,
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
        raw = f"wikipedia|{title}|{chunk_idx}"
        return hashlib.md5(raw.encode("utf-8")).hexdigest()
