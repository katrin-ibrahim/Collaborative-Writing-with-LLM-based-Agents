"""
Enhanced FAISS Wikipedia Retrieval Manager
Uses efficient disk-based FAISS indices to handle full Wikipedia dumps.
Supports lazy loading and streaming for memory efficiency.
"""

from pathlib import Path

import faiss
import logging
import pickle
from sentence_transformers import SentenceTransformer
from typing import Dict, List, Optional

from .data_loader import WikipediaDataLoader

logger = logging.getLogger(__name__)


class EnhancedFAISSWikiRM:
    """
    Enhanced FAISS semantic search that can handle full Wikipedia dumps efficiently.
    Uses disk-based indices and lazy loading to minimize memory usage.
    """

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        index_prefix: str = "faiss_full",
        fallback_articles: int = 10000,
        project_root: Optional[str] = None,
    ):
        """
        Initialize enhanced FAISS retrieval manager.

        Args:
            embedding_model: SentenceTransformer model name
            index_prefix: Prefix for index files (e.g., 'faiss_full', 'faiss_1000')
            fallback_articles: Number of articles to use if full index not available
            project_root: Project root directory (auto-detected if None)
        """
        logger.info("Initializing Enhanced FAISSWikiRM...")

        self.embedding_model_name = embedding_model
        self.index_prefix = index_prefix

        # Determine project root
        if project_root is None:
            self.project_root = self._find_project_root()
        else:
            self.project_root = project_root

        logger.info(f"Using project root: {self.project_root}")

        # Load embedding model
        logger.info(f"Loading embedding model: {embedding_model}")
        self.encoder = SentenceTransformer(embedding_model)

        # Find and load the best available index
        self.index, self.metadata, self.num_articles = self._load_best_index(
            fallback_articles
        )

        logger.info(f"Enhanced FAISSWikiRM ready with {self.num_articles} articles!")

    def _find_project_root(self) -> str:
        """Find the project root directory."""
        current = Path(__file__).parent.parent.parent.absolute()

        # Look for indicators of project root
        indicators = ["requirements.txt", "README.md", ".git", "src"]

        while current != current.parent:
            if any((current / indicator).exists() for indicator in indicators):
                return str(current)
            current = current.parent

        # Fallback to current directory
        return str(Path.cwd())

    def _find_index_files(self) -> List[tuple]:
        """
        Find all available FAISS index files.
        Returns list of (num_articles, index_file, metadata_file) tuples.
        """
        index_files = []

        # Look for index files in project root
        for file_path in Path(self.project_root).glob(f"{self.index_prefix}_*.index"):
            # Extract number of articles from filename
            try:
                parts = file_path.stem.split("_")
                num_articles = int(parts[-1])

                # Check for corresponding metadata file
                metadata_file = file_path.with_suffix(".pkl").with_name(
                    file_path.stem.replace("index", "metadata") + ".pkl"
                )

                if metadata_file.exists():
                    index_files.append(
                        (num_articles, str(file_path), str(metadata_file))
                    )
                else:
                    logger.warning(f"No metadata file found for {file_path}")

            except (ValueError, IndexError):
                logger.warning(f"Could not parse article count from {file_path}")
                continue

        # Sort by number of articles (largest first)
        index_files.sort(key=lambda x: x[0], reverse=True)
        return index_files

    def _load_best_index(self, fallback_articles: int):
        """
        Load the best available FAISS index.
        Falls back to building a smaller index if no full index is available.
        """
        # Try to find existing full indices
        index_files = self._find_index_files()

        if index_files:
            # Use the largest available index
            num_articles, index_file, metadata_file = index_files[0]
            logger.info(f"Loading existing FAISS index with {num_articles} articles...")

            try:
                # Load index
                index = faiss.read_index(index_file)

                # Load metadata
                with open(metadata_file, "rb") as f:
                    metadata = pickle.load(f)

                logger.info(f"Successfully loaded index with {num_articles} articles")
                return index, metadata, num_articles

            except Exception as e:
                logger.error(f"Failed to load index {index_file}: {e}")
                logger.info("Falling back to building new index...")

        # Fallback: build a new index with limited articles
        logger.info(
            f"Building fallback FAISS index with {fallback_articles} articles..."
        )
        return self._build_fallback_index(fallback_articles)

    def _build_fallback_index(self, num_articles: int):
        """Build a fallback index with limited articles."""
        # Load articles
        articles = WikipediaDataLoader.load_articles(num_articles)

        # Create embeddings
        texts = [article["text"][:512] for article in articles]
        embeddings = self.encoder.encode(texts, show_progress_bar=True)

        # Build FAISS index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)

        # Normalize and add embeddings
        faiss.normalize_L2(embeddings)
        index.add(embeddings.astype("float32"))

        # Create metadata
        metadata = [
            {
                "title": article["title"],
                "url": article["url"],
                "text_preview": article["text"][:200],
                "full_text": article["text"],  # Keep full text for fallback
            }
            for article in articles
        ]

        logger.info(f"Built fallback index with {len(articles)} articles")
        return index, metadata, len(articles)

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
        STORM-compatible search method using FAISS vector similarity.
        """
        # Handle STORM calling conventions (same as WikiRM)
        if query is not None:
            query = query
        elif args:
            query = args[0] if len(args) > 0 else None
            if max_results is None and len(args) > 1:
                max_results = args[1]
        elif "query_or_queries" in kwargs and kwargs["query_or_queries"] is not None:
            query = kwargs.pop("query_or_queries", None)
        elif len(args) == 0 and not kwargs:
            return []

        if query is None:
            raise ValueError("No valid query provided")

        max_results = max_results if max_results is not None else 10
        format_type = format_type if format_type else "rag"

        # Normalize input
        if isinstance(query, str):
            query_list = [query]
        else:
            query_list = list(query)

        all_results = []

        for q in query_list:
            query_results = self._search_single_query(q, max_results)
            all_results.extend(query_results)

            if len(all_results) >= max_results:
                break

        # Format results based on type
        if format_type == "rag":
            return [result["content"] for result in all_results[:max_results]]
        elif format_type == "storm":
            storm_results = []
            for result in all_results[:max_results]:
                storm_result = {
                    "snippets": result["content"],
                    "title": result["title"],
                    "url": result["url"],
                    "source": "enhanced_faiss_wiki",
                }
                storm_results.append(storm_result)
            return storm_results
        else:
            return all_results[:max_results]

    def _search_single_query(self, query: str, max_results: int) -> List[Dict]:
        """Search using FAISS semantic similarity."""
        try:
            # Encode query
            query_embedding = self.encoder.encode([query])
            faiss.normalize_L2(query_embedding)

            # Search FAISS index
            scores, indices = self.index.search(
                query_embedding.astype("float32"), max_results
            )

            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx != -1 and idx < len(self.metadata):  # Valid result
                    metadata = self.metadata[idx]

                    results.append(
                        {
                            "content": metadata.get(
                                "full_text", metadata.get("text_preview", "")
                            ),
                            "title": metadata["title"],
                            "url": metadata.get(
                                "url", f"wikipedia:///{metadata['title']}"
                            ),
                            "source": "enhanced_faiss_wiki",
                            "score": float(score),
                        }
                    )

            return results
        except Exception as e:
            logger.error(f"FAISS search failed: {e}")
            return []

    def __call__(self, *args, **kwargs):
        """Support callable interface for STORM compatibility."""
        return self.search(*args, **kwargs)

    def get_stats(self) -> Dict:
        """Get statistics about the loaded index."""
        return {
            "num_articles": self.num_articles,
            "embedding_model": self.embedding_model_name,
            "index_type": type(self.index).__name__,
            "dimension": self.index.d if hasattr(self.index, "d") else "unknown",
        }
