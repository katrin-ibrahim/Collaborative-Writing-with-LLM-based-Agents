from pathlib import Path

import faiss
import json
import logging
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import Dict, List, Optional

from src.config.retrieval_config import DEFAULT_RETRIEVAL_CONFIG, RetrievalConfig
from src.retrieval.rms.base_retriever import BaseRetriever
from src.utils.experiment import find_project_root

logger = logging.getLogger(__name__)


class FaissRM(BaseRetriever):
    """
    FAISS-based retrieval manager using pre-built Supabase Wikipedia embeddings index.
    Focused on search performance with minimal initialization overhead.
    """

    def __init__(
        self,
        format_type: str = "rag",
        cache_dir: str = "data/supabase_cache",
        cache_results: bool = True,
        config: Optional["RetrievalConfig"] = None,
    ):
        # Use passed config or default
        retrieval_config = config or DEFAULT_RETRIEVAL_CONFIG

        # Automatically resolve cache_dir to absolute path from project root
        project_root_path = Path(find_project_root())
        if not Path(cache_dir).is_absolute():
            cache_dir = str(project_root_path / cache_dir)

        logger.debug(f"FaissRM using cache_dir: {cache_dir}")

        # Set embedding model name before loading index (needed by _load_index)
        self.supabase_embedding_model_name = (
            retrieval_config.supabase_embedding_model_name
        )

        # Store cache_dir for _load_index method
        self.cache_dir = cache_dir

        # Load pre-built index (required)
        self._load_index()

        # Initialize parent class with absolute cache path
        super().__init__(
            cache_dir=cache_dir, cache_results=cache_results, config=retrieval_config
        )

        self.retrieval_config = retrieval_config  # Store for use in methods
        self.format_type = format_type
        self.results_per_query = retrieval_config.results_per_query

        self.encoder = SentenceTransformer(
            self.supabase_embedding_model_name, device="cpu"
        )

        # Add semantic filtering setup
        self.semantic_enabled = retrieval_config.semantic_filtering_enabled

    def _retrieve_article(
        self, query: str, topic: str = None, max_results: int = None
    ) -> List[Dict]:
        """Retrieve articles for a single query using FAISS search."""
        # Use config default if max_results not provided
        if max_results is None:
            max_results = self.retrieval_config.results_per_query

        # Check cache first
        cache_key = self._get_cache_key(query)
        cached_results = self._load_from_cache(cache_key)
        if cached_results:
            return cached_results

        if self.faiss_index is None:
            raise RuntimeError("FAISS index not initialized")

        try:
            # Encode query
            query_embedding = self.encoder.encode([query], convert_to_numpy=True)
            query_embedding = query_embedding.astype(np.float32)

            # Normalize for cosine similarity
            faiss.normalize_L2(query_embedding)

            # Search FAISS index
            scores, indices = self.faiss_index.search(query_embedding, max_results)

            # Convert to result dictionaries
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):

                if idx >= len(self.articles):
                    continue

                article = self.articles[idx]

                result = {
                    "title": article.get("title", "Unknown"),
                    "snippets": [article["content"]],
                    "description": article["content"][:100],
                    "url": article["url"],
                    "source": "faiss_wikipedia",
                    "relevance_score": float(score),
                }
                results.append(result)

            # Cache the results
            self._save_to_cache(cache_key, results)

            logger.debug(
                f"Retrieved {len(results)} articles for query: {query[:50]}..."
            )
            return results

        except Exception as e:
            logger.error(f"FAISS search failed for '{query}': {e}")
            return []

    def _load_index(self):
        """Load pre-built FAISS index (index must exist)."""
        # Set cache path based on embedding model
        cache_filename = f"supabase_{self.supabase_embedding_model_name.replace('/', '_').replace('-', '_')}_full.json"
        cache_path = Path(self.cache_dir, cache_filename)

        logger.debug(f"Looking for FAISS index at: {cache_path.absolute()}")
        logger.debug(f"Current working directory: {Path.cwd()}")
        logger.debug(f"Cache directory: {self.cache_dir}")

        if not cache_path.exists():
            logger.error(f"FAISS index file missing: {cache_path.absolute()}")
            logger.error(f"Working directory: {Path.cwd()}")
            logger.error(f"Cache dir setting: {self.cache_dir}")
            raise FileNotFoundError(
                f"Pre-built FAISS index not found at {cache_path.absolute()}. "
                f"Run scripts/build_supabase_faiss_index.py first with --embedding-model {self.supabase_embedding_model_name}"
            )

        try:
            logger.info("Loading cached FAISS index and articles...")

            # Load metadata
            with open(cache_path, "r") as f:
                cache_data = json.load(f)

            self.articles = cache_data["articles"]
            self.embedding_dim = cache_data["embedding_dim"]

            # Load index
            index_file = cache_path.with_suffix(".index")
            if not index_file.exists():
                raise FileNotFoundError(f"FAISS index file not found: {index_file}")

            self.faiss_index = faiss.read_index(str(index_file))

            logger.info(f"Loaded cached index with {len(self.articles)} articles")

        except Exception as e:
            logger.error(f"Error loading index: {e}")
            raise RuntimeError(f"Failed to load pre-built index: {e}")

    # def _get_encoder(self):
    #     """Get or create SentenceTransformer encoder with better device handling."""
    #     if self.encoder is None:
    #         try:
    #             logger.debug(
    #                 f"🔤 Loading SentenceTransformer model: {self.supabase_embedding_model_name}"
    #             )

    #             # Import torch for device management
    #             import torch

    #             # Determine best device
    #             if torch.cuda.is_available():
    #                 device = "cuda"
    #             elif (
    #                 hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    #             ):
    #                 device = "mps"
    #             else:
    #                 device = "cpu"

    #             logger.debug(f"🔤 Using device: {device}")

    #             # Load model with explicit device
    #             self.encoder = SentenceTransformer(
    #                 self.supabase_embedding_model_name, device=device
    #             )

    #             # Test encoding to make sure it works
    #             test_embedding = self.encoder.encode(
    #                 ["test query"], convert_to_numpy=True
    #             )
    #             logger.debug(
    #                 f"🔤 Encoder initialized successfully with output shape: {test_embedding.shape}"
    #             )

    #         except Exception as e:
    #             logger.error(f"Failed to initialize SentenceTransformer encoder: {e}")
    #             # Try CPU fallback
    #             try:
    #                 logger.debug(f"🔤 Trying CPU fallback...")
    #                 self.encoder = SentenceTransformer(
    #                     self.supabase_embedding_model_name, device="cpu"
    #                 )
    #                 test_embedding = self.encoder.encode(
    #                     ["test query"], convert_to_numpy=True
    #                 )
    #                 logger.debug(
    #                     f"🔤 CPU fallback successful with output shape: {test_embedding.shape}"
    #                 )
    #             except Exception as fallback_error:
    #                 logger.error(f"CPU fallback also failed: {fallback_error}")
    #                 raise RuntimeError(f"Could not load embedding model: {e}")

    #     return self.encoder
