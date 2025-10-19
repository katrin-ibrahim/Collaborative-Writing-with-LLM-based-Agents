from pathlib import Path

import faiss
import json
import logging
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import Any, Dict, List, Optional  # Added Any for metadata

from src.config.config_context import ConfigContext
from src.retrieval.rms.base_retriever import BaseRetriever
from src.utils.data.models import ResearchChunk
from src.utils.experiment import find_project_root

logger = logging.getLogger(__name__)


class FaissRM(BaseRetriever):
    """
    FAISS-based retrieval manager using a pre-built, chunked Wikipedia embeddings index.
    The index uses Scalar Quantization for memory efficiency and returns structured
    ResearchChunk objects (Pydantic model).
    """

    def __init__(
        self,
        format_type: str = "rag",
        cache_dir: str = "data/faiss_cache",
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

        logger.debug(f"FaissRM using cache_dir for results: {cache_dir}")

        # Set embedding model name before loading index (needed by _load_index)
        self.embedding_model_name = self.retrieval_config.embedding_model

        # Store cache_dir for _load_index method
        self.cache_dir = cache_dir

        # Load pre-built index (required)
        self._load_index()

        # Initialize parent class with absolute cache path
        super().__init__(
            cache_dir=cache_dir,
            cache_results=cache_results,
            config=self.retrieval_config,
        )

        self.format_type = format_type
        self.results_per_query = self.retrieval_config.results_per_query

        self.encoder = SentenceTransformer(self.embedding_model_name, device="cpu")

        # Add semantic filtering setup
        self.semantic_enabled = self.retrieval_config.semantic_filtering_enabled

    def _retrieve_article(
        self,
        query: str,
        topic: Optional[str] = None,
        max_results: Optional[int] = None,
        **kwargs,
    ) -> List[ResearchChunk]:
        """Retrieve chunks for a single query using FAISS search."""
        # Use config default if max_results not provided
        if max_results is None:
            if self.retrieval_config is None:
                raise RuntimeError(
                    "retrieval_config is None. Cannot determine results_per_query."
                )
            max_results = self.retrieval_config.results_per_query

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

            # Convert to ResearchChunk objects
            results: List[ResearchChunk] = []

            for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
                # Skip invalid indices
                if idx < 0 or idx >= len(self.chunks_metadata):
                    continue

                # Retrieve the chunk metadata and add retrieval metadata
                chunk = self.chunks_metadata[idx].copy()
                chunk["relevance_score_normalized"] = float(score)
                chunk["rank"] = rank

                # Add faiss_index to metadata if it exists
                if "metadata" not in chunk:
                    chunk["metadata"] = {}
                chunk["metadata"]["faiss_index"] = int(idx)

                # Use Pydantic's model_validate directly
                try:
                    result = ResearchChunk.model_validate(chunk)
                    results.append(result)
                except Exception as validation_error:
                    logger.warning(
                        f"Failed to validate chunk at index {idx}: {validation_error}"
                    )
                    continue

            logger.debug(f"Retrieved {len(results)} chunks for query: {query[:50]}...")
            return results

        except Exception as e:
            logger.error(f"FAISS search failed for '{query}': {e}")
            return []

    def _load_index(self):
        """Load the three files required for the chunked FAISS index."""
        project_root = Path(find_project_root())

        # --- NEW PATH LOGIC: Use paths from the successful build log ---
        # Base name for index and metadata (derived from your previous output)
        index_base_name = "wikipedia_thenlper_gte_small_cs512_ov50"
        index_dir = project_root / "data" / "faiss_indexes"

        # File paths
        index_file = index_dir / f"{index_base_name}.index"
        metadata_file = index_dir / f"{index_base_name}.json"
        chunk_data_path_file = index_dir / "chunks.jsonl"

        logger.debug(f"Looking for FAISS index at: {index_file.absolute()}")

        if (
            not index_file.exists()
            or not metadata_file.exists()
            or not chunk_data_path_file.exists()
        ):
            raise FileNotFoundError(
                f"Chunked FAISS index files not found in {index_dir.absolute()}. "
                f"Missing one or more of: {index_file.name}, {metadata_file.name}, {chunk_data_path_file.name}. "
                f"Ensure the build script completed successfully."
            )

        try:
            logger.info("Loading chunked FAISS index and chunk metadata...")

            # 1. Load FAISS index
            self.faiss_index = faiss.read_index(str(index_file))

            # 2. Load Index Metadata (embedding dim, etc.)
            with open(metadata_file, "r") as f:
                index_metadata = json.load(f)
            self.embedding_dim = index_metadata["embedding_dim"]

            # 3. Load Chunk Text Metadata (The large JSONL file)
            logger.info(f"Loading chunk text data from: {chunk_data_path_file}")

            # Using 'chunks_metadata' instead of 'articles'
            self.chunks_metadata = []
            with open(chunk_data_path_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        # Each line is a JSON object (a chunk)
                        self.chunks_metadata.append(json.loads(line))

            logger.info(
                f"Loaded index with {self.faiss_index.ntotal} vectors and {len(self.chunks_metadata)} chunks"
            )

        except Exception as e:
            logger.error(f"Error loading chunked index components: {e}")
            raise RuntimeError(f"Failed to load pre-built chunked index: {e}")
