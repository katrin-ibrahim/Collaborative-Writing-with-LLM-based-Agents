import threading  # NEW
from pathlib import Path

import faiss
import json
import logging
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Optional

from src.config.config_context import ConfigContext
from src.retrieval.rms.base_retriever import BaseRetriever
from src.utils.data.models import ResearchChunk
from src.utils.experiment import find_project_root

_FAISS_GLOBAL_LOCK = threading.RLock()  # NEW: serialize all faiss.search calls
faiss.omp_set_num_threads(1)  # To avoid segfaults on some systems
logger = logging.getLogger(__name__)


class FaissRM(BaseRetriever):
    """
    FAISS-based retrieval manager using a pre-built, chunked Wikipedia embeddings index.
    The index uses Scalar Quantization for memory efficiency and returns ResearchChunk objects.
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

        # Resolve cache_dir to absolute path from project root
        project_root_path = Path(find_project_root())
        if not Path(cache_dir).is_absolute():
            cache_dir = str(project_root_path / cache_dir)
        logger.debug(f"FaissRM using cache_dir for results: {cache_dir}")

        # Set embedding model name before loading index (needed by _load_index)
        self.embedding_model_name = self.retrieval_config.embedding_model

        # Store cache_dir for _load_index method
        self.cache_dir = cache_dir

        # Load pre-built index (required)
        self._load_index()  # sets: self.faiss_index, self.embedding_dim, self.embedding_model_in_index, self.metric_type

        # Initialize parent class
        super().__init__(
            cache_dir=cache_dir,
            cache_results=cache_results,
            config=self.retrieval_config,
        )

        self.format_type = format_type
        self.results_per_query = self.retrieval_config.results_per_query

        # Build encoder
        self.encoder = SentenceTransformer(self.embedding_model_name, device="cpu")

        # Dimension guard (prevents segfaults)
        try:
            enc_dim = self.encoder.get_sentence_embedding_dimension()
        except Exception:
            # Fallback for older sentence-transformers
            _probe = self.encoder.encode(["_probe_"], convert_to_numpy=True)
            enc_dim = int(_probe.shape[-1])

        if enc_dim != self.embedding_dim:
            raise RuntimeError(
                f"Embedding dimension mismatch: encoder={enc_dim}, index={self.embedding_dim}. "
                f"Index was built with {self.embedding_model_in_index}"
            )

    def _retrieve_article(
        self, query: str, max_results: Optional[int] = None
    ) -> List[ResearchChunk]:
        """Retrieve chunks for a single query using FAISS search."""
        if not query or not query.strip():
            logger.warning("Empty query; returning no results.")
            return []

        if self.faiss_index is None:
            raise RuntimeError("FAISS index not initialized")

        if self.faiss_index.ntotal <= 0:
            logger.error("FAISS index is empty (ntotal=0); returning no results.")
            return []

        # Use config default if max_results not provided
        if max_results is None:
            if self.retrieval_config is None:
                raise RuntimeError(
                    "retrieval_config is None. Cannot determine results_per_query."
                )
            max_results = int(self.retrieval_config.results_per_query)

        try:
            # Encode -> 2D, C-contiguous float32
            q = self.encoder.encode([query], convert_to_numpy=True)
            if q.ndim == 1:
                q = q.reshape(1, -1)
            q = np.asarray(q, dtype=np.float32, order="C")

            # Guard: finite BEFORE any normalization
            if not np.isfinite(q).all():
                logger.error(
                    "Query embedding contains NaN/Inf pre-normalization; skipping search."
                )
                return []

            # If IP (cosine), normalize but only if norm > 0
            if str(self.metric_type).lower() == "ip":
                # compute row-wise L2 norm
                norms = np.linalg.norm(q, axis=1)
                if not np.all(norms > 0):
                    logger.warning(
                        "Zero-norm embedding; cannot cosine-normalize. Skipping search."
                    )
                    return []
                faiss.normalize_L2(q)

                # Guard: finite AFTER normalization (just in case)
                if not np.isfinite(q).all():
                    logger.error(
                        "Query embedding became NaN/Inf after normalization; skipping search."
                    )
                    return []

            # Cap k to a valid range
            k = max(1, int(max_results))
            k = min(k, int(self.faiss_index.ntotal))

            # Core FAISS call
            with _FAISS_GLOBAL_LOCK:
                scores, indices = self.faiss_index.search(q, k)

            # Convert to ResearchChunk objects
            results: List[ResearchChunk] = []
            row_scores = scores[0] if scores.ndim == 2 else scores
            row_indices = indices[0] if indices.ndim == 2 else indices

            for rank, (score, idx) in enumerate(zip(row_scores, row_indices), start=1):
                if idx < 0 or idx >= len(self.chunks_metadata):
                    continue

                chunk = self.chunks_metadata[idx].copy()
                chunk["relevance_score_normalized"] = float(
                    score
                )  # cosine if IP+normalized
                chunk["rank"] = rank

                meta = chunk.get("metadata") or {}
                meta["faiss_index"] = int(idx)
                chunk["metadata"] = meta

                try:
                    results.append(ResearchChunk.model_validate(chunk))
                except Exception as ve:
                    logger.warning(f"Failed to validate chunk at index {idx}: {ve}")
                    continue

            logger.debug(f"Retrieved {len(results)} chunks for query: {query[:50]}...")
            return results

        except AssertionError as e:
            logger.error(f"FAISS assertion error during search: {e}")
            return []
        except Exception as e:
            logger.error(f"FAISS search failed for '{query}': {e}")
            return []

    def _load_index(self):
        """Load the three files required for the chunked FAISS index."""
        project_root = Path(find_project_root())

        # Base name for index and metadata
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

            self.embedding_dim = int(
                index_metadata.get("embedding_dim", self.faiss_index.d)
            )
            self.embedding_model_in_index = index_metadata.get(
                "embedding_model", "unknown"
            )
            self.metric_type = index_metadata.get("metric", "ip")

            # 3. Load Chunk Text Metadata (The large JSONL file)
            logger.info(f"Loading chunk text data from: {chunk_data_path_file}")
            self.chunks_metadata = []
            with open(chunk_data_path_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        self.chunks_metadata.append(json.loads(line))

            logger.info(
                f"Loaded index with {self.faiss_index.ntotal} vectors and {len(self.chunks_metadata)} chunks"
            )

        except Exception as e:
            logger.error(f"Error loading chunked index components: {e}")
            raise RuntimeError(f"Failed to load pre-built chunked index: {e}")
