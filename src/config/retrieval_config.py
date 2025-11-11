"""
Centralized configuration for all retrieval and context generation parameters.
Eliminates redundancy and provides consistent naming.
"""

import logging
from dataclasses import dataclass

from src.config.base_config import BaseConfig

logger = logging.getLogger(__name__)


@dataclass
class RetrievalConfig(BaseConfig):

    # Retrieval Manager Configuration
    retrieval_manager: str = "wiki"  # Options: "wiki", "faiss"

    # Single Source of Truth for Retrieval Flow
    num_queries: int = 5  # How many search queries to generate
    results_per_query: int = 8
    max_content_pieces: int = (
        3  # How many sections/chunks per result (replaces wiki_max_sections)
    )
    final_passages: int = (
        8  # Final context size (replaces max_final_passages, retrieve_top_k)
    )

    # Content Processing
    embedding_model: str = "thenlper/gte-small"
    chunk_size: int = 512
    chunk_overlap: int = 50

    # Semantic Filtering Configuration
    semantic_filtering_enabled: bool = True
    # gte-small for better embeddings, gte-base for better quality (slower)
    similarity_threshold: float = 0.4
    semantic_cache_size: int = 1000

    # Storm
    queries_per_turn: int = 3  # How many queries per conversation turn

    def get_file_pattern(self) -> str:
        # Define how experiment outputs for retrieval should be named
        return f"configs/retrieval_{self.retrieval_manager}.json"

    @classmethod
    def get_default(cls) -> "RetrievalConfig":
        """Get default configuration."""
        return cls()


# Global default instance
DEFAULT_RETRIEVAL_CONFIG = RetrievalConfig.get_default()
