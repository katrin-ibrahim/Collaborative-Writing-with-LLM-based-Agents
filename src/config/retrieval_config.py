"""
Centralized configuration for all retrieval and context generation parameters.
Eliminates redundancy and provides consistent naming.
"""

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class RetrievalConfig:
    """Truly unified retrieval parameters - eliminates all redundancy."""

    # Single Source of Truth for Retrieval Flow
    num_queries: int = 3  # How many search queries to generate
    results_per_query: int = (
        2  # How many results per query (replaces max_results_per_query, search_top_k, wiki_max_articles)
    )
    max_content_pieces: int = (
        10  # How many sections/chunks per result (replaces wiki_max_sections)
    )
    final_passages: int = (
        5  # Final context size (replaces max_final_passages, retrieve_top_k)
    )
    queries_per_turn: int = 2

    # Content Processing
    passage_max_length: int = 600
    passage_min_length: int = 100

    # Batch Processing Configuration
    parallel_threshold: int = 3  # Threshold for parallel processing
    max_workers_direct: int = 3
    max_workers_rag: int = 2
    max_workers_storm: int = 1
    max_workers_wiki: int = 1

    # Semantic Filtering Configuration
    semantic_filtering_enabled: bool = True
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    similarity_threshold: float = 0.4
    semantic_cache_size: int = 1000

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging and serialization."""
        return {
            "retrieval_flow": {
                "num_queries": self.num_queries,
                "results_per_query": self.results_per_query,
                "max_content_pieces": self.max_content_pieces,
                "final_passages": self.final_passages,
                "queries_per_turn": self.queries_per_turn,
            },
            "content": {
                "max_length": self.passage_max_length,
                "min_length": self.passage_min_length,
            },
            "wiki": {
                "parallel_workers": self.max_workers_wiki,
            },
            "semantic": {
                "filtering_enabled": self.semantic_filtering_enabled,
                "embedding_model": self.embedding_model,
                "similarity_threshold": self.similarity_threshold,
                "cache_size": self.semantic_cache_size,
            },
            "batch": {
                "parallel_threshold": self.parallel_threshold,
                "max_workers": {
                    "direct": self.max_workers_direct,
                    "rag": self.max_workers_rag,
                    "storm": self.max_workers_storm,
                },
            },
        }

    @classmethod
    def get_default(cls) -> "RetrievalConfig":
        """Get default configuration instance."""
        return cls()

    def validate(self) -> bool:
        """Validate configuration parameters."""
        if self.num_queries <= 0 or self.results_per_query <= 0:
            return False
        if self.max_content_pieces <= 0 or self.final_passages <= 0:
            return False
        if self.passage_max_length <= 0 or self.batch_parallel_threshold <= 0:
            return False
        return True

    def get_storm_config(self) -> Dict[str, Any]:
        """Get STORM-compatible configuration mapping."""
        return {
            "max_search_queries_per_turn": self.num_queries,
            "search_top_k": self.results_per_query,
            "retrieve_top_k": self.final_passages,
        }

    def get_wiki_config(self) -> Dict[str, Any]:
        """Get WikiRM-compatible configuration mapping."""
        return {
            "max_articles": self.results_per_query,
            "max_sections": self.max_content_pieces,
        }


# Global default instance
DEFAULT_RETRIEVAL_CONFIG = RetrievalConfig.get_default()
