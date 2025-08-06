"""
Centralized configuration for all retrieval and context generation parameters.
Replaces scattered hardcoded k values throughout the codebase.
"""

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class RetrievalConfig:
    """Centralized retrieval parameters for consistent evaluation."""

    num_queries: int = 1
    max_results: int = 1
    max_passages: int = 1
    passage_max_length: int = 600

    # Wikipedia Retrieval Configuration
    max_articles: int = 1
    wiki_parallel_workers: int = 1

    # Context Generation Configuration
    context_max_passages: int = 15
    context_passage_max_length: int = 1200
    context_min_passage_length: int = 100

    # Batch Processing Configuration
    batch_parallel_threshold: int = 3
    batch_max_workers_direct: int = 3
    batch_max_workers_rag: int = 2
    batch_max_workers_storm: int = 1

    # Semantic Filtering Configuration
    semantic_filtering_enabled: bool = True
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    similarity_threshold: float = 0.4
    semantic_cache_size: int = 1000

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging and serialization."""
        return {
            "rag": {
                "num_queries": self.rag_num_queries,
                "max_results": self.rag_max_results,
                "max_passages": self.rag_max_passages,
                "passage_max_length": self.rag_passage_max_length,
            },
            "wikipedia": {
                "max_articles": self.wiki_max_articles,
                "max_sections": self.wiki_max_sections,
                "parallel_workers": self.wiki_parallel_workers,
            },
            "semantic": {
                "filtering_enabled": self.semantic_filtering_enabled,
                "embedding_model": self.embedding_model,
                "similarity_threshold": self.similarity_threshold,
                "cache_size": self.semantic_cache_size,
            },
            "context": {
                "max_passages": self.context_max_passages,
                "passage_max_length": self.context_passage_max_length,
                "min_passage_length": self.context_min_passage_length,
            },
            "batch": {
                "parallel_threshold": self.batch_parallel_threshold,
                "max_workers": {
                    "direct": self.batch_max_workers_direct,
                    "rag": self.batch_max_workers_rag,
                    "storm": self.batch_max_workers_storm,
                },
            },
        }

    @classmethod
    def get_default(cls) -> "RetrievalConfig":
        """Get default configuration instance."""
        return cls()

    def validate(self) -> bool:
        """Validate configuration parameters."""
        if self.rag_num_queries <= 0 or self.rag_max_results <= 0:
            return False
        if self.wiki_max_articles <= 0 or self.wiki_max_sections <= 0:
            return False
        if self.context_max_passages <= 0 or self.context_passage_max_length <= 0:
            return False
        if self.batch_parallel_threshold <= 0:
            return False
        return True


# Global default instance
DEFAULT_RETRIEVAL_CONFIG = RetrievalConfig.get_default()
