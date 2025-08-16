"""
Centralized configuration for all retrieval and context generation parameters.
Eliminates redundancy and provides consistent naming.
"""

import logging
import os
import yaml
from dataclasses import dataclass
from typing import Any, Dict

logger = logging.getLogger(__name__)


@dataclass
class RetrievalConfig:
    """Truly unified retrieval parameters - eliminates all redundancy."""

    # Retrieval Manager Configuration
    retrieval_manager_type: str = "wiki"  # Options: "wiki", "bm25_wiki", "faiss_wiki"

    # Single Source of Truth for Retrieval Flow
    num_queries: int = 3  # How many search queries to generate
    results_per_query: int = (
        3  # How many results per query (replaces max_results_per_query, search_top_k, wiki_max_articles)
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

    # Enhancement Configuration
    use_wikidata_enhancement: bool = False
    wikidata_cache_file: str = "wikidata_cache.json"

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

    def validate(self) -> bool:
        """Validate configuration parameters."""
        if self.num_queries <= 0 or self.results_per_query <= 0:
            return False
        if self.max_content_pieces <= 0 or self.final_passages <= 0:
            return False
        if self.passage_max_length <= 0 or self.parallel_threshold <= 0:
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

    @classmethod
    def from_yaml(cls, config_name: str) -> "RetrievalConfig":
        """
        Load configuration from YAML file.

        Args:
            config_name: Name of config (e.g., 'wiki', 'txtai', 'bm25_wikidump', etc.)
                        or full path to YAML file

        Returns:
            RetrievalConfig instance loaded from YAML
        """
        # Determine config path
        if config_name.endswith(".yaml") or config_name.endswith(".yml"):
            config_path = config_name
        else:
            # Standard config names
            config_dir = os.path.dirname(__file__)
            config_path = os.path.join(config_dir, f"retrieval_{config_name}.yaml")

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Retrieval config not found: {config_path}")

        with open(config_path, "r") as f:
            config_data = yaml.safe_load(f)

        # Create instance from YAML data
        return cls(**config_data)

    @classmethod
    def from_base_config_with_overrides(
        cls, rm_type: str, **overrides
    ) -> "RetrievalConfig":
        """Create RetrievalConfig from base RM config + CLI overrides."""
        # Try to load base config for the RM type
        config_dir = os.path.dirname(__file__)
        base_config_path = os.path.join(config_dir, f"rm_{rm_type}.yaml")

        base_data = {}
        if os.path.exists(base_config_path):
            # Load base config if it exists
            with open(base_config_path, "r") as f:
                base_data = yaml.safe_load(f)
        else:
            # If no specific config file exists, use defaults for this RM type
            logger.info(f"No specific config found for {rm_type}, using defaults")

        # Set default experimental parameters
        config_data = {
            # Set the retrieval manager type
            "retrieval_manager_type": rm_type,
            # Default experimental parameters
            "num_queries": 3,
            "results_per_query": 3,
            "max_content_pieces": 10,
            "final_passages": 5,
            "queries_per_turn": 2,
            "passage_max_length": 800,
            "passage_min_length": 100,
            "semantic_filtering_enabled": False,
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "similarity_threshold": 0.4,
            "semantic_cache_size": 1000,
            # Batch processing
            "parallel_threshold": 3,
            "max_workers_direct": 3,
            "max_workers_rag": 2,
            "max_workers_storm": 1,
            "max_workers_wiki": 1,
            # Enhancement options (include from base config if they exist)
            "use_wikidata_enhancement": base_data.get(
                "use_wikidata_enhancement", False
            ),
        }

        # Apply any settings from base config file if it exists
        if base_data:
            for key, value in base_data.items():
                if key in config_data:
                    config_data[key] = value

        # Apply CLI overrides (filter out None values)
        for key, value in overrides.items():
            if value is not None:
                config_data[key] = value

        return cls(**config_data)

    @classmethod
    def get_default(cls) -> "RetrievalConfig":
        """Get default configuration."""
        return cls()


# Global default instance
DEFAULT_RETRIEVAL_CONFIG = RetrievalConfig.get_default()
