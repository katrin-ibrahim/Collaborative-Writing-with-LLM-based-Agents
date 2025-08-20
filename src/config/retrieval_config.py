"""
Centralized configuration for all retrieval and context generation parameters.
Eliminates redundancy and provides consistent naming.
"""

import logging
import os
import yaml
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RetrievalConfig:
    """Truly unified retrieval parameters - eliminates all redundancy."""

    # Retrieval Manager Configuration
    retrieval_manager_type: str = "wiki"  # Options: "wiki", "bm25_wiki", "faiss_wiki"

    # Single Source of Truth for Retrieval Flow
    num_queries: int = 5  # How many search queries to generate
    results_per_query: int = (
        3  # How many results per query (replaces max_results_per_query, search_top_k, wiki_max_articles)
    )
    max_content_pieces: int = (
        3  # How many sections/chunks per result (replaces wiki_max_sections)
    )
    final_passages: int = (
        10  # Final context size (replaces max_final_passages, retrieve_top_k)
    )
    queries_per_turn: int = 2

    # Content Processing
    passage_max_length: int = 2048
    passage_min_length: int = 1500

    # Batch Processing Configuration
    parallel_threshold: int = 3  # Threshold for parallel processing
    max_workers_direct: int = 3
    max_workers_rag: int = 2
    max_workers_storm: int = 1
    max_workers_wiki: int = 1

    # Semantic Filtering Configuration
    semantic_filtering_enabled: bool = True
    embedding_model: str = "all-MiniLM-L6-v2"
    supabase_embedding_model_name: str = "all-MiniLM-L6-v2"
    similarity_threshold: float = 0.4
    semantic_cache_size: int = 1000

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

        # Start with dataclass defaults
        default_instance = cls()
        config_data = {
            # Set the retrieval manager type
            "retrieval_manager_type": rm_type,
            # Use dataclass defaults instead of hardcoded experimental values
            "num_queries": default_instance.num_queries,
            "results_per_query": default_instance.results_per_query,
            "max_content_pieces": default_instance.max_content_pieces,
            "final_passages": default_instance.final_passages,
            "queries_per_turn": default_instance.queries_per_turn,
            "passage_max_length": default_instance.passage_max_length,
            "passage_min_length": default_instance.passage_min_length,
            "semantic_filtering_enabled": default_instance.semantic_filtering_enabled,
            "embedding_model": default_instance.embedding_model,
            "similarity_threshold": default_instance.similarity_threshold,
            "semantic_cache_size": default_instance.semantic_cache_size,
            # Batch processing
            "parallel_threshold": default_instance.parallel_threshold,
            "max_workers_direct": default_instance.max_workers_direct,
            "max_workers_rag": default_instance.max_workers_rag,
            "max_workers_storm": default_instance.max_workers_storm,
            "max_workers_wiki": default_instance.max_workers_wiki,
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
