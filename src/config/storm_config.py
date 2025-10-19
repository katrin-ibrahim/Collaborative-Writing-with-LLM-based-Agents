"""
STORM configuration class.
"""

from dataclasses import dataclass

from src.config.base_config import BaseConfig


@dataclass
class StormConfig(BaseConfig):
    """
    Configuration for STORM method parameters.

    Attributes:
        max_conv_turn: Maximum conversation turns for perspective simulation
        max_perspective: Maximum number of perspectives to generate
        max_search_queries_per_turn: Maximum search queries per conversation turn
        search_top_k: Number of top search results to retrieve
        max_thread_num: Maximum number of threads for parallel processing
    """

    max_conv_turn: int = 3
    max_perspective: int = 2
    max_search_queries_per_turn: int = 3
    search_top_k: int = 5
    max_thread_num: int = 2

    def adapt_to_retrieval_config(self, retrieval_config) -> "StormConfig":
        """
        Adapt STORM configuration to match retrieval configuration.

        Args:
            retrieval_config: RetrievalConfig instance

        Returns:
            New StormConfig with adapted parameters
        """
        return StormConfig(
            max_conv_turn=self.max_conv_turn,
            max_perspective=self.max_perspective,
            max_search_queries_per_turn=retrieval_config.queries_per_turn,
            search_top_k=retrieval_config.results_per_query,
            max_thread_num=self.max_thread_num,
        )

    def get_file_pattern(self) -> str:
        return "storm"

    @classmethod
    def get_default(cls) -> "StormConfig":
        """Get default configuration."""
        return cls()


# Global default instance
DEFAULT_STORM_CONFIG = StormConfig.get_default()
