"""
STORM configuration class.
"""

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class StormConfig:
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

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "StormConfig":
        """Create StormConfig from dictionary."""
        return cls(
            max_conv_turn=config_dict.get("max_conv_turn", 3),
            max_perspective=config_dict.get("max_perspective", 2),
            max_search_queries_per_turn=config_dict.get(
                "max_search_queries_per_turn", 3
            ),
            search_top_k=config_dict.get("search_top_k", 5),
            max_thread_num=config_dict.get("max_thread_num", 2),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert StormConfig to dictionary."""
        return {
            "max_conv_turn": self.max_conv_turn,
            "max_perspective": self.max_perspective,
            "max_search_queries_per_turn": self.max_search_queries_per_turn,
            "search_top_k": self.search_top_k,
            "max_thread_num": self.max_thread_num,
        }

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
