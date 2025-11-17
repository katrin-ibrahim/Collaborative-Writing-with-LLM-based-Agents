"""
Configuration for collaborative writing parameters.
Follows the same pattern as ModelConfig for consistency.
"""

from dataclasses import dataclass

from src.config.base_config import BaseConfig


@dataclass
class CollaborationConfig(BaseConfig):
    """Configuration for collaborative writing parameters."""

    # Core collaboration parameters
    max_iterations: int = 3
    min_iterations: int = 2  # don't stop before this
    resolution_rate_threshold: float = (
        0.75  # Realistic target after fixing underlying issues
    )
    stall_tolerance: int = 2  # consecutive low-improvement iters allowed
    min_improvement: float = 0.05  # required progress per iter (5%)
    small_tail_max: int = 3  # Reasonable number of remaining low/medium items
    writing_mode: str = "section"  # "section" or "article"
    revise_mode: str = (
        "pending"  # "pending" (batch: fast, 1 LLM call) or "section" (sequential: slow, 1 call per section)
    )
    should_self_refine: bool = False  # whether writers self-refine

    max_suggested_queries: int = 3  # maximum queries reviewer can suggest per iteration

    # Adaptive iteration extension parameters
    adaptive_iterations: bool = (
        False  # enable dynamic extension beyond initial max_iterations
    )
    adaptive_extension_max: int = 5  # upper bound if adaptation triggers
    adaptive_improvement_threshold: float = (
        0.02  # required improvement between check iterations (2%)
    )
    adaptive_check_iteration: int = (
        2  # iteration index (0-based) at which to evaluate extension condition
    )

    def get_file_pattern(self) -> str:
        return "configs/collaboration_{}.yaml"

    @classmethod
    def get_default(cls) -> "CollaborationConfig":
        """Get default configuration."""
        return cls()


DEFAULT_COLLABORATION_CONFIG = CollaborationConfig.get_default()
