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
    min_iterations: int = 1  # don't stop before this
    resolution_rate_threshold: float = 0.9
    stall_tolerance: int = 2  # consecutive low-improvement iters allowed
    min_improvement: float = 0.02  # required progress per iter (2%)
    small_tail_max: int = 2  # e.g., 5 remaining low/medium items
    writing_mode: str = "section"  # "section" or "article"
    revise_mode: str = "pending"  # "section" or "pending"
    should_self_refine: bool = False  # whether writers self-refine

    # Reviewer enhancement parameters
    ground_reviewer_with_research: bool = (
        True  # if True: reviewer sees full research chunks for fact-checking
    )
    max_suggested_queries: int = 3  # maximum queries reviewer can suggest per iteration

    def get_file_pattern(self) -> str:
        return "configs/collaboration_{}.yaml"

    @classmethod
    def get_default(cls) -> "CollaborationConfig":
        """Get default configuration."""
        return cls()


DEFAULT_COLLABORATION_CONFIG = CollaborationConfig.get_default()
