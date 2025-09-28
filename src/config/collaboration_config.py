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
    convergence_threshold: float = 0.85
    min_improvement_threshold: float = 0.02

    # Writer agent configuration
    writer_use_external_knowledge: bool = True

    # Reviewer agent configuration
    reviewer_max_claims_per_article: int = 10
    reviewer_fact_check_timeout: int = 30

    # Theory of Mind configuration
    tom_enabled: bool = False

    # Convergence detection configuration
    semantic_similarity_threshold: float = 0.95
    feedback_delta_threshold: float = 0.1

    def get_file_pattern(self) -> str:
        return "collaboration_{}.yaml"

    @classmethod
    def get_default(cls) -> "CollaborationConfig":
        """Get default configuration."""
        return cls()


DEFAULT_COLLABORATION_CONFIG = CollaborationConfig.get_default()
