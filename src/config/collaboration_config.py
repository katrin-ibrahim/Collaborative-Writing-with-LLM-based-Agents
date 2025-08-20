"""
Configuration for collaborative writing parameters.
Follows the same pattern as ModelConfig for consistency.
"""

import os
import yaml
from dataclasses import dataclass
from typing import Dict


@dataclass
class CollaborationConfig:
    """Configuration for collaborative writing parameters."""

    # Core collaboration parameters
    max_iterations: int = 3
    convergence_threshold: float = 0.85
    min_improvement_threshold: float = 0.02

    # Writer agent configuration
    writer_max_research_iterations: int = 2
    writer_use_external_knowledge: bool = True

    # Reviewer agent configuration
    reviewer_max_claims_per_article: int = 10
    reviewer_fact_check_timeout: int = 30

    # Theory of Mind configuration
    writer_tom_enabled: bool = True
    reviewer_tom_enabled: bool = True

    # Convergence detection configuration
    semantic_similarity_threshold: float = 0.95
    feedback_delta_threshold: float = 0.1

    def to_dict(self) -> Dict:
        """Convert to dictionary format for method configuration."""
        return {
            "collaboration.max_iterations": self.max_iterations,
            "collaboration.convergence_threshold": self.convergence_threshold,
            "collaboration.min_improvement_threshold": self.min_improvement_threshold,
            "writer.max_research_iterations": self.writer_max_research_iterations,
            "writer.use_external_knowledge": self.writer_use_external_knowledge,
            "reviewer.max_claims_per_article": self.reviewer_max_claims_per_article,
            "reviewer.fact_check_timeout": self.reviewer_fact_check_timeout,
            "writer.tom_enabled": self.writer_tom_enabled,
            "reviewer.tom_enabled": self.reviewer_tom_enabled,
            "collaboration.semantic_similarity_threshold": self.semantic_similarity_threshold,
            "collaboration.feedback_delta_threshold": self.feedback_delta_threshold,
        }

    @classmethod
    def from_dict(cls, config_dict: Dict) -> "CollaborationConfig":
        """Create CollaborationConfig from dictionary."""
        return cls(**{k: v for k, v in config_dict.items() if hasattr(cls, k)})

    @classmethod
    def from_yaml(cls, config_name: str) -> "CollaborationConfig":
        """Load collaboration config from YAML file or preset name."""
        # If it's a preset name, convert to file path
        if config_name in ["default", "aggressive", "conservative", "experimental"]:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(base_dir, f"collaboration_{config_name}.yaml")
        else:
            config_path = config_name

        # Check if file exists
        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"Collaboration config file not found: {config_path}"
            )

        # Load YAML file
        with open(config_path, "r") as f:
            config_data = yaml.safe_load(f)

        return cls.from_dict(config_data)
