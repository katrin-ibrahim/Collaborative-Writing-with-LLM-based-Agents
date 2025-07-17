from dataclasses import dataclass
from typing import Dict


@dataclass
class ModelConfig:
    """Configuration for model selection based on task complexity."""

    # Task-specific model assignments
    outline_model: str = "qwen2.5:14b"  # Balanced, for structure
    writing_model: str = "qwen2.5:32b"  # Quality, for content
    critique_model: str = "qwen2.5:14b"  # Reasoning, for self-critique
    polish_model: str = "qwen2.5:7b"  # Final polish

    # Default fallback
    default_model: str = "qwen2.5:7b"

    # Temperature settings per task
    temperatures: Dict[str, float] = None
    # Token limits per task
    token_limits: Dict[str, int] = None

    def __post_init__(self):
        if self.temperatures is None:
            self.temperatures = {
                "fast": 0.5,  # Reduced for more focused queries
                "outline": 0.4,  # More structured outlines
                "writing": 0.6,  # Balanced creativity and accuracy
                "critique": 0.2,  # More conservative critique
                "polish": 0.3,  # More conservative polishing
                "retrieval": 0.1,  # Very focused retrieval decisions
                "generation": 0.6,  # Balanced generation
                "reflection": 0.2,  # Conservative reflection
            }

        if self.token_limits is None:
            self.token_limits = {
                "fast": 500,  # Increased for better query generation
                "outline": 600,  # Increased for better structure
                "writing": 2500,  # Increased for longer articles
                "critique": 1000,  # Increased for thorough critique
                "polish": 1000,  # Increased for better polishing
                "retrieval": 300,  # Increased for better retrieval
                "generation": 1500,  # Increased for better generation
                "reflection": 800,  # Increased for better reflection
            }

    def get_model_for_task(self, task: str) -> str:
        """Get appropriate model for a specific task."""
        task_model_map = {
            "fast": self.default_model,
            "outline": self.outline_model,
            "writing": self.writing_model,
            "critique": self.critique_model,
            "polish": self.polish_model,
        }
        return task_model_map.get(task, self.default_model)

    def get_temperature_for_task(self, task: str) -> float:
        """Get appropriate temperature for a specific task."""
        return self.temperatures.get(task, 0.7)

    def get_token_limit_for_task(self, task: str) -> int:
        """Get appropriate token limit for a specific task."""
        return self.token_limits.get(task, 1000)

    @classmethod
    def from_dict(cls, config_dict: Dict) -> "ModelConfig":
        """Create ModelConfig from dictionary."""
        return cls(**{k: v for k, v in config_dict.items() if hasattr(cls, k)})
