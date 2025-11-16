from dataclasses import dataclass
from typing import Dict, Optional

from src.config.base_config import BaseConfig


@dataclass
class ModelConfig(BaseConfig):
    """Configuration for model selection based on task complexity."""

    # Mode of operation: "local" or "ollama"
    mode: str = "ollama"

    # Override model to use for all tasks if set
    override_model: Optional[str] = None

    # Ollama host URL
    ollama_host: Optional[str] = None

    # Writer-Reviewer specific model assignments
    research_model: str = "qwen2.5:32b"  # Model for generating search queries
    outline_model: str = "qwen2.5:14b"  # Model for creating article outlines
    writer_model: str = "qwen2.5:32b"  # High-quality model for writing content
    revision_model: str = "qwen2.5:32b"  # Model for revising sections based on feedback
    self_refine_model: str = "qwen2.5:32b"  # Model for self-refinement
    reviewer_model: str = (
        "qwen2.5:32b"  # Model for holistic review and feedback generation
    )

    # Default fallback
    default_model: str = "qwen2.5:14b"

    # Local models configuration
    local_model_mapping: Optional[Dict[str, str]] = None

    # Ollama models configuration
    ollama_model_mapping: Optional[Dict[str, str]] = None

    # Temperature settings per task
    temperatures: Optional[Dict[str, float]] = None

    # Token limits per task
    token_limits: Optional[Dict[str, int]] = None

    def __post_init__(self):
        if self.temperatures is None:
            self.temperatures = {
                # Writer-Reviewer specific temperatures
                "research": 0.3,  # More focused queries
                "outline": 0.4,  # More structured outlines
                "writer": 0.6,  # Balanced for creative content generation
                "revision": 0.4,  # Moderate for thoughtful revision
                "self_refine": 0.5,  # Balanced for self-refinement
                "reviewer": 0.3,  # Low for analytical review and feedback
            }

        if self.token_limits is None:
            self.token_limits = {
                # Writer-Reviewer specific token limits
                "outline": 6192,  # Short for outline creation
                "writer": 6192,  # Long for detailed section content
                "revision": 6192,  # Medium for revision with feedback
                "self_refine": 6192,  # Long for self-refinement
                "reviewer": 6192,  # Long for comprehensive review and feedback
            }

        if self.local_model_mapping is None:
            self.local_model_mapping = {
                "qwen2.5:7b": "models/models--Qwen2.5-7B-Instruct",
                "qwen2.5:14b": "models/models--Qwen2.5-14B-Instruct",
                "qwen2.5:32b": "models/models--Qwen2.5-32B-Instruct",
            }

        if self.ollama_model_mapping is None:
            self.ollama_model_mapping = {
                "qwen2.5:7b": "qwen2.5:7b",
                "qwen2.5:14b": "qwen2.5:14b",
                "qwen2.5:32b": "qwen2.5:32b",
                "gpt-oss:20b": "gpt-oss:20b",
            }

    def get_model_for_task(self, task: str) -> str:
        """Get appropriate model for a specific task.

        Returns the task-specific model. The override_model logic is handled
        during config initialization in from_yaml_with_overrides(), not here.
        """
        # Use task-specific models (which may have been set by override_model during init)
        task_model_map = {
            # Writer-Reviewer specific tasks
            "research": self.research_model,
            "outline": self.outline_model,
            "writer": self.writer_model,
            "revision": self.revision_model,
            "self_refine": self.self_refine_model,
            "reviewer": self.reviewer_model,
        }
        return task_model_map.get(task, self.default_model)

    def get_temperature_for_task(self, task: str) -> float:
        """Get appropriate temperature for a specific task."""
        temps = self.temperatures or {}
        return temps.get(task, 0.7)

    def get_token_limit_for_task(self, task: str) -> int:
        """Get appropriate token limit for a specific task."""
        limits = self.token_limits or {}
        return limits.get(task, 1000)

    @classmethod
    def get_default(cls) -> "ModelConfig":
        """Get default configuration."""
        return cls()

    @classmethod
    def from_yaml_with_overrides(
        cls, config_name: Optional[str] = None, **overrides
    ) -> "ModelConfig":
        """
        Load from YAML with overrides, applying special override_model logic.

        If override_model is set, it applies to all task-specific models UNLESS
        that specific model is explicitly provided in overrides.
        """
        # Extract override_model if present
        override_model = overrides.get("override_model")

        # Track which task models were explicitly provided
        task_model_keys = {
            "research_model" "outline_model",
            "writer_model",
            "revision_model",
            "self_refine_model",
            "reviewer_model",
        }

        explicitly_set_models = {
            key
            for key in task_model_keys
            if key in overrides and overrides[key] is not None
        }

        # Call parent implementation
        config = super().from_yaml_with_overrides(config_name, **overrides)

        # Apply override_model to any task models that weren't explicitly set
        if override_model:
            for model_key in task_model_keys:
                if model_key not in explicitly_set_models:
                    setattr(config, model_key, override_model)

        return config

    def get_file_pattern(self):
        return "configs/model_{}.yaml"


DEFAULT_MODEL_CONFIG = ModelConfig.get_default()
