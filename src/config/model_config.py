from dataclasses import dataclass
from typing import Dict, Optional

from src.config.base_config import BaseConfig


@dataclass
class ModelConfig(BaseConfig):
    """Configuration for model selection based on task complexity."""

    # Mode of operation: "local" or "ollama"
    mode: str = "local"

    # Override model to use for all tasks if set
    override_model: Optional[str] = None

    # Ollama host URL
    ollama_host: Optional[str] = None

    # Storm-specific model assignments
    conv_simulator_model: str = "qwen3:4b"  # Fast model for conversation simulation
    outline_model: str = "qwen3:4b"  # Balanced, for structure
    writing_model: str = "qwen3:4b"  # Quality, for content
    polish_model: str = "qwen3:4b"  # Final polish

    # Writer-Reviewer specific model assignments
    query_generation_model: str = "qwen3:4b"  # Fast model for generating search queries
    create_outline_model: str = "qwen3:4b"  # Model for creating article outlines
    section_selection_model: str = "qwen3:4b"  # Model for selecting relevant chunks
    writer_model: str = "qwen3:4b"  # High-quality model for writing content
    revision_model: str = "qwen3:4b"  # Model for revising sections based on feedback
    revision_batch_model: str = (
        "qwen3:4b"  # Model for revising multiple sections in batch
    )
    self_refine_model: str = "qwen3:4b"  # Model for self-refinement
    reviewer_model: str = (
        "qwen3:4b"  # Model for holistic review and feedback generation
    )

    # Default fallback
    default_model: str = "gemma3:4b"

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
                # Baseline method temperatures
                "conv_simulator": 0.5,  # Reduced for more focused queries
                "outline": 0.4,  # More structured outlines
                "writing": 0.6,  # Balanced creativity and accuracy
                "critique": 0.2,  # More conservative critique
                "polish": 0.3,  # More conservative polishing
                # Writer-Reviewer specific temperatures
                "query_generation": 0.3,  # Low for focused query generation
                "create_outline": 0.4,  # More structured outlines
                "section_selection": 0.2,  # Very low for analytical chunk selection
                "writer": 0.6,  # Balanced for creative content generation
                "revision": 0.4,  # Moderate for thoughtful revision
                "revision_batch": 0.4,  # Moderate for thoughtful batch revision
                "self_refine": 0.5,  # Balanced for self-refinement
                "reviewer": 0.3,  # Low for analytical review and feedback
            }

        if self.token_limits is None:
            self.token_limits = {
                # Baseline method token limits
                "conv_simulator": 1200,  # Increased for better query generation
                "outline": 1200,  # Increased for better structure
                "writing": 2500,  # Increased for better articles
                "critique": 1200,  # Increased for thorough critique
                "polish": 1200,  # Increased for better polishing
                # Writer-Reviewer specific token limits
                "query_generation": 800,  # Short for focused queries
                "create_outline": 600,  # Short for outline creation
                "section_selection": 1200,  # Short for chunk ID selection
                "writer": 2000,  # Long for detailed section content
                "revision": 3000,  # Medium for revision with feedback
                "revision_batch": 2000,  # Long for batch revisions
                "self_refine": 3000,  # Long for self-refinement
                "reviewer": 1800,  # Long for comprehensive review and feedback
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
                "qwen3:4b": "qwen3:4b",
                "qwen3:8b": "qwen3:8b",
            }

    def get_model_for_task(self, task: str) -> str:
        """Get appropriate model for a specific task."""
        # If override_model is set, use it for all tasks
        if self.override_model:
            return self.override_model

        # Otherwise use task-specific models
        task_model_map = {
            # Baseline method tasks
            "conv_simulator": self.default_model,
            "outline": self.outline_model,
            "writing": self.writing_model,
            "polish": self.polish_model,
            # Writer-Reviewer specific tasks
            "query_generation": self.query_generation_model,
            "create_outline": self.create_outline_model,
            "section_selection": self.section_selection_model,
            "writer": self.writer_model,
            "section_revision": self.revision_model,
            "revision_batch": self.revision_batch_model,
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

    def get_model_path(self, task: str) -> str:
        """Get full model path based on mode and task."""
        model_name = self.get_model_for_task(task)

        if self.mode == "local":
            # For local mode, convert the model name to a local path
            local_map = self.local_model_mapping or {}
            local_path = local_map.get(model_name)
            if local_path:
                # Ensure path is properly formatted
                return local_path
            # Fallback to a default model directory
            return f"models/{model_name.replace(':', '-')}"
        elif self.mode == "ollama":
            # For ollama mode, convert to appropriate ollama model name
            ollama_map = self.ollama_model_mapping or {}
            ollama_name = ollama_map.get(model_name, model_name)
            return ollama_name
        else:
            # For any other mode, just return the model name as is
            return model_name

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
            "query_generation_model",
            "create_outline_model",
            "section_selection_model",
            "writer_model",
            "revision_model",
            "revision_batch_model",
            "self_refine_model",
            "reviewer_model",
            "conv_simulator_model",
            "outline_model",
            "writing_model",
            "polish_model",
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
