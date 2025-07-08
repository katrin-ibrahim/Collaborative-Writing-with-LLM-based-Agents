from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class ModelConfig:
    """Configuration for model selection based on task complexity."""
    
    # Task-specific model assignments
    outline_model: str = "qwen2.5:14b"          # Balanced, for structure
    writing_model: str = "qwen2.5:32b"          # Quality, for content
    critique_model: str = "qwen2.5:14b"         # Reasoning, for self-critique
    polish_model: str = "qwen2.5:7b"       # Final polish
    
    # Self-RAG specific models
    retrieval_model: str = "mistral:7b-instruct"   # Fast retrieval decisions
    generation_model: str = "qwen2.5:14b"           # Main generation
    reflection_model: str = "qwen2.5:14b"           # Self-reflection
    
    # Default fallback
    default_model: str = "qwen2.5:7b"
    
    # Temperature settings per task
    temperatures: Dict[str, float] = None
    
    def __post_init__(self):
        if self.temperatures is None:
            self.temperatures = {
                "fast": 0.7,
                "outline": 0.5,
                "writing": 0.7,
                "critique": 0.3,
                "polish": 0.5,
                "retrieval": 0.3,
                "generation": 0.7,
                "reflection": 0.3
            }
    
    def get_model_for_task(self, task: str) -> str:
        """Get appropriate model for a specific task."""
        task_model_map = {
            "fast": self.default_model,
            "outline": self.outline_model,
            "writing": self.writing_model,
            "critique": self.critique_model,
            "polish": self.polish_model,
            "retrieval": self.retrieval_model,
            "generation": self.generation_model,
            "reflection": self.reflection_model
        }
        return task_model_map.get(task, self.default_model)
    
    def get_temperature_for_task(self, task: str) -> float:
        """Get appropriate temperature for a specific task."""
        return self.temperatures.get(task, 0.7)
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> "ModelConfig":
        """Create ModelConfig from dictionary."""
        return cls(**{k: v for k, v in config_dict.items() if hasattr(cls, k)})