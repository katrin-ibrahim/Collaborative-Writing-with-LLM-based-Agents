# src/agents/base_agent.py
from abc import ABC, abstractmethod
from typing import Any, Dict
import logging

logger = logging.getLogger(__name__)

class BaseAgent(ABC):
    """Abstract base class for all agents in the system."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def process(self, input_data: Any) -> Any:
        """Main processing method that each agent must implement."""
        pass
    
    def call_api(self, prompt: str, **kwargs) -> str:
        """
        Helper method to call Hugging Face API or local model.
        
        This is a placeholder that would be replaced with actual
        model inference code using transformers, vllm, or API calls.
        """
        # Simulate API call delay and response
        import time
        time.sleep(0.1)  # Simulate processing time
        
        # In real implementation, this would call your HuggingFace model
        # For now, return a structured mock response
        return f"Generated response for prompt: {prompt[:50]}..."