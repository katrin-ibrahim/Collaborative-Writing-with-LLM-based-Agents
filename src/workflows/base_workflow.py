from abc import ABC, abstractmethod
from typing import Any
import logging

logger = logging.getLogger(__name__)

class BaseWorkflow(ABC):
    """Abstract base class for all content generation workflows."""
    
    def __init__(self, config: dict):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def generate_content(self, topic: str) -> Any:
        """Generate content for the given topic."""
        pass
    
    def call_api(self, prompt: str, **kwargs) -> str:
        """Helper method to call language model API."""
        # Placeholder for actual model inference
        import time
        time.sleep(0.1)
        return f"Generated content for: {prompt[:50]}..."