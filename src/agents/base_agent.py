# src/agents/base_agent.py
from abc import ABC, abstractmethod
import os
from typing import Any, Dict
import logging

from utils.api import APIClient


logger = logging.getLogger(__name__)

class BaseAgent(ABC):
    """Abstract base class for all agents in the system."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize shared API client
        api_token = os.getenv('HF_TOKEN')
        self.api_client = APIClient(api_token=api_token)
    
    @abstractmethod
    def process(self, input_data: Any) -> Any:
        """Main processing method that each agent must implement."""
        pass
    
    def call_api(self, prompt: str, **kwargs) -> str:
        """Helper method to call language model API using shared client."""
        try:
            response = self.api_client.call_api(prompt, **kwargs)
            self.logger.info(f"API call successful, response length: {len(response)} chars")
            return response
        except Exception as e:
            self.logger.error(f"API call failed: {e}")
            return f"Error in content generation for: {prompt[:100]}... Please check API configuration."