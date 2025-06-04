# src/workflows/base_workflow.py 
from abc import ABC, abstractmethod
from typing import Any, Dict
import logging
import os
from utils.api import APIClient

logger = logging.getLogger(__name__)

class BaseWorkflow(ABC):
    """
    Abstract base class for all content generation workflows.
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        api_token = os.getenv('HF_TOKEN') or os.getenv('API_TOKEN')
        self.api_client = APIClient(api_token=api_token)
        
        self.logger.info(f"BaseWorkflow initialized with real API client for {self.__class__.__name__}")
    
    @abstractmethod
    def generate_content(self, topic: str) -> Any:
        """Generate content for the given topic."""
        pass
    
    def call_api(self, prompt: str, **kwargs) -> str:
        """
        Helper method to call the language model API using the shared client.
        """
        try:
            response = self.api_client.call_api(prompt, **kwargs)
            self.logger.info(f"API call successful, response length: {len(response)} chars")
            return response
        except Exception as e:
            self.logger.error(f"API call failed: {e}")
            # Return a meaningful fallback instead of placeholder
            return f"""# Error in Content Generation

An error occurred while generating content: {e}

Please check your API configuration and try again. The system attempted to generate content for the prompt:

{prompt[:200]}...

Make sure your API token is properly set in the HF_TOKEN or API_TOKEN environment variable."""