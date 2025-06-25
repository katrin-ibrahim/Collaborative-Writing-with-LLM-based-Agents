from abc import ABC, abstractmethod
from typing import Optional, Dict, Any


class BaseQueryHandler(ABC):
    """Abstract base class for all query handlers."""
    
    @abstractmethod
    def query(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        """
        Execute a query with the model.
        
        Args:
            prompt: The main prompt/query text
            system_prompt: Optional system prompt for instruction
            **kwargs: Additional model-specific parameters
            
        Returns:
            Generated response text
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if the query handler is ready and available.
        
        Returns:
            True if handler can process queries, False otherwise
        """
        pass
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit for cleanup."""
        pass