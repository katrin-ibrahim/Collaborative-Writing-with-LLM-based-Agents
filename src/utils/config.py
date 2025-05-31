import yaml
from typing import Dict, Any

class Config:
    """Configuration management for the system."""
    
    def __init__(self, config_path: str = "configs/base_config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            # Return default configuration if file doesn't exist
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Default configuration values."""
        return {
            "model": {
                "name": "microsoft/DialoGPT-medium",
                "max_tokens": 1024,
                "temperature": 0.7
            },
            "retrieval": {
                "top_k": 5,
                "search_engine": "duckduckgo"
            },
            "evaluation": {
                "rouge_types": ["rouge1", "rouge2", "rougeL"]
            }
        }
    
    def get(self, key: str, default=None):
        """Get configuration value by key."""
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value