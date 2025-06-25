import json
import logging
import requests
from typing import Optional

class OllamaClient:
    """Local Ollama client for development/testing"""
    
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3.2"):
        self.base_url = base_url
        self.model = model
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def call_api(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Call Ollama API with prompt"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            return response.json()["message"]["content"].strip()
        except Exception as e:
            self.logger.error(f"Ollama API call failed: {e}")
            raise