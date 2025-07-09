import json
import logging
from ollama import Client
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class OllamaClient:
    """Clean Ollama client wrapper for baseline experiments."""

    def __init__(
        self,
        host: str = "http://10.167.31.201:11434/",
        default_model: str = "qwen2.5:7b",
    ):
        self.host = host
        self.default_model = default_model
        self.client = Client(host=host)
        self._available_models = None
        logger.info(
            f"OllamaClient initialized with host: {host}, default model: {default_model}"
        )

    def list_models(self) -> List[str]:
        """List available models on the Ollama server."""
        if self._available_models is None:
            try:
                response = self.client.list()
                self._available_models = [model.model for model in response.models]
                logger.info(f"Available Ollama models: {self._available_models}")
            except Exception as e:
                logger.error(f"Failed to list models: {e}")
                self._available_models = []
        return self._available_models

    def call_api(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs,
    ) -> str:
        """
        Call Ollama API with unified interface matching APIClient.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            model: Model to use (defaults to self.default_model)
            **kwargs: Additional parameters (temperature, max_tokens, etc.)
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = self.client.chat(
                model=model or self.default_model,
                messages=messages,
            )
            return response.message.content

        except Exception as e:
            logger.error(f"Ollama API call failed: {e}")
            raise RuntimeError(f"Ollama API error: {e}")

    def generate(self, prompt: str, model: Optional[str] = None, **kwargs) -> str:
        """Alternative interface for compatibility."""
        return self.call_api(prompt, model=model, **kwargs)

    def extract_json(self, text: str) -> Optional[Dict]:
        """Extract JSON from text (compatibility method)."""
        import re

        json_pattern = r"({[\s\S]*})"
        match = re.search(json_pattern, text)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
        return None

    def is_available(self) -> bool:
        """Check if Ollama server is accessible."""
        try:
            self.list_models()
            return len(self._available_models) > 0
        except:
            return False
