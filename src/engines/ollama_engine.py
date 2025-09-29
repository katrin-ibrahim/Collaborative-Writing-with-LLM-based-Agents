"""
Simplified Ollama engine that focuses purely on Ollama API interaction.
Configuration is handled by ConfigContext.
"""

import json
import logging
import re
from ollama import Client
from typing import Any, Dict, List, Optional, Union

THINKING_LINE_LENGTH_THRESHOLD = (
    20  # Lines shorter than this are likely thinking fragments
)

from src.engines.base_engine import BaseEngine

logger = logging.getLogger(__name__)


class OllamaEngine(BaseEngine):
    """
    Ollama engine that provides both simple and LiteLLM-compatible interfaces.
    Simplified to focus only on Ollama API calls - no configuration parsing.
    """

    def __init__(
        self,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        host: str = "http://localhost:11434/",
        **kwargs,
    ):
        """
        Initialize Ollama engine with parameters from ConfigContext.

        Args:
            model: Ollama model name
            temperature: Generation temperature
            max_tokens: Maximum tokens to generate
            host: Ollama server URL
            **kwargs: Additional parameters
        """
        super().__init__(
            model=model, temperature=temperature, max_tokens=max_tokens, **kwargs
        )

        self.host = host
        self.client = Client(host=host)
        self._available_models = None

        logger.info(f"OllamaEngine initialized with host: {host}, model: {model}")

    def __call__(self, messages=None, **kwargs):
        """Make client callable for STORM compatibility."""
        if "max_tokens" not in kwargs and "max_output_tokens" not in kwargs:
            kwargs["max_tokens"] = self.max_tokens

        if messages is not None:
            return self.complete(messages, **kwargs)

        # Handle string prompts
        if isinstance(kwargs.get("prompt"), str):
            return self.complete(kwargs["prompt"], **kwargs)

        # Default case
        return self.complete(str(kwargs), **kwargs)

    def complete(self, messages: Union[str, List[Dict]], **kwargs) -> Any:
        """LiteLLM-compatible completion method."""
        try:
            # Parse messages using base class method
            prompt, system_prompt = self._parse_messages(messages)

            # Get parameters (allow override)
            model = kwargs.get("model", self.model)
            temperature = kwargs.get("temperature", self.temperature)
            max_tokens = kwargs.get("max_tokens", self.max_tokens)

            # Generate response using Ollama API
            response_text = self._call_ollama_api(
                prompt=prompt,
                system_prompt=system_prompt,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            # Create and return LiteLLM-compatible response
            return self._create_response(response_text, prompt)

        except Exception as e:
            logger.error(f"Completion failed: {e}")
            return self._create_error_response(str(e))

    def call_api(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Direct Ollama API call - returns just the text content.
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = self.client.chat(
                model=model or self.model,
                messages=messages,
                options={
                    "temperature": temperature or self.temperature,
                    "num_predict": max_tokens or self.max_tokens,
                },
            )

            content = response.message.content

            # Clean up thinking tags
            # Method 1: Remove closed thinking tags
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)

            # Method 2: Handle unclosed thinking tags at the start
            if content.strip().startswith("<think>"):
                # Find where actual content starts (look for patterns that indicate real content)
                lines = content.split("\n")
                content_start_idx = None

                for i, line in enumerate(lines):
                    line_stripped = line.strip()
                    # Skip empty lines and thinking content
                    if not line_stripped or line_stripped.startswith("<think>"):
                        continue
                    # Look for signs of real content (markdown headers, structured text, etc.)
                    if (
                        line_stripped.startswith("#")
                        or line_stripped.startswith("## ")
                        or len(line_stripped) > 50  # Substantial content line
                        or any(
                            keyword in line_stripped.lower()
                            for keyword in [
                                "content quality",
                                "structural",
                                "improvement",
                                "overall assessment",
                            ]
                        )
                    ):
                        content_start_idx = i
                        break

                if content_start_idx is not None:
                    content = "\n".join(lines[content_start_idx:])
                else:
                    # Fallback: remove first few lines that look like thinking
                    filtered_lines = []
                    skip_thinking = True
                    for line in lines:
                        if skip_thinking and (
                            not line.strip()
                            or line.strip().startswith("<think>")
                            or len(line.strip()) < THINKING_LINE_LENGTH_THRESHOLD
                        ):  # Very short lines are likely thinking fragments
                            continue
                        skip_thinking = False
                        filtered_lines.append(line)
                    content = "\n".join(filtered_lines)

            return content.strip()

        except Exception as e:
            logger.error(f"Ollama API call failed: {e}")
            raise RuntimeError(f"Ollama API error: {e}")

    def list_available_models(self) -> List[str]:
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

    def is_available(self) -> bool:
        """Check if Ollama server is accessible."""
        try:
            self.list_models()
            return len(self._available_models) > 0
        except:
            return False

    def extract_json(self, text: str) -> Optional[Dict]:
        """Extract JSON from text (utility method)."""
        json_pattern = r"({[\s\S]*})"
        match = re.search(json_pattern, text)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
        return None


# Backward compatibility alias for existing STORM integration
OllamaLiteLLMWrapper = OllamaEngine
