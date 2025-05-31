import json
import logging
import re
import requests
from typing import Dict, List, Optional, Any

class APIClient:
    """Centralized API client for LLM interactions"""

    def __init__(self, api_url: str = None, api_token: Optional[str] = None):
        self.api_url = api_url or "https://router.huggingface.co/novita/v3/openai/chat/completions"
        self.headers = {"Authorization": f"Bearer {api_token}"} if api_token else {}
        logging.info(f"Using API with token: {api_token is not None}")

    def call_api(self, prompt: str, system_prompt: Optional[str] = None, model: str = "deepseek/deepseek-v3-0324") -> str:
        """Send prompt to API and return generated text"""
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        payload = {
            "messages": messages,
            "model": model,
        }
        response = requests.post(self.api_url, headers=self.headers, json=payload)
        response.raise_for_status()
        output = response.json()

        # Handle OpenAI-compatible response format
        if isinstance(output, dict) and "choices" in output:
            choices = output.get("choices", [])
            if choices and isinstance(choices[0], dict):
                message = choices[0].get("message", {})
                if isinstance(message, dict) and "content" in message:
                    return message["content"].strip()

        # Handle older Hugging Face formats
        if isinstance(output, list) and "generated_text" in output[0]:
            return output[0]["generated_text"].strip()
        if isinstance(output, dict) and "generated_text" in output:
            return output["generated_text"].strip()

        raise RuntimeError(f"Unexpected API response: {output}")

    def extract_json(self, text: str) -> Optional[Dict]:
        """Extract JSON from text that might have explanatory content around it"""
        # Find anything between { and }
        json_pattern = r"({[\s\S]*})"
        match = re.search(json_pattern, text)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
        return None
