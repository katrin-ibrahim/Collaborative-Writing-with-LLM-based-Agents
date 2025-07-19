import time

import logging
from typing import Any, Dict, List, Union

logger = logging.getLogger(__name__)


class OllamaLiteLLMWrapper:
    """LiteLLM-compatible wrapper for Ollama to work with STORM."""

    def __init__(
        self,
        ollama_client,
        model: str = "qwen2.5:7b",
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ):
        self.client = ollama_client
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        # LiteLLM compatibility attributes
        self.model_name = model
        self.kwargs = {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        logger.info(f"OllamaLiteLLMWrapper initialized with model: {model}")

    def __call__(self, messages=None, **kwargs):
        """Make wrapper callable for STORM compatibility."""
        if "max_tokens" not in kwargs and "max_output_tokens" not in kwargs:
            kwargs["max_tokens"] = self.max_tokens
        if messages is not None:
            return self.complete(messages, **kwargs)

        # Handle string prompts
        if isinstance(kwargs.get("prompt"), str):
            return self._generate(kwargs["prompt"], **kwargs)

        # Default case
        return self.complete(str(kwargs), **kwargs)

    def complete(self, messages: Union[str, List[Dict]], **kwargs) -> Any:
        """LiteLLM-compatible completion method."""
        try:
            # Parse messages
            prompt, system_prompt = self._parse_messages(messages)
            # print(f"Parsed prompt: {prompt}, system prompt: {system_prompt}")
            # Get parameters
            model = kwargs.get("model", self.model)
            temperature = kwargs.get("temperature", self.temperature)
            max_tokens = kwargs.get("max_tokens", self.max_tokens)
            print(
                f"Using model: {model}, temperature: {temperature}, max_tokens: {max_tokens}"
            )

            # Generate response
            response_text = self.client.call_api(
                prompt=prompt,
                system_prompt=system_prompt,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            # print(f"Generated response: {response_text}")
            created_response = self._create_response(response_text, prompt)
            # print(f"Created response: {created_response}")
            # Return LiteLLM-compatible response
            return created_response

        except Exception as e:
            logger.error(f"Completion failed: {e}")
            return self._create_error_response(str(e))

    def _generate(self, prompt: str, **kwargs) -> str:
        """Direct generation method."""
        response_text = self.client.call_api(
            prompt=prompt,
            model=kwargs.get("model", self.model),
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
        )
        return response_text

    def _parse_messages(self, messages: Union[str, List[Dict]]) -> tuple:
        """Parse various message formats."""
        if isinstance(messages, str):
            return messages, None

        if not isinstance(messages, list):
            return str(messages), None

        prompt_parts = []
        system_prompt = None

        for message in messages:
            if isinstance(message, dict):
                role = message.get("role", "")
                content = message.get("content", "")

                if role == "system":
                    system_prompt = content
                elif role in ["user", "assistant"]:
                    prompt_parts.append(content)
            else:
                prompt_parts.append(str(message))

        return "\n".join(prompt_parts), system_prompt

    def _create_response(self, content: str, prompt: str) -> Any:
        """Create LiteLLM-compatible response object."""

        class Response:
            def __init__(self, content, model):
                self.content = content
                self.choices = [
                    type(
                        "Choice",
                        (),
                        {
                            "message": type(
                                "Message", (), {"content": content, "role": "assistant"}
                            )(),
                            "finish_reason": "stop",
                            "index": 0,
                        },
                    )()
                ]
                self.model = model
                self.usage = type(
                    "Usage",
                    (),
                    {
                        "prompt_tokens": len(prompt.split()),
                        "completion_tokens": len(content.split()),
                        "total_tokens": len(prompt.split()) + len(content.split()),
                    },
                )()
                self.id = f"ollama-{int(time.time())}"
                self.object = "chat.completion"
                self.created = int(time.time())

                # Dict-like interface
                self._data = {
                    "choices": self.choices,
                    "model": self.model,
                    "usage": self.usage,
                    "id": self.id,
                    "object": self.object,
                    "created": self.created,
                }

            def __getitem__(self, key):
                print(f"Accessing key: {key}")
                # print(f"Data keys: {self._data}")
                if isinstance(key, int):
                    # Return choice at index key
                    # print(f"Returning choice at index {key} with content: {self._data['choices'][key].message.content}")
                    return self._data["choices"][key].message.content
                else:
                    # Return value from dict
                    return self._data[key]

            def get(self, key, default=None):
                return self._data.get(key, default)

            def __str__(self):
                return self.content

        return Response(content, self.model)

    def _create_error_response(self, error_msg: str) -> Any:
        """Create error response in LiteLLM format."""
        return self._create_response(f"Error: {error_msg}", "")
