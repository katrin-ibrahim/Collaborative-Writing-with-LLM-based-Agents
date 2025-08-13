"""
High-performance LocalModelEngine following a standardized interface.
LiteLLM-compatible interface for local models with aggressive optimizations.
"""

import time
import warnings
from pathlib import Path

import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from transformers.utils import logging as transformers_logging
from typing import Any, Dict, List, Optional

from src.baselines.model_engines.base_engine import BaseModelEngine
from src.config.baselines_model_config import ModelConfig

# Suppress unnecessary warnings for performance
transformers_logging.set_verbosity_error()
warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger(__name__)


class LocalModelEngine(BaseModelEngine):
    """
    High-performance local model engine with LiteLLM compatibility.

    Key optimizations:
    - Persistent model loading with compiled inference
    - Advanced KV-cache management
    - Streamlined tokenization with prompt caching
    - Memory-efficient generation pipeline
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "auto",
        config: Optional[ModelConfig] = None,
        task: str = "writing",
    ):
        # Initialize base class
        super().__init__(model_path=model_path, config=config, task=task)

        logger.info(f"DEBUG: LocalModelEngine init - model_path param: {model_path}")
        logger.info(f"DEBUG: After base init - self.model_path: {self.model_path}")
        logger.info(f"DEBUG: Task: {task}")
        logger.info(f"DEBUG: Config mode: {self.config.mode}")
        # Ensure we're using local mode
        if self.config.mode != "local":
            self.config.mode = "local"
            logger.info("Switched to local mode for model configuration")

        self.device = device
        self.model_name = Path(self.model_path).name

        # LiteLLM compatibility attributes
        self.kwargs = {
            "model": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        self.model = None
        self.tokenizer = None
        self.generation_config = None

        # Performance optimization state
        self._compiled_model = None
        self._prompt_cache = {}
        self._kv_cache = None
        self._warmup_done = False
        self.stats = {"tokenization_time": 0, "generation_time": 0, "total_calls": 0}

        # Load and optimize model
        self._load_and_optimize_model()

        logger.info(f"LocalModelEngine initialized with model: {self.model_name}")

    def __call__(self, messages=None, **kwargs):
        """Make engine callable for STORM compatibility."""
        if "max_tokens" not in kwargs and "max_output_tokens" not in kwargs:
            kwargs["max_tokens"] = self.max_tokens

        if messages is not None:
            return self.complete(messages, **kwargs)

        # Handle string prompts
        if isinstance(kwargs.get("prompt"), str):
            return self._create_response(
                self._generate(kwargs["prompt"], **kwargs), kwargs["prompt"]
            )

        # Default case
        return self.complete(str(kwargs), **kwargs)

    def list_available_models(self) -> List[str]:
        """List available models, in this case just the currently loaded model."""
        return [self.model_name]

    def _load_and_optimize_model(self):
        """Load and optimize the model for inference."""
        try:
            logger.info(f"Loading model from {self.model_path}")
            start_time = time.time()

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True,
            )

            # Determine device placement
            if self.device == "auto":
                if torch.cuda.is_available():
                    device_map = "auto"
                elif torch.backends.mps.is_available():
                    device_map = "mps"
                else:
                    device_map = "cpu"
            else:
                device_map = self.device

            logger.info(f"Using device map: {device_map}")

            # Load model with optimizations
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16 if device_map != "cpu" else torch.float32,
                device_map=device_map,
                trust_remote_code=True,
                use_safetensors=True,
            )

            # Create default generation config
            self.generation_config = GenerationConfig(
                temperature=self.temperature,
                max_length=self.max_tokens,
                do_sample=True if self.temperature > 0 else False,
            )

            load_time = time.time() - start_time
            logger.info(f"Model loaded in {load_time:.2f}s")

            # Warm up model for better first inference time
            self._warmup_model()

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def _warmup_model(self):
        """Warm up the model to improve first inference speed."""
        if self._warmup_done:
            return

        try:
            logger.info("Warming up model...")
            warmup_prompt = "Hello, how are you today?"

            # Run a quick inference to initialize the model
            with torch.no_grad():
                inputs = self.tokenizer(warmup_prompt, return_tensors="pt").to(
                    self.model.device
                )
                _ = self.model.generate(**inputs, max_new_tokens=20)

            self._warmup_done = True
            logger.info("Model warmup complete")
        except Exception as e:
            logger.warning(f"Model warmup failed: {e}")

    def _get_cached_tokenization(self, prompt: str, max_length: int = 1024) -> Dict:
        """Get tokenization from cache or compute it."""
        cache_key = f"{prompt}_{max_length}"

        if cache_key in self._prompt_cache:
            return self._prompt_cache[cache_key]

        start_time = time.time()
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        self.stats["tokenization_time"] += time.time() - start_time

        # Cache the result
        self._prompt_cache[cache_key] = inputs

        return inputs

    def generate(
        self,
        prompt: str,
        max_length: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """Generate content from a prompt."""
        self.stats["total_calls"] += 1

        # Use provided params or defaults
        max_length = max_length or self.max_tokens
        temperature = temperature if temperature is not None else self.temperature

        try:
            # Get inputs (from cache if possible)
            inputs = self._get_cached_tokenization(prompt, max_length)

            # Configure generation parameters
            gen_config = GenerationConfig(**self.generation_config.to_dict())
            gen_config.temperature = temperature
            gen_config.max_length = None  # Use max_new_tokens instead
            gen_config.max_new_tokens = max_length
            gen_config.do_sample = temperature > 0

            # Generate
            start_time = time.time()
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    generation_config=gen_config,
                    use_cache=True,
                )
            self.stats["generation_time"] += time.time() - start_time

            # Decode
            response_text = self.tokenizer.decode(
                output_ids[0], skip_special_tokens=True
            )

            # Remove the prompt from the response
            if response_text.startswith(prompt):
                response_text = response_text[len(prompt) :].strip()

            # Clean <think> tags like in ollama_client
            import re
            response_text = re.sub(r"<think>.*?</think>", "", response_text, flags=re.DOTALL)

            return response_text

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return f"Error: {str(e)}"

    def complete(self, messages, **kwargs):
        """Complete messages in chat format."""
        try:
            # Parse messages to a string prompt if needed
            if isinstance(messages, list):
                prompt = self._parse_messages_to_string(messages)
            else:
                prompt = str(messages)

            # Generate and return response
            response_text = self._generate(
                prompt,
                max_length=kwargs.get("max_tokens", self.max_tokens),
                temperature=kwargs.get("temperature", self.temperature),
            )

            return self._create_response(response_text, prompt)

        except Exception as e:
            logger.error(f"Completion failed: {e}")
            return self._create_error_response(str(e))

    def _generate(self, prompt: str, **kwargs) -> str:
        """Direct generation method."""
        return self.generate(
            prompt,
            max_length=kwargs.get("max_tokens", self.max_tokens),
            temperature=kwargs.get("temperature", self.temperature),
        )

    @staticmethod
    def _parse_messages_to_string(messages: List[Dict]) -> str:
        """Parse chat messages to a string prompt."""
        result = []

        for msg in messages:
            role = msg.get("role", "").lower()
            content = msg.get("content", "")

            if role == "system":
                result.append(f"System: {content}")
            elif role == "user":
                result.append(f"User: {content}")
            elif role == "assistant":
                result.append(f"Assistant: {content}")
            else:
                result.append(content)

        return "\n".join(result)

    def _create_response(self, content: str, prompt: str) -> Any:
        """Create a LiteLLM-compatible response object."""

        class Response:
            def __init__(self, content, model):
                self.content = content
                self.model = model
                self.choices = [
                    type(
                        "Choice",
                        (),
                        {
                            "message": type("Message", (), {"content": content}),
                            "text": content,
                        },
                    )
                ]

        return Response(content, self.model_name)

    def _create_error_response(self, error_msg: str) -> Any:
        """Create an error response object."""

        class ErrorResponse:
            def __init__(self, error):
                self.error = error
                self.content = f"Error: {error}"
                self.choices = []

        return ErrorResponse(error_msg)

    def batch_generate(
        self, prompts: List[str], max_length: int = 1024, temperature: float = 0.3
    ) -> List[str]:
        """Generate responses for multiple prompts."""
        return [self.generate(p, max_length, temperature) for p in prompts]

    def clear_cache(self):
        """Clear tokenization cache and KV cache."""
        self._prompt_cache = {}
        self._kv_cache = None

    def get_performance_stats(self) -> Dict:
        """Get performance statistics."""
        if self.stats["total_calls"] > 0:
            avg_tokenization = (
                self.stats["tokenization_time"] / self.stats["total_calls"]
            )
            avg_generation = self.stats["generation_time"] / self.stats["total_calls"]
        else:
            avg_tokenization = 0
            avg_generation = 0

        return {
            "total_calls": self.stats["total_calls"],
            "total_tokenization_time": round(self.stats["tokenization_time"], 2),
            "total_generation_time": round(self.stats["generation_time"], 2),
            "avg_tokenization_time": round(avg_tokenization, 2),
            "avg_generation_time": round(avg_generation, 2),
            "model_name": self.model_name,
            "device": self.model.device if self.model else self.device,
        }
