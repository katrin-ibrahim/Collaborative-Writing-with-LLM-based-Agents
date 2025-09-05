"""
Simplified SLURM engine that focuses purely on local model inference.
Configuration is handled by ConfigContext.
"""

import time
import warnings
from pathlib import Path

import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from transformers.utils import logging as transformers_logging
from typing import Any, Dict, List, Optional, Union

from src.engines.base_engine import BaseEngine

# Suppress unnecessary warnings for performance
transformers_logging.set_verbosity_error()
warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger(__name__)


class SlurmEngine(BaseEngine):
    """
    High-performance local model engine with LiteLLM compatibility.
    Simplified to focus only on model inference - no configuration parsing.
    """

    def __init__(
        self,
        model_path: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        device: str = "auto",
        **kwargs,
    ):
        """
        Initialize SLURM engine with parameters from ConfigContext.

        Args:
            model_path: Path to local model
            temperature: Generation temperature
            max_tokens: Maximum tokens to generate
            device: Device for model placement
            **kwargs: Additional parameters
        """
        super().__init__(
            model=model_path, temperature=temperature, max_tokens=max_tokens, **kwargs
        )

        self.model_path = model_path
        self.device = device
        self.model_name = Path(model_path).name

        # Model components
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

        logger.info(f"SlurmEngine initialized with model: {self.model_name}")

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

    def complete(self, messages: Union[str, List[Dict]], **kwargs) -> Any:
        """LiteLLM-compatible completion method."""
        try:
            # Parse messages using base class method
            prompt, system_prompt = self._parse_messages(messages)

            # Combine system and user prompts
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"
            else:
                full_prompt = prompt

            # Get parameters (allow override)
            temperature = kwargs.get("temperature", self.temperature)
            max_tokens = kwargs.get("max_tokens", self.max_tokens)

            # Generate response
            response_text = self._generate(
                prompt=full_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            # Create and return LiteLLM-compatible response
            return self._create_response(response_text, full_prompt)

        except Exception as e:
            logger.error(f"Completion failed: {e}")
            return self._create_error_response(str(e))

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

            # Ensure pad_token is set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Determine device placement
            if self.device == "auto":
                if torch.cuda.is_available():
                    device_map = {"": 0}  # Place entire model on GPU 0
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
                torch_dtype=(
                    torch.float16 if device_map not in ["cpu", "mps"] else torch.float32
                ),
                device_map=device_map,
                trust_remote_code=True,
                use_safetensors=True,
            )

            # Create default generation config
            self.generation_config = GenerationConfig(
                temperature=self.temperature,
                max_new_tokens=self.max_tokens,
                do_sample=True if self.temperature > 0 else False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
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
            self._generate(warmup_prompt, temperature=0.1, max_tokens=10)
            self._warmup_done = True
            logger.info("Model warmup completed")

        except Exception as e:
            logger.warning(f"Model warmup failed: {e}")

    def _generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Generate text using the local model."""
        try:
            start_time = time.time()

            # Tokenize input
            inputs = self.tokenizer(prompt, return_tensors="pt")

            # Move to model device
            if hasattr(self.model, "device"):
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            # Update generation config for this call
            gen_config = GenerationConfig(
                temperature=temperature or self.temperature,
                max_new_tokens=max_tokens or self.max_tokens,
                do_sample=True if (temperature or self.temperature) > 0 else False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    generation_config=gen_config,
                    use_cache=True,
                )

            # Decode only the new tokens
            generated_text = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
            )

            generation_time = time.time() - start_time
            self.stats["generation_time"] += generation_time
            self.stats["total_calls"] += 1

            logger.debug(
                f"Generated {len(generated_text)} chars in {generation_time:.2f}s"
            )

            return generated_text.strip()

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise RuntimeError(f"Local model generation error: {e}")

    def get_stats(self) -> Dict[str, float]:
        """Get performance statistics."""
        return self.stats.copy()

    def clear_cache(self):
        """Clear any cached data."""
        self._prompt_cache.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
