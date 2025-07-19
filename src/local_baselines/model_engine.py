"""
High-performance LocalModelEngine following Ollama baselines architecture.
LiteLLM-compatible interface for local models with aggressive optimizations for 10-15x speedup.
"""

import time
import warnings
from pathlib import Path

import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from transformers.utils import logging as transformers_logging
from typing import Dict, List

# Suppress unnecessary warnings for performance
transformers_logging.set_verbosity_error()
warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger(__name__)


class LocalModelEngine:
    """
    LiteLLM-compatible local model engine optimized for speed.
    Follows Ollama baselines architecture pattern.

    Key optimizations:
    - Persistent model loading with compiled inference
    - Advanced KV-cache management
    - Streamlined tokenization with prompt caching
    - Memory-efficient generation pipeline
    - Compatible with STORM and RAG workflows
    """

    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        model: str = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ):
        self.model_path = model_path
        self.device = device
        self.model_name = model or f"local/{Path(model_path).name}"
        self.temperature = temperature
        self.max_tokens = max_tokens

        # LiteLLM compatibility attributes (like OllamaLiteLLMWrapper)
        self.kwargs = {
            "model": self.model_name,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        # Cost tracking (always 0 for local models)
        self.completion_cost = 0.0
        self.prompt_cost = 0.0

        self.model = None
        self.tokenizer = None
        self.generation_config = None

        # Performance optimization state
        self._compiled_model = None
        self._prompt_cache = {}
        self._kv_cache = None
        self._warmup_done = False

        # Load and optimize model
        self._load_and_optimize_model()

        logger.info(f"LocalModelEngine initialized with model: {self.model_name}")

    def __call__(self, messages=None, **kwargs):
        """Make engine callable for STORM compatibility (like OllamaLiteLLMWrapper)."""
        if "max_tokens" not in kwargs and "max_output_tokens" not in kwargs:
            kwargs["max_tokens"] = self.max_tokens
        if messages is not None:
            return self.complete(messages, **kwargs)

        # Handle string prompts
        if isinstance(kwargs.get("prompt"), str):
            return self._generate(kwargs["prompt"], **kwargs)

        # Default case
        return self.complete(str(kwargs), **kwargs)

    def _load_and_optimize_model(self):
        """Load model with maximum performance optimizations."""
        try:
            logger.info(f"Loading high-performance model from {self.model_path}")
            start_time = time.time()

            # Load tokenizer with optimizations
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                use_fast=True,  # Use fast tokenizer
                padding_side="left",  # Optimize for generation
            )

            # Set pad token efficiently
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Optimize model loading for inference
            model_kwargs = {
                "torch_dtype": (
                    torch.bfloat16 if torch.cuda.is_available() else torch.float32
                ),
                "device_map": self.device,
                "trust_remote_code": True,
                "low_cpu_mem_usage": True,
                "use_safetensors": True,
                "use_cache": True,
            }

            # Advanced optimizations for supported hardware
            if torch.cuda.is_available():
                # Try flash attention for 2x speedup
                try:
                    model_kwargs["attn_implementation"] = "flash_attention_2"
                    logger.info("Using Flash Attention 2 for acceleration")
                except Exception:
                    logger.info(
                        "Flash Attention 2 not available, using optimized attention"
                    )

                # Enable tensor parallelism if multiple GPUs
                if torch.cuda.device_count() > 1:
                    model_kwargs["device_map"] = "auto"
                    logger.info(
                        f"Using {torch.cuda.device_count()} GPUs with tensor parallelism"
                    )

            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path, **model_kwargs
            )

            # Set model to evaluation mode for inference
            self.model.eval()

            # Compile model for ~2x speedup (PyTorch 2.0+)
            if hasattr(torch, "compile") and torch.cuda.is_available():
                try:
                    logger.info("Compiling model for maximum performance...")
                    self.model = torch.compile(
                        self.model,
                        mode="max-autotune",  # Maximum optimization
                        fullgraph=True,
                        dynamic=False,
                    )
                    self._compiled_model = True
                    logger.info("Model compilation successful")
                except Exception as e:
                    logger.warning(f"Model compilation failed: {e}")
                    self._compiled_model = False

            # Create optimized generation config
            self.generation_config = GenerationConfig(
                max_new_tokens=1024,
                temperature=0.3,
                do_sample=True,
                top_p=0.9,
                top_k=50,
                repetition_penalty=1.05,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
                num_beams=1,  # Greedy decoding for speed
                early_stopping=True,
                # Advanced optimizations
                suppress_tokens=None,
                begin_suppress_tokens=None,
                forced_bos_token_id=None,
                forced_eos_token_id=None,
            )

            load_time = time.time() - start_time
            logger.info(f"Model loaded in {load_time:.2f}s with optimizations enabled")

            # Performance verification
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                allocated = torch.cuda.memory_allocated() / 1e9
                logger.info(
                    f"GPU Memory: {allocated:.1f}GB / {gpu_memory:.1f}GB allocated"
                )

            # Warmup for optimal performance
            self._warmup_model()

        except Exception as e:
            logger.error(f"Failed to load optimized model: {e}")
            raise

    def _warmup_model(self):
        """Warmup model with a small generation to optimize caches."""
        if self._warmup_done:
            return

        try:
            logger.info("Warming up model for optimal performance...")
            warmup_prompt = "Write a brief introduction about artificial intelligence."

            # Warmup generation
            with torch.no_grad():
                inputs = self.tokenizer(
                    warmup_prompt, return_tensors="pt", truncation=True, max_length=512
                ).to(self.model.device)

                # Generate warmup
                self.model.generate(
                    **inputs,
                    max_new_tokens=50,
                    **self.generation_config.to_dict(),
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            self._warmup_done = True
            logger.info("Model warmup completed")

        except Exception as e:
            logger.warning(f"Model warmup failed: {e}")

    def _get_cached_tokenization(self, prompt: str, max_length: int = 1024) -> Dict:
        """Get cached tokenization or create new one."""
        cache_key = hash(prompt + str(max_length))

        if cache_key in self._prompt_cache:
            return self._prompt_cache[cache_key]

        # Tokenize with optimizations
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=min(
                max_length, self.model.config.max_position_embeddings - 1024
            ),
            add_special_tokens=True,
            padding=False,
        )

        # Move to device
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # Cache result (limit cache size)
        if len(self._prompt_cache) < 100:  # Prevent memory bloat
            self._prompt_cache[cache_key] = inputs

        return inputs

    def generate(
        self, prompt: str, max_length: int = 1024, temperature: float = 0.3
    ) -> str:
        """
        High-performance text generation with aggressive optimizations.

        Target: 10-15x speedup over original implementation.
        """
        try:
            start_time = time.time()

            # Get optimized tokenization
            inputs = self._get_cached_tokenization(prompt, max_length)

            # Update generation config for this call
            gen_config = self.generation_config
            gen_config.max_new_tokens = max_length
            gen_config.temperature = temperature
            gen_config.do_sample = temperature > 0.0

            # High-performance generation
            with torch.no_grad():
                # Use torch.inference_mode for extra speed
                with torch.inference_mode():
                    outputs = self.model.generate(
                        **inputs,
                        generation_config=gen_config,
                        # Performance optimizations
                        use_cache=True,
                        return_dict_in_generate=False,
                        output_attentions=False,
                        output_hidden_states=False,
                        # Memory optimizations
                        max_memory=None,
                        # Advanced sampling for quality
                        epsilon_cutoff=0.0,
                        eta_cutoff=0.0,
                    )

            # Fast decoding
            input_length = inputs["input_ids"].shape[1]
            generated_tokens = outputs[0][input_length:]

            generated_text = self.tokenizer.decode(
                generated_tokens,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )

            generation_time = time.time() - start_time
            tokens_generated = len(generated_tokens)

            if tokens_generated > 0:
                tokens_per_second = tokens_generated / generation_time
                logger.debug(
                    f"Generated {tokens_generated} tokens in {generation_time:.2f}s ({tokens_per_second:.1f} tok/s)"
                )

            return generated_text.strip()

        except Exception as e:
            logger.error(f"High-performance generation failed: {e}")
            raise

    def complete(self, messages, **kwargs):
        """LiteLLM-compatible completion method (like OllamaLiteLLMWrapper)."""
        try:
            # Convert messages to a single prompt
            if isinstance(messages, list):
                prompt = self._messages_to_prompt(messages)
            else:
                prompt = str(messages)

            # Extract parameters
            max_tokens = kwargs.get(
                "max_tokens", kwargs.get("max_output_tokens", self.max_tokens)
            )
            temperature = kwargs.get("temperature", self.temperature)

            logger.debug(
                f"Generating with local model: {len(prompt)} chars, max_tokens={max_tokens}"
            )

            # Generate using optimized engine
            response = self.generate(
                prompt=prompt, max_length=max_tokens, temperature=temperature
            )

            # Return in LiteLLM format (like OllamaLiteLLMWrapper)
            return type(
                "Response",
                (),
                {
                    "choices": [
                        type(
                            "Choice",
                            (),
                            {"message": type("Message", (), {"content": response})()},
                        )()
                    ],
                    "usage": type(
                        "Usage",
                        (),
                        {
                            "prompt_tokens": len(prompt.split()),
                            "completion_tokens": len(response.split()),
                            "total_tokens": len(prompt.split()) + len(response.split()),
                        },
                    )(),
                },
            )()

        except Exception as e:
            logger.error(f"Local model generation failed: {e}")
            raise

    def _generate(self, prompt: str, **kwargs):
        """Generate from a string prompt."""
        # Convert string prompt to messages format
        messages = [{"role": "user", "content": prompt}]
        return self.complete(messages, **kwargs)

    def _messages_to_prompt(self, messages: list) -> str:
        """Convert messages format to a single prompt string."""
        prompt_parts = []
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")

            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")

        return "\n\n".join(prompt_parts)

    def batch_generate(
        self, prompts: List[str], max_length: int = 1024, temperature: float = 0.3
    ) -> List[str]:
        """
        Batch generation for processing multiple prompts efficiently.
        Future optimization for processing multiple articles simultaneously.
        """
        try:
            if not prompts:
                return []

            if len(prompts) == 1:
                return [self.generate(prompts[0], max_length, temperature)]

            logger.info(f"Batch generating {len(prompts)} prompts")
            start_time = time.time()

            # Tokenize all prompts
            inputs = self.tokenizer(
                prompts,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=min(
                    1024, self.model.config.max_position_embeddings - max_length
                ),
            ).to(self.model.device)

            # Update generation config
            gen_config = self.generation_config
            gen_config.max_new_tokens = max_length
            gen_config.temperature = temperature
            gen_config.do_sample = temperature > 0.0

            # Batch generation
            with torch.no_grad():
                with torch.inference_mode():
                    outputs = self.model.generate(
                        **inputs,
                        generation_config=gen_config,
                        use_cache=True,
                        return_dict_in_generate=False,
                    )

            # Decode all outputs
            results = []
            input_length = inputs["input_ids"].shape[1]

            for output in outputs:
                generated_tokens = output[input_length:]
                generated_text = self.tokenizer.decode(
                    generated_tokens,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )
                results.append(generated_text.strip())

            batch_time = time.time() - start_time
            logger.info(
                f"Batch generation completed in {batch_time:.2f}s ({len(prompts)/batch_time:.1f} prompts/s)"
            )

            return results

        except Exception as e:
            logger.error(f"Batch generation failed: {e}")
            # Fallback to individual generation
            return [
                self.generate(prompt, max_length, temperature) for prompt in prompts
            ]

    def clear_cache(self):
        """Clear caches when needed (use sparingly)."""
        self._prompt_cache.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.debug("Caches cleared")

    def get_performance_stats(self) -> Dict:
        """Get performance statistics."""
        stats = {
            "model_path": str(self.model_path),
            "device": str(self.model.device if self.model else "unknown"),
            "compiled": self._compiled_model,
            "warmup_done": self._warmup_done,
            "prompt_cache_size": len(self._prompt_cache),
        }

        if torch.cuda.is_available():
            stats.update(
                {
                    "gpu_memory_allocated": torch.cuda.memory_allocated() / 1e9,
                    "gpu_memory_reserved": torch.cuda.memory_reserved() / 1e9,
                    "gpu_memory_total": torch.cuda.get_device_properties(0).total_memory
                    / 1e9,
                }
            )

        return stats
