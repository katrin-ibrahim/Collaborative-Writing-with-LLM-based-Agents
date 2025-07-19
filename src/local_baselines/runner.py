"""
Local model runner for baseline experiments.
Uses locally hosted Qwen models instead of Ollama.
"""

import sys
import time
from pathlib import Path

import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional

# Add src directory to path
src_dir = Path(__file__).parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from local_baselines.utils import (
    build_direct_prompt,
    error_article,
    post_process_article,
)
from utils.data_models import Article
from utils.output_manager import OutputManager

logger = logging.getLogger(__name__)


class LocalModelWrapper:
    """Wrapper for local Qwen models."""

    def __init__(self, model_path: str, device: str = "auto"):
        self.model_path = model_path
        self.device = device
        self.model = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self):
        """Load the model and tokenizer."""
        try:
            logger.info(f"Loading model from {self.model_path}")

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, trust_remote_code=True
            )

            # Load model with SLURM-optimized settings for speed
            model_kwargs = {
                "torch_dtype": (
                    torch.float16 if torch.cuda.is_available() else torch.float32
                ),
                "device_map": self.device,
                "trust_remote_code": True,
                "low_cpu_mem_usage": True,
                "use_cache": True,
            }

            # Add SLURM-specific optimizations
            if torch.cuda.is_available():
                # Use mixed precision for faster inference
                model_kwargs["torch_dtype"] = (
                    torch.bfloat16
                )  # Often faster than float16
                # Optimize for inference
                model_kwargs["use_safetensors"] = True

            # Try flash attention, fallback if not available on SLURM
            try:
                model_kwargs["attn_implementation"] = "flash_attention_2"
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path, **model_kwargs
                )
            except Exception:
                logger.warning("Flash attention not available, using default attention")
                model_kwargs.pop("attn_implementation", None)
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path, **model_kwargs
                )

            # Set pad token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Verify GPU setup for optimal performance
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                logger.info(f"Model loaded on GPU with {gpu_memory:.1f}GB memory")
                # Ensure model is using GPU efficiently
                if hasattr(self.model, "hf_device_map"):
                    logger.info(f"Device map: {self.model.hf_device_map}")
            else:
                logger.warning("CUDA not available - using CPU (will be much slower)")

            logger.info(f"Model loaded successfully on device: {self.model.device}")

        except Exception as e:
            logger.error(f"Failed to load model from {self.model_path}: {e}")
            raise

    def generate(
        self, prompt: str, max_length: int = 1024, temperature: float = 0.3
    ) -> str:
        """Generate text using the local model with optimized settings for speed."""
        try:
            # Tokenize input - keep original context window for consistency with Ollama
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=min(
                    1024, self.model.config.max_position_embeddings - max_length
                ),
            )

            # Move to model device
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            # Generate with speed-optimized settings
            with torch.no_grad():
                # Clear cache before generation to prevent memory buildup
                if hasattr(torch.cuda, "empty_cache"):
                    torch.cuda.empty_cache()

                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    temperature=temperature,
                    do_sample=temperature > 0,  # Only sample if temp > 0
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=True,
                    num_beams=1,  # Greedy decoding for speed
                    early_stopping=True,  # Stop early when EOS is generated
                    repetition_penalty=1.05,  # Lighter penalty for speed
                    top_p=0.9,  # Nucleus sampling for faster generation
                    top_k=50,  # Limit vocabulary for speed
                )

            # Decode output
            generated_text = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
            )

            return generated_text.strip()

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise


class LocalBaselineRunner:
    """Runner for local model-based baseline experiments."""

    def __init__(
        self,
        model_path: str = "models/",
        output_manager: Optional[OutputManager] = None,
        device: str = "auto",
    ):
        self.model_path = Path(model_path)
        self.output_manager = output_manager
        self.device = device
        self.model_wrapper = None

        # Find and load the Qwen model
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the local model wrapper with specific Qwen models only."""
        # Define the three models we want to test
        target_models = [
            "models--Qwen2.5-32B-Instruct",  # 32B model
            "models--Qwen2.5-72B-Instruct",  # 72B model
            "models--Qwen3-14B",  # 14B model
        ]

        # Try to find one of the target models
        model_dir = None
        for model_name in target_models:
            candidate = self.model_path / model_name
            if candidate.exists():
                model_dir = candidate
                logger.info(f"Found and using target model: {model_dir}")
                break

        if model_dir is None:
            # List available models for debugging
            available_models = [d.name for d in self.model_path.iterdir() if d.is_dir()]
            logger.error(f"None of the target models found in {self.model_path}")
            logger.error(f"Target models: {target_models}")
            logger.error(f"Available models: {available_models}")
            raise FileNotFoundError(
                f"None of the target Qwen models (32B, 72B, 3B) found in {self.model_path}"
            )

        self.model_wrapper = LocalModelWrapper(str(model_dir), self.device)

    def run_direct_prompting(self, topic: str) -> Article:
        """Run direct prompting using the local model."""
        logger.info(f"Running Local Direct Prompting for: {topic}")

        prompt = build_direct_prompt(topic)

        try:
            start_time = time.time()

            # Generate response using local model - keep original parameters for Ollama consistency
            response = self.model_wrapper.generate(
                prompt,
                max_length=1024,  # Keep original for consistency with Ollama experiments
                temperature=0.3,  # Keep original for consistency with Ollama experiments
            )

            content = response
            logger.debug(f"Generated content length: {len(content)}")

            # Post-process the content for better quality
            if content:
                content = post_process_article(content, topic)
                logger.debug(f"Post-processed content length: {len(content)}")

                # REMOVED: Enhancement pass to eliminate double generation
                # This was the main bottleneck causing 50-minute generation times

            if content and not content.startswith("#"):
                content = f"# {topic}\n\n{content}"

            content_words = len(content.split()) if content else 0
            generation_time = time.time() - start_time

            # Clear GPU cache after generation to prevent memory buildup
            if hasattr(torch.cuda, "empty_cache"):
                torch.cuda.empty_cache()

            # Create article with same structure as baselines
            article = Article(
                title=topic,
                content=content,
                sections={},
                metadata={
                    "method": "direct",
                    "model": str(self.model_path),
                    "word_count": content_words,
                    "generation_time": generation_time,
                    "temperature": 0.3,  # Keep original for consistency
                    "max_tokens": 1024,  # Keep original for consistency
                    "optimized": True,  # Indicates memory/system optimizations only
                },
            )

            # Save article to OutputManager if available (using "direct" to match baselines)
            if self.output_manager:
                self.output_manager.save_article(article, "direct")

            logger.info(
                f"Local Direct Prompting completed for {topic} ({content_words} words)"
            )
            return article

        except Exception as e:
            logger.error(f"Direct prompting failed for '{topic}': {e}")
            return error_article(topic, str(e), "direct")
