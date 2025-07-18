"""
Local model runner for baseline experiments.
Uses locally hosted Qwen models instead of Ollama.
"""
import sys
import time
from pathlib import Path
import logging
from typing import Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add src directory to path
src_dir = Path(__file__).parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from utils.output_manager import OutputManager
from local.data_models import Article
from local.utils import (
    build_direct_prompt,
    post_process_article,
    error_article,
)

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
                self.model_path,
                trust_remote_code=True
            )
            
            # Load model with optimizations
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map=self.device,
                trust_remote_code=True,
                attn_implementation="flash_attention_2",  # Much faster attention
                low_cpu_mem_usage=True,
                use_cache=True,
            )
            
            # Set pad token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            logger.info(f"Model loaded successfully on device: {self.model.device}")
            
        except Exception as e:
            logger.error(f"Failed to load model from {self.model_path}: {e}")
            raise
    
    def generate(self, prompt: str, max_length: int = 2048, temperature: float = 0.7) -> str:
        """Generate text using the local model."""
        try:
            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.model.config.max_position_embeddings - max_length
            )
            
            # Move to model device
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            # Generate with optimizations
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=True,
                    num_beams=1,  # Disable beam search for speed
                    early_stopping=False,
                )
            
            # Decode output
            generated_text = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
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
            "models--Qwen3-14B",             # 14B model
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
            raise FileNotFoundError(f"None of the target Qwen models (32B, 72B, 3B) found in {self.model_path}")
        
        self.model_wrapper = LocalModelWrapper(str(model_dir), self.device)
    
    def run_direct_prompting(self, topic: str) -> Article:
        """Run direct prompting using the local model."""
        logger.info(f"Running Local Direct Prompting for: {topic}")
        
        prompt = build_direct_prompt(topic)
        
        try:
            start_time = time.time()
            
            # Generate response using local model
            response = self.model_wrapper.generate(
                prompt,
                max_length=2048,
                temperature=0.7
            )
            
            content = response
            logger.debug(f"Generated content length: {len(content)}")
            
            # Post-process the content for better quality
            if content:
                content = post_process_article(content, topic)
                logger.debug(f"Post-processed content length: {len(content)}")
                
                # Optionally enhance content with additional pass
                if len(content.split()) < 800:  # Only enhance if content is too short
                    content = self._enhance_article_content(content, topic)
                    logger.debug(f"Enhanced content length: {len(content)}")
            
            if content and not content.startswith("#"):
                content = f"# {topic}\n\n{content}"
            
            content_words = len(content.split()) if content else 0
            generation_time = time.time() - start_time
            
            article = Article(
                title=topic,
                content=content,
                word_count=content_words,
                generation_time=generation_time,
                method="local_direct_prompting",
                timestamp=time.time(),
            )
            
            # Save article to OutputManager if available
            if self.output_manager:
                self.output_manager.save_article(article, "direct_prompting")
            
            return article
            
        except Exception as e:
            logger.error(f"Direct prompting failed for '{topic}': {e}")
            return error_article(topic, str(e), "local_direct_prompting")
    
    def _enhance_article_content(self, content: str, topic: str) -> str:
        """Enhance article content with an additional model pass."""
        enhancement_prompt = f"""Please enhance and expand the following article about "{topic}". 
Make it more comprehensive, add more details, and ensure it's well-structured:

{content}

Enhanced article:"""
        
        try:
            enhanced_response = self.model_wrapper.generate(
                enhancement_prompt,
                max_length=1024,
                temperature=0.7
            )
            
            if enhanced_response and len(enhanced_response.split()) > len(content.split()):
                return enhanced_response
            else:
                return content
                
        except Exception as e:
            logger.warning(f"Content enhancement failed: {e}")
            return content
