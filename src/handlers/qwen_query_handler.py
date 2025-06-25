import torch
import logging
from typing import Optional, Dict, Any
from .base_query_handler import BaseQueryHandler

logger = logging.getLogger(__name__)


class QwenQueryHandler(BaseQueryHandler):
    """Query handler for local Qwen2.5-VL model."""
    
    def __init__(self, model_path: str = "/storage/ukp/shared/shared_model_weights/Qwen2.5-VL-7B-Instruct"):
        self.model_path = model_path
        self.model = None
        self.processor = None
        self._load_model()
    
    def _load_model(self):
        """Load the Qwen model and processor."""
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
            
            logger.info(f"Loading Qwen model from {self.model_path}")
            
            self.processor = AutoProcessor.from_pretrained(self.model_path)
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_path,
                torch_dtype="auto",
                device_map="auto"
            )
            
            logger.info("Qwen model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load Qwen model: {e}")
            self.model = None
            self.processor = None
    
    def is_available(self) -> bool:
        """Check if model is loaded and ready."""
        return self.model is not None and self.processor is not None
    
    def query(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        """
        Execute a query with the Qwen model.
        
        Args:
            prompt: The main query text
            system_prompt: Optional system instruction
            **kwargs: Additional parameters (max_new_tokens, temperature, etc.)
            
        Returns:
            Generated response text
        """
        if not self.is_available():
            raise RuntimeError("Qwen model not available")
        
        # Prepare messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        # Get generation parameters
        max_new_tokens = kwargs.get('max_new_tokens', 512)
        temperature = kwargs.get('temperature', 0.7)
        
        try:
            # Apply chat template
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            # Process input (no images/videos for text-only)
            inputs = self.processor(
                text=[text],
                images=None,
                videos=None,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.model.device)
            
            # Generate response
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    pad_token_id=self.processor.tokenizer.eos_token_id
                )
            
            # Decode response
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            
            return output_text.strip()
            
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            return f"Error generating response: {str(e)}"
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up GPU memory."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()