"""
LiteLLM wrapper for local Qwen model integration with STORM.

This wrapper makes the local QwenQueryHandler compatible with STORM's
LiteLLM interface expectations.
"""

import time
import logging

logger = logging.getLogger(__name__)


class LocalLiteLLMWrapper:
    """
    Wrapper to make QwenQueryHandler compatible with STORM's LiteLLM interface.
    
    Handles both LiteLLM-style calls and direct calls from STORM components.
    """
    
    def __init__(self, query_handler, max_tokens: int = 512, temperature: float = 0.7):
        self.query_handler = query_handler
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        # Add LiteLLM-compatible attributes that STORM expects
        self.kwargs = {
            'max_tokens': max_tokens,
            'temperature': temperature,
            'model': 'qwen-local'
        }
        self.model_name = 'qwen-local'
        self.model = 'qwen-local'
        self.completion_cost = 0.0
        self.prompt_cost = 0.0
        self.api_key = None
        self.api_base = None
        
        logger.info(f"LocalLiteLLMWrapper initialized (handler available: {self.query_handler.is_available()})")
    
    def __call__(self, messages, **kwargs):
        """Make the wrapper callable - STORM sometimes calls the LLM object directly."""
        logger.debug(f"LocalLiteLLMWrapper.__call__ with messages type: {type(messages)}")
        return self.complete(messages, **kwargs)
    
    def complete(self, messages, max_tokens=None, temperature=None, **kwargs):
        """
        LiteLLM-compatible completion method.
        
        Args:
            messages: Either string prompt or list of message dicts
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional parameters
            
        Returns:
            LiteLLM-compatible response object
        """
        logger.debug(f"complete() called with messages type: {type(messages)}")
        
        try:
            # Extract prompt from messages
            prompt, system_prompt = self._extract_prompts(messages)
            
            # Use provided parameters or defaults
            actual_max_tokens = max_tokens or self.max_tokens
            actual_temperature = temperature or self.temperature
            
            logger.debug(f"Calling query handler (prompt length: {len(prompt) if prompt else 0})")
            
            # Check if query handler is available
            if not self.query_handler.is_available():
                raise RuntimeError("Query handler is not available")
            
            # Call our query handler
            response_text = self.query_handler.query(
                prompt=prompt,
                system_prompt=system_prompt,
                max_new_tokens=actual_max_tokens,
                temperature=actual_temperature
            )
            
            # Validate response
            if response_text is None:
                logger.warning("Query handler returned None")
                response_text = "Error: Query handler returned None"
            
            if not isinstance(response_text, str):
                logger.warning(f"Query handler returned {type(response_text)}, converting to string")
                response_text = str(response_text) if response_text is not None else "Error: Invalid response type"
            
            logger.debug(f"Query handler response length: {len(response_text)}")
            
            # Create LiteLLM-compatible response
            return self._create_response(response_text, prompt)
            
        except Exception as e:
            logger.error(f"complete() failed: {e}")
            return self._create_error_response(str(e))
    
    def _extract_prompts(self, messages):
        """Extract prompt and system prompt from messages."""
        if isinstance(messages, str):
            return messages, None
        
        prompt_parts = []
        system_prompt = None
        
        if isinstance(messages, list):
            for message in messages:
                if isinstance(message, dict):
                    role = message.get("role")
                    content = message.get("content", "")
                    
                    if role == "system":
                        system_prompt = content
                    elif role == "user":
                        prompt_parts.append(content)
                else:
                    prompt_parts.append(str(message))
        else:
            prompt_parts.append(str(messages))
        
        prompt = "\n".join(prompt_parts)
        return prompt, system_prompt
    
    def _create_response(self, content, prompt):
        """Create LiteLLM-compatible response object."""
        class LiteLLMResponse:
            def __init__(self, content, prompt):
                self.choices = [type('Choice', (), {
                    'message': type('Message', (), {
                        'content': content,
                        'role': 'assistant'
                    })(),
                    'finish_reason': 'stop',
                    'index': 0
                })()]
                self.model = 'qwen-local'
                self.usage = type('Usage', (), {
                    'prompt_tokens': len(prompt.split()) if prompt else 0,
                    'completion_tokens': len(content.split()) if content else 0,
                    'total_tokens': (len(prompt.split()) if prompt else 0) + (len(content.split()) if content else 0)
                })()
                self.id = f'local-{hash(prompt) % 10000}' if prompt else 'local-0'
                self.object = 'chat.completion'
                self.created = int(time.time())
                
                # Make it iterable like a dict for STORM compatibility
                self._data = {
                    'choices': self.choices,
                    'model': self.model,
                    'usage': self.usage,
                    'id': self.id,
                    'object': self.object,
                    'created': self.created
                }
            
            def __getitem__(self, key):
                return self._data[key]
            
            def __iter__(self):
                return iter(self._data)
            
            def get(self, key, default=None):
                return self._data.get(key, default)
            
            def keys(self):
                return self._data.keys()
            
            def values(self):
                return self._data.values()
            
            def items(self):
                return self._data.items()
        
        return LiteLLMResponse(content, prompt)
    
    def _create_error_response(self, error_msg):
        """Create error response in LiteLLM format."""
        class ErrorResponse:
            def __init__(self, error_msg):
                self.choices = [type('Choice', (), {
                    'message': type('Message', (), {
                        'content': f"Error: {error_msg}",
                        'role': 'assistant'
                    })(),
                    'finish_reason': 'error',
                    'index': 0
                })()]
                self.model = 'qwen-local'
                self.usage = type('Usage', (), {
                    'prompt_tokens': 0,
                    'completion_tokens': 0,
                    'total_tokens': 0
                })()
                self.id = f'error-{hash(str(error_msg)) % 10000}'
                self.object = 'chat.completion'
                self.created = int(time.time())
                self.error = error_msg
                
                # Make it iterable for STORM compatibility
                self._data = {
                    'choices': self.choices,
                    'model': self.model,
                    'usage': self.usage
                }
            
            def __getitem__(self, key):
                return self._data[key]
            
            def __iter__(self):
                return iter(self._data)
            
            def get(self, key, default=None):
                return self._data.get(key, default)
        
        return ErrorResponse(error_msg)