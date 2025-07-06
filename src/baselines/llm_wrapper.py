"""
Simplified LLM wrapper for STORM integration with cleaner architecture.
"""
import time
import logging

logger = logging.getLogger(__name__)


class UnifiedLocalLLMWrapper:
    """
    Unified wrapper that handles both LiteLLM and direct calls for STORM.
    
    This replaces the complex multi-layer wrapper system with a single,
    focused implementation that handles all STORM integration needs.
    """
    
    def __init__(self, query_handler, max_tokens=512, temperature=0.7):
        self.query_handler = query_handler
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        # LiteLLM compatibility attributes
        self.model = 'qwen-local'
        self.model_name = 'qwen-local'
        self.kwargs = {'max_tokens': max_tokens, 'temperature': temperature}
        
        # Cost tracking (always 0 for local models)
        self.completion_cost = 0.0
        self.prompt_cost = 0.0
        
        # API compatibility
        self.api_key = None
        self.api_base = None
        
        logger.info(f"UnifiedLocalLLMWrapper initialized (available: {self.query_handler.is_available()})")
    
    def __call__(self, messages=None, **kwargs):
        """Handle direct calls from STORM components."""
        if messages is not None:
            return self.complete(messages, **kwargs)
        
        # Handle DSPy-style calls
        prompt = kwargs.get('prompt', str(kwargs))
        return self._generate_text(prompt, **kwargs)
    
    def complete(self, messages, max_tokens=None, temperature=None, **kwargs):
        """LiteLLM-compatible completion method for STORM."""
        try:
            # Extract prompt from messages
            prompt, system_prompt = self._parse_messages(messages)
            
            # Generate response
            response_text = self._generate_text(
                prompt, 
                system_prompt=system_prompt,
                max_tokens=max_tokens or self.max_tokens,
                temperature=temperature or self.temperature
            )
            
            # Return LiteLLM-compatible response
            return self._create_litellm_response(response_text, prompt)
            
        except Exception as e:
            logger.error(f"Completion failed: {e}")
            return self._create_error_response(str(e))
    
    def _generate_text(self, prompt, system_prompt=None, max_tokens=None, temperature=None):
        """Core text generation method."""
        if not self.query_handler.is_available():
            raise RuntimeError("Query handler not available")
        
        response = self.query_handler.query(
            prompt=str(prompt),
            system_prompt=system_prompt,
            max_new_tokens=max_tokens or self.max_tokens,
            temperature=temperature or self.temperature
        )
        
        if response is None:
            return "Error: No response generated"
        
        return str(response)
    
    def _parse_messages(self, messages):
        """Extract prompt and system prompt from message format."""
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
                elif role == "user":
                    prompt_parts.append(content)
            else:
                prompt_parts.append(str(message))
        
        return "\n".join(prompt_parts), system_prompt
    
    def _create_litellm_response(self, content, prompt):
        """Create LiteLLM-compatible response object."""
        class Response:
            def __init__(self, content, prompt):
                # Response structure
                self.choices = [self._create_choice(content)]
                self.model = 'qwen-local'
                self.usage = self._create_usage(content, prompt)
                self.id = f'local-{hash(prompt) % 10000}' if prompt else 'local-0'
                self.object = 'chat.completion'
                self.created = int(time.time())
                
                # Dict-like interface for STORM
                self._data = {
                    'choices': self.choices,
                    'model': self.model,
                    'usage': self.usage,
                    'id': self.id,
                    'object': self.object,
                    'created': self.created
                }
            
            def _create_choice(self, content):
                return type('Choice', (), {
                    'message': type('Message', (), {
                        'content': content,
                        'role': 'assistant'
                    })(),
                    'finish_reason': 'stop',
                    'index': 0
                })()
            
            def _create_usage(self, content, prompt):
                return type('Usage', (), {
                    'prompt_tokens': len(prompt.split()) if prompt else 0,
                    'completion_tokens': len(content.split()) if content else 0,
                    'total_tokens': (len(prompt.split()) if prompt else 0) + (len(content.split()) if content else 0)
                })()
            
            # Dict-like methods for STORM compatibility
            def __getitem__(self, key):
                return self._data[key]
            
            def __iter__(self):
                return iter(self._data)
            
            def get(self, key, default=None):
                return self._data.get(key, default)
            
            def keys(self):
                return self._data.keys()
        
        return Response(content, prompt)
    
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
                self.error = error_msg
                
                self._data = {
                    'choices': self.choices,
                    'model': self.model,
                    'error': error_msg
                }
            
            def __getitem__(self, key):
                return self._data[key]
            
            def get(self, key, default=None):
                return self._data.get(key, default)
        
        return ErrorResponse(error_msg)