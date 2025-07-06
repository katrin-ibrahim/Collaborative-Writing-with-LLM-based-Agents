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
            
            # Debug: Log what STORM is asking for
            logger.debug(f"🔍 STORM Request - Prompt length: {len(str(prompt))} chars")
            logger.debug(f"🔍 STORM Request - First 150 chars: {repr(str(prompt)[:150])}")
            
            # Generate response
            response_text = self._generate_text(
                prompt, 
                system_prompt=system_prompt,
                max_tokens=max_tokens or self.max_tokens,
                temperature=temperature or self.temperature
            )
            
            # Debug: Log what we're returning to STORM
            logger.debug(f"🔍 STORM Response - Length: {len(response_text)} chars")
            logger.debug(f"🔍 STORM Response - First 150 chars: {repr(response_text[:150])}")
            
            # Ensure response is non-empty and valid
            if not response_text or response_text.strip() == "":
                logger.warning("⚠️ Empty response generated, providing fallback")
                response_text = "Unable to generate content for this request."
            
            # Return LiteLLM-compatible response
            return self._create_litellm_response(response_text, str(prompt))
            
        except Exception as e:
            logger.error(f"❌ Completion failed: {e}")
            error_response = f"Error generating response: {str(e)}"
            return self._create_litellm_response(error_response, str(messages))
            
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
        """Core text generation method with enhanced debugging."""
        if not self.query_handler.is_available():
            raise RuntimeError("Query handler not available")
        
        # Debug: Log input details
        prompt_str = str(prompt)
        logger.debug(f"🔍 LLM Input - Prompt length: {len(prompt_str)} chars")
        logger.debug(f"🔍 LLM Input - System prompt: {system_prompt is not None}")
        logger.debug(f"🔍 LLM Input - Max tokens: {max_tokens or self.max_tokens}")
        
        # Check for specific STORM prompts that might be problematic
        if "outline" in prompt_str.lower():
            logger.info("🔍 STORM Outline Generation Request Detected")
        elif "article" in prompt_str.lower():
            logger.info("🔍 STORM Article Generation Request Detected")
        elif "conversation" in prompt_str.lower():
            logger.info("🔍 STORM Conversation Request Detected")
        
        response = self.query_handler.query(
            prompt=prompt_str,
            system_prompt=system_prompt,
            max_new_tokens=max_tokens or self.max_tokens,
            temperature=temperature or self.temperature
        )
        
        # Debug: Log response details
        response_str = str(response) if response else ""
        logger.debug(f"🔍 LLM Output - Response length: {len(response_str)} chars")
        logger.debug(f"🔍 LLM Output - Word count: {len(response_str.split())} words")
        
        # Special handling for empty or very short responses
        if not response_str or len(response_str.strip()) < 10:
            logger.warning(f"⚠️ Very short/empty response: '{response_str}'")
            logger.warning(f"⚠️ For prompt: {prompt_str[:200]}...")
            
            # Provide context-appropriate fallback
            if "outline" in prompt_str.lower():
                response_str = "1. Introduction\n2. Main Topic\n3. Conclusion"
            elif "article" in prompt_str.lower():
                response_str = f"This is a brief article about the requested topic. Due to processing limitations, a full article could not be generated."
            else:
                response_str = "I understand the request but cannot provide a detailed response at this time."
        
        return response_str

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
                # Ensure content is a string
                self.content = str(content) if content else ""
                self.prompt = str(prompt) if prompt else ""
                
                # Response structure for LiteLLM
                self.choices = [self._create_choice(self.content)]
                self.model = 'qwen-local'
                self.usage = self._create_usage(self.content, self.prompt)
                self.id = f'local-{hash(self.prompt) % 10000}' if self.prompt else 'local-0'
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
            
            # String-like behavior for direct access
            def __str__(self):
                return self.content
            
            def __len__(self):
                return len(self.content)
            
            def __getitem__(self, key):
                # Handle slice operations for string-like access
                if isinstance(key, slice):
                    return self.content[key]
                # Handle dict-like access for STORM
                elif isinstance(key, str):
                    return self._data[key]
                else:
                    return self._data[key]
            
            def __iter__(self):
                return iter(self._data)
            
            def get(self, key, default=None):
                return self._data.get(key, default)
            
            def keys(self):
                return self._data.keys()
        
        return Response(content, prompt)
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
                
                # Store content for direct access
                self.content = content
                
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
            
            # String-like methods for slice operations
            def __str__(self):
                return self.content
            
            def __getitem__(self, key):
                if isinstance(key, slice):
                    return str(self.content)[key]
                return self._data[key]
            
            def __len__(self):
                return len(str(self.content))
        
        return Response(content, prompt)

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