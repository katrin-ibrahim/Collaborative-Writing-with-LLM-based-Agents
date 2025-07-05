import argparse
import logging
import json
import os
import time
import random
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, Dict, Any

from utils.freshwiki_loader import FreshWikiLoader
from evaluation.evaluator import ArticleEvaluator
from utils.logging_setup import setup_logging
from utils.data_models import Article
from config.storm_config import load_config
from handlers import QwenQueryHandler

# Add this before creating the retrieval manager
class MockSearchRM:
    def __init__(self, k=3):
        self.k = k
    
    def __call__(self, query_or_queries, exclude_urls=None, **kwargs):
        return self.retrieve(query_or_queries, exclude_urls, **kwargs)
    
    def retrieve(self, query_or_queries, exclude_urls=None, **kwargs):
        # Handle both single query and multiple queries
        if isinstance(query_or_queries, list):
            queries = query_or_queries
        else:
            queries = [query_or_queries]
        
        results = []
        for i, query in enumerate(queries):
            for j in range(self.k):
                # STORM expects this exact structure
                result = {
                    'url': f'https://mocksite{j+1}.com/article_{i+1}',
                    'snippets': [f'Mock search result {j+1} for query: {query}. This contains relevant information about the topic and provides detailed context for article generation.'],
                    'title': f'Article about {query} - Result {j+1}',
                    'description': f'Detailed information about {query}'
                }
                results.append(result)
        
        return results[:self.k * len(queries)]



class LocalLiteLLMWrapper:
    """Wrapper to make QwenQueryHandler compatible with both LiteLLM and DSPy interfaces."""
    
    def __init__(self, query_handler: QwenQueryHandler, max_tokens: int = 512, temperature: float = 0.7):
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
        
        print(f"[DEBUG] LocalLiteLLMWrapper initialized with handler available: {self.query_handler.is_available()}")
    
    def __call__(self, messages=None, **kwargs):
        """Handle both direct calls and DSPy calls."""
        if messages is not None:
            return self.complete(messages, **kwargs)
        else:
            # DSPy style call - extract from kwargs
            return self._handle_dspy_call(**kwargs)
    
    def _handle_dspy_call(self, **kwargs):
        """Handle DSPy-style calls."""
        print(f"[DEBUG] DSPy call with kwargs: {kwargs}")
        
        # DSPy often passes prompt directly
        if 'prompt' in kwargs:
            prompt = kwargs['prompt']
        elif 'messages' in kwargs:
            prompt = self._extract_prompt_from_messages(kwargs['messages'])
        else:
            # Extract from other possible DSPy formats
            prompt = str(kwargs)
        
        print(f"[DEBUG] DSPy extracted prompt: {prompt[:200]}...")
        
        try:
            response_text = self.query_handler.query(
                prompt=prompt,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            print(f"[DEBUG] DSPy response: {response_text[:200]}...")
            
            # Return in format DSPy expects - usually just the text
            return response_text
            
        except Exception as e:
            print(f"[DEBUG] DSPy call failed: {e}")
            return f"Error: {str(e)}"
    
    def _extract_prompt_from_messages(self, messages):
        """Extract prompt from message format."""
        if isinstance(messages, str):
            return messages
        elif isinstance(messages, list):
            prompt_parts = []
            for message in messages:
                if isinstance(message, dict):
                    if message.get("role") == "user":
                        prompt_parts.append(message.get("content", ""))
                else:
                    prompt_parts.append(str(message))
            return "\n".join(prompt_parts)
        else:
            return str(messages)
    
    def complete(self, messages, max_tokens=None, temperature=None, **kwargs):
        """LiteLLM-compatible completion method."""
        print(f"[DEBUG] LiteLLM complete() called")
        print(f"[DEBUG] Messages type: {type(messages)}")
        
        try:
            # Handle both string and list of messages
            if isinstance(messages, str):
                prompt = messages
                system_prompt = None
            else:
                prompt_parts = []
                system_prompt = None
                
                for message in messages:
                    if isinstance(message, dict):
                        role = message.get("role")
                        content = message.get("content")
                        
                        if role == "system":
                            system_prompt = content
                        elif role == "user":
                            prompt_parts.append(content)
                    else:
                        prompt_parts.append(str(message))
                
                prompt = "\n".join(prompt_parts)
            
            # Use provided parameters or defaults
            actual_max_tokens = max_tokens or self.max_tokens
            actual_temperature = temperature or self.temperature
            
            # Call our query handler
            response_text = self.query_handler.query(
                prompt=prompt,
                system_prompt=system_prompt,
                max_new_tokens=actual_max_tokens,
                temperature=actual_temperature
            )
            
            if response_text is None:
                response_text = "Error: Query handler returned None"
            
            if not isinstance(response_text, str):
                response_text = str(response_text) if response_text is not None else "Error: Invalid response type"
            
            # Create proper response structure
            class LiteLLMResponse:
                def __init__(self, content):
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
                    
                    # Make it iterable like a dict
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
            
            return LiteLLMResponse(response_text)
            
        except Exception as e:
            print(f"[DEBUG] Exception in complete: {e}")
            import traceback
            traceback.print_exc()
            
            # Return error response
            error_msg = f"Error: {str(e)}"
            
            class ErrorResponse:
                def __init__(self, error_msg):
                    self.choices = [type('Choice', (), {
                        'message': type('Message', (), {
                            'content': error_msg,
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
                    self.id = f'error-{hash(str(e)) % 10000}'
                    self.object = 'chat.completion'
                    self.created = int(time.time())
                    self.error = str(e)
                    
                    # Make it iterable
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

    """Wrapper to make QwenQueryHandler compatible with STORM's LiteLLM interface."""
    
    def __init__(self, query_handler: QwenQueryHandler, max_tokens: int = 512, temperature: float = 0.7):
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
        
        print(f"[DEBUG] LocalLiteLLMWrapper initialized with handler available: {self.query_handler.is_available()}")
    
    def __call__(self, messages, **kwargs):
        """Make the wrapper callable - STORM sometimes calls the LLM object directly."""
        print(f"[DEBUG] __call__ invoked with messages type: {type(messages)}")
        print(f"[DEBUG] __call__ kwargs: {kwargs}")
        return self.complete(messages, **kwargs)
    
    def complete(self, messages, max_tokens=None, temperature=None, **kwargs):
        """LiteLLM-compatible completion method."""
        print(f"[DEBUG] complete() called")
        print(f"[DEBUG] Messages type: {type(messages)}")
        print(f"[DEBUG] Messages content: {messages}")
        print(f"[DEBUG] max_tokens: {max_tokens}, temperature: {temperature}")
        print(f"[DEBUG] kwargs: {kwargs}")
        
        try:
            # Handle both string and list of messages
            if isinstance(messages, str):
                print(f"[DEBUG] Processing string message")
                prompt = messages
                system_prompt = None
            else:
                print(f"[DEBUG] Processing message list with {len(messages)} items")
                prompt_parts = []
                system_prompt = None
                
                for i, message in enumerate(messages):
                    print(f"[DEBUG] Message {i}: type={type(message)}, content={message}")
                    if isinstance(message, dict):
                        role = message.get("role")
                        content = message.get("content")
                        print(f"[DEBUG] Message {i}: role={role}, content_length={len(str(content)) if content else 0}")
                        
                        if role == "system":
                            system_prompt = content
                        elif role == "user":
                            prompt_parts.append(content)
                    else:
                        print(f"[DEBUG] Message {i} is not a dict, converting to string")
                        prompt_parts.append(str(message))
                
                prompt = "\n".join(prompt_parts)
            
            print(f"[DEBUG] Final prompt length: {len(prompt) if prompt else 0}")
            print(f"[DEBUG] System prompt length: {len(system_prompt) if system_prompt else 0}")
            
            # Use provided parameters or defaults
            actual_max_tokens = max_tokens or self.max_tokens
            actual_temperature = temperature or self.temperature
            
            print(f"[DEBUG] Using max_tokens={actual_max_tokens}, temperature={actual_temperature}")
            
            # Debug: Check if query handler is available
            is_available = self.query_handler.is_available()
            print(f"[DEBUG] Query handler available: {is_available}")
            
            if not is_available:
                raise RuntimeError("Query handler is not available")
            
            print(f"[DEBUG] Calling query handler...")
            
            # Call our query handler
            response_text = self.query_handler.query(
                prompt=prompt,
                system_prompt=system_prompt,
                max_new_tokens=actual_max_tokens,
                temperature=actual_temperature
            )
            
            print(f"[DEBUG] Query handler returned: type={type(response_text)}, length={len(str(response_text)) if response_text else 0}")
            
            # Debug: Check if response is None
            if response_text is None:
                print(f"[DEBUG] WARNING: Query handler returned None!")
                response_text = "Error: Query handler returned None"
            
            # Ensure response is a string
            if not isinstance(response_text, str):
                print(f"[DEBUG] Converting response from {type(response_text)} to string")
                response_text = str(response_text) if response_text is not None else "Error: Invalid response type"
            
            print(f"[DEBUG] Final response text: {response_text[:200]}...")
            
            # Create proper response structure
            class LiteLLMResponse:
                def __init__(self, content):
                    print(f"[DEBUG] Creating LiteLLMResponse with content length: {len(content)}")
                    
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
                    
                    # Make it iterable like a dict
                    self._data = {
                        'choices': self.choices,
                        'model': self.model,
                        'usage': self.usage,
                        'id': self.id,
                        'object': self.object,
                        'created': self.created
                    }
                    
                    print(f"[DEBUG] LiteLLMResponse created successfully")
                
                def __getitem__(self, key):
                    print(f"[DEBUG] LiteLLMResponse.__getitem__({key})")
                    return self._data[key]
                
                def __iter__(self):
                    print(f"[DEBUG] LiteLLMResponse.__iter__() called")
                    return iter(self._data)
                
                def get(self, key, default=None):
                    print(f"[DEBUG] LiteLLMResponse.get({key}, {default})")
                    return self._data.get(key, default)
                
                def keys(self):
                    return self._data.keys()
                
                def values(self):
                    return self._data.values()
                
                def items(self):
                    return self._data.items()
            
            response_obj = LiteLLMResponse(response_text)
            print(f"[DEBUG] Returning LiteLLMResponse object: {type(response_obj)}")
            return response_obj
            
        except Exception as e:
            print(f"[DEBUG] Exception in LocalLiteLLMWrapper.complete: {e}")
            print(f"[DEBUG] Exception type: {type(e)}")
            print(f"[DEBUG] Messages type: {type(messages)}")
            print(f"[DEBUG] Messages content: {messages}")
            
            import traceback
            print(f"[DEBUG] Full traceback:")
            traceback.print_exc()
            
            # Return error response
            error_msg = f"Error: {str(e)}"
            
            class ErrorResponse:
                def __init__(self, error_msg):
                    print(f"[DEBUG] Creating ErrorResponse with: {error_msg}")
                    
                    self.choices = [type('Choice', (), {
                        'message': type('Message', (), {
                            'content': error_msg,
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
                    self.id = f'error-{hash(str(e)) % 10000}'
                    self.object = 'chat.completion'
                    self.created = int(time.time())
                    self.error = str(e)
                    
                    # Make it iterable
                    self._data = {
                        'choices': self.choices,
                        'model': self.model,
                        'usage': self.usage
                    }
                    
                    print(f"[DEBUG] ErrorResponse created successfully")
                
                def __getitem__(self, key):
                    print(f"[DEBUG] ErrorResponse.__getitem__({key})")
                    return self._data[key]
                
                def __iter__(self):
                    print(f"[DEBUG] ErrorResponse.__iter__() called")
                    return iter(self._data)
                
                def get(self, key, default=None):
                    print(f"[DEBUG] ErrorResponse.get({key}, {default})")
                    return self._data.get(key, default)
            
            error_obj = ErrorResponse(error_msg)
            print(f"[DEBUG] Returning ErrorResponse object: {type(error_obj)}")
            return error_obj


    """Wrapper to make QwenQueryHandler compatible with STORM's LiteLLM interface."""
    
    def __init__(self, query_handler: QwenQueryHandler, max_tokens: int = 512, temperature: float = 0.7):
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
    
    def __call__(self, messages, **kwargs):
        """Make the wrapper callable - STORM sometimes calls the LLM object directly."""
        return self.complete(messages, **kwargs)
    
   
    def complete(self, messages, max_tokens=None, temperature=None, **kwargs):
        """LiteLLM-compatible completion method."""
        try:
            # Handle both string and list of messages
            if isinstance(messages, str):
                # Direct string prompt
                prompt = messages
                system_prompt = None
            else:
                # Extract prompt from message list
                prompt_parts = []
                system_prompt = None
                
                for message in messages:
                    if isinstance(message, dict):
                        if message.get("role") == "system":
                            system_prompt = message.get("content")
                        elif message.get("role") == "user":
                            prompt_parts.append(message.get("content"))
                    else:
                        # Handle case where message is not a dict
                        prompt_parts.append(str(message))
                
                prompt = "\n".join(prompt_parts)
            
            # Use provided parameters or defaults
            actual_max_tokens = max_tokens or self.max_tokens
            actual_temperature = temperature or self.temperature
            
            # Debug: Check if query handler is available
            if not self.query_handler.is_available():
                raise RuntimeError("Query handler is not available")
            
            # Call our query handler
            response_text = self.query_handler.query(
                prompt=prompt,
                system_prompt=system_prompt,
                max_new_tokens=actual_max_tokens,
                temperature=actual_temperature
            )
            
            # Debug: Check if response is None
            if response_text is None:
                response_text = "Error: Query handler returned None"
                print(f"Warning: Query handler returned None for prompt: {prompt[:100]}...")
            
            # Ensure response is a string
            if not isinstance(response_text, str):
                response_text = str(response_text) if response_text is not None else "Error: Invalid response type"
            
            # Create proper response structure
            # Create proper response structure
            class LiteLLMResponse:
                def __init__(self, content):
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
                    
                    # Make it iterable like a dict
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
            
        except Exception as e:
            print(f"Error in LocalLiteLLMWrapper.complete: {e}")
            print(f"Error type: {type(e)}")
            print(f"Messages type: {type(messages)}")
            print(f"Messages content: {messages}")
            
            # Return error response
            error_msg = f"Error: {str(e)}"
            
            class ErrorResponse:
                def __init__(self, error_msg):
                    self.choices = [type('Choice', (), {
                        'message': type('Message', (), {
                            'content': error_msg,
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
                    self.id = f'error-{hash(str(e)) % 10000}'
                    self.object = 'chat.completion'
                    self.created = int(time.time())
                    self.error = str(e)
                    
                    # Make it iterable
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

import logging
class BaselinesRunner:
    """Runner for all baseline methods."""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize query handler
        if config.model_type == "local":
            self.query_handler = QwenQueryHandler(config.local_model_path)
            if not self.query_handler.is_available():
                raise RuntimeError("Local Qwen model not available")
            self.logger.info("Using local Qwen model")
        else:
            raise NotImplementedError("Only local models supported in this version")
    
    def run_direct_prompting(self, topic: str) -> Article:
        """Run direct prompting baseline - pure internal knowledge."""
        self.logger.info(f"Running Direct Prompting for: {topic}")
        
        prompt = f"""Write a comprehensive, well-structured article about "{topic}".

Requirements:
1. Create a detailed article with multiple sections
2. Use only your internal knowledge (no external sources needed)
3. Include an introduction, several main sections, and a conclusion
4. Write in an encyclopedic style similar to Wikipedia
5. Aim for 800-1200 words
6. Use clear headings and subheadings

Topic: {topic}

Article:"""

        try:
            start_time = time.time()
            content = self.query_handler.query(
                prompt=prompt,
                max_new_tokens=1024,
                temperature=0.7
            )
            generation_time = time.time() - start_time
            
            # Basic post-processing
            if not content.startswith("#"):
                content = f"# {topic}\n\n{content}"
            
            article = Article(
                title=topic,
                content=content,
                sections={},
                metadata={
                    "method": "direct_prompting",
                    "word_count": len(content.split()),
                    "generation_time": generation_time,
                    "model": "qwen_local"
                }
            )
            
            self.logger.info(f"Direct Prompting completed: {len(content.split())} words in {generation_time:.1f}s")
            return article
            
        except Exception as e:
            self.logger.error(f"Direct Prompting failed: {e}")
            return Article(
                title=topic,
                content=f"# {topic}\n\nError in direct prompting: {str(e)}",
                sections={},
                metadata={"error": str(e), "method": "direct_prompting"}
            )
    
    def run_storm(self, topic: str) -> Article:
        """Run STORM baseline using local model."""
        self.logger.info(f"Running STORM for: {topic}")
        
        # Fix permission issues - set writable directories
        work_dir = os.getcwd()
        storm_output_dir = os.path.join(work_dir, "storm_output")
        os.makedirs(storm_output_dir, exist_ok=True)
        
        # Override environment variables that might cause permission issues
        old_home = os.environ.get("HOME", "")
        old_tmpdir = os.environ.get("TMPDIR", "")
        
        os.environ["HOME"] = work_dir
        os.environ["TMPDIR"] = os.path.join(work_dir, "tmp")
        os.makedirs(os.environ["TMPDIR"], exist_ok=True)
        
        self.logger.info(f"Set HOME from {old_home} to {os.environ['HOME']}")
        self.logger.info(f"Set TMPDIR from {old_tmpdir} to {os.environ['TMPDIR']}")
        self.logger.info(f"Storm output dir: {storm_output_dir}")
        
        try:
            import sys
            try:
                import pysqlite3 as sqlite3
                sys.modules['sqlite3'] = sqlite3
                self.logger.info("✅ Using pysqlite3 as sqlite3 replacement for STORM")
            except ImportError:
                self.logger.warning("⚠️ pysqlite3 not available, STORM may fail")
            
            # Import STORM components
            from knowledge_storm import STORMWikiLMConfigs, STORMWikiRunner, STORMWikiRunnerArguments
            from knowledge_storm.rm import DuckDuckGoSearchRM
            
            llm_wrapper = LocalLiteLLMWrapper(
                self.query_handler,
                max_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature
            )
            # Configure DSPy to use our local model
            import dspy
            print(f"[DEBUG] Configuring DSPy...")

            # Create a DSPy-compatible LLM class
            class DSPyLocalLLM(dspy.LM):
                def __init__(self, query_handler, max_tokens=512, temperature=0.7):
                    self.query_handler = query_handler
                    self.max_tokens = max_tokens
                    self.temperature = temperature
                    super().__init__("qwen-local")
                
                def basic_request(self, prompt, **kwargs):
                    print(f"[DEBUG] DSPy basic_request called with prompt: {prompt[:200]}...")
                    
                    try:
                        response = self.query_handler.query(
                            prompt=prompt,
                            max_new_tokens=kwargs.get('max_tokens', self.max_tokens),
                            temperature=kwargs.get('temperature', self.temperature)
                        )
                        
                        print(f"[DEBUG] DSPy response: {response[:200]}...")
                        
                        # DSPy expects a list of completions
                        return [response] if response else ["Error: No response generated"]
                        
                    except Exception as e:
                        print(f"[DEBUG] DSPy request failed: {e}")
                        return [f"Error: {str(e)}"]
                
                def __call__(self, prompt, **kwargs):
                    return self.basic_request(prompt, **kwargs)

            # Create DSPy LLM instance
            dspy_lm = DSPyLocalLLM(self.query_handler, self.config.max_new_tokens, self.config.temperature)

            # Configure DSPy to use our LLM
            dspy.configure(lm=dspy_lm)
            print(f"[DEBUG] DSPy configured with local LLM")
            print(f"[DEBUG] Created LLM wrapper: {type(llm_wrapper)}")
            
            # Set up STORM LM configs with our wrapper
            print(f"[DEBUG] Creating STORM LM configs...")
            lm_config = STORMWikiLMConfigs()
            print(f"[DEBUG] Setting LM configurations...")
            
            lm_config.set_conv_simulator_lm(llm_wrapper)
            print(f"[DEBUG] Set conv_simulator_lm")
            
            lm_config.set_question_asker_lm(llm_wrapper)
            print(f"[DEBUG] Set question_asker_lm")
            
            lm_config.set_outline_gen_lm(llm_wrapper)
            print(f"[DEBUG] Set outline_gen_lm")
            
            lm_config.set_article_gen_lm(llm_wrapper)
            print(f"[DEBUG] Set article_gen_lm")
            
            lm_config.set_article_polish_lm(llm_wrapper)
            print(f"[DEBUG] Set article_polish_lm")
            
            # Set up retrieval
            print(f"[DEBUG] Creating MockSearchRM...")
            rm = MockSearchRM(k=self.config.search_top_k)
            print(f"[DEBUG] Created search RM: {type(rm)}")
            
            # Create STORM runner
            print(f"[DEBUG] Creating STORM runner arguments...")
            engine_args = STORMWikiRunnerArguments(
                output_dir=storm_output_dir,
                max_conv_turn=self.config.max_conv_turn,
                max_perspective=self.config.max_perspective,
                search_top_k=self.config.search_top_k,
                max_thread_num=self.config.max_thread_num
            )
            print(f"[DEBUG] Created engine args: {engine_args}")
            
            print(f"[DEBUG] Creating STORM runner...")
            storm_runner = STORMWikiRunner(engine_args, lm_config, rm)
            print(f"[DEBUG] Created STORM runner: {type(storm_runner)}")
            
            # Run STORM
            print(f"[DEBUG] Starting STORM run for topic: {topic}")
            start_time = time.time()
            
            result = storm_runner.run(
                topic=topic,
                do_research=True,
                do_generate_outline=True,
                do_generate_article=True,
                do_polish_article=self.config.enable_polish
            )
            
            print(f"[DEBUG] STORM run completed, result: {result}")
            generation_time = time.time() - start_time
            
        except Exception as e:
            print(f"[DEBUG] Exception in STORM setup/execution: {e}")
            print(f"[DEBUG] Exception type: {type(e)}")
            import traceback
            print(f"[DEBUG] Full traceback:")
            traceback.print_exc()
            raise e
            generation_time = time.time() - start_time
            
        #     # Read generated article
        #     topic_subdir = Path(storm_output_dir) / topic.replace(" ", "_").replace("/", "_")
            
        #     article_files = [
        #         topic_subdir / "storm_gen_article_polished.txt",
        #         topic_subdir / "storm_gen_article.txt"
        #     ]
            
        #     content = None
        #     for article_file in article_files:
        #         if article_file.exists():
        #             content = article_file.read_text(encoding='utf-8')
        #             break
            
        #     if not content:
        #         content = f"# {topic}\n\nSTORM completed but no article content found"
            
        #     article = Article(
        #         title=topic,
        #         content=content,
        #         sections={},
        #         metadata={
        #             "method": "storm_local",
        #             "word_count": len(content.split()),
        #             "generation_time": generation_time,
        #             "model": "qwen_local",
        #             "output_dir": str(topic_subdir)
        #         }
        #     )
            
        #     self.logger.info(f"STORM completed: {len(content.split())} words in {generation_time:.1f}s")
        #     return article
            
        # except Exception as e:
        #     self.logger.error(f"STORM failed: {e}")
        #     self.logger.error(f"Exception type: {type(e)}")
            
        #     # If it's a permission error, try to get more details
        #     if "Permission denied" in str(e):
        #         import traceback
        #         self.logger.error(f"Full traceback: {traceback.format_exc()}")
            
        #     # Restore environment
        #     if old_home:
        #         os.environ["HOME"] = old_home
        #     if old_tmpdir:
        #         os.environ["TMPDIR"] = old_tmpdir
                
        #     return Article(
        #         title=topic,
        #         content=f"# {topic}\n\nSTORM Error: {str(e)}",
        #         sections={},
        #         metadata={"error": str(e), "method": "storm_local"}
        #     )
    
    def run_all_baselines(self, topics, methods=None):
        """Run specified baselines on all topics."""
        if methods is None:
            methods = ["direct_prompting", "storm"]
        
        all_results = {}
        
        for i, topic in enumerate(topics, 1):
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"Processing {i}/{len(topics)}: {topic}")
            self.logger.info(f"{'='*60}")
            
            topic_results = {}
            
            # Run each baseline
            for method in methods:
                self.logger.info(f"Running {method}...")
                
                if method == "direct_prompting":
                    article = self.run_direct_prompting(topic)
                elif method == "storm":
                    article = self.run_storm(topic)
                else:
                    self.logger.warning(f"Unknown method: {method}")
                    continue
                
                topic_results[method] = {
                    "article": article,
                    "word_count": article.metadata.get("word_count", 0),
                    "success": "error" not in article.metadata
                }
            
            all_results[topic] = topic_results
            
            # Add delay between topics
            if i < len(topics):
                delay = 10.0  # Shorter delay for local models
                self.logger.info(f"Waiting {delay}s before next topic...")
                time.sleep(delay)
        
        return all_results


def main():
    parser = argparse.ArgumentParser(description="Baselines Runner with Local Models")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument("--num_topics", type=int, default=5)
    parser.add_argument("--methods", nargs="+", default=["direct_prompting", "storm"], 
                        help="Baselines to run")
    parser.add_argument("--skip_evaluation", action="store_true")
    parser.add_argument("--log_level", default="INFO")

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    logger.info("🚀 Baselines Runner with Local Models")
    logger.info(f"Methods: {', '.join(args.methods)}")

    # Load config
    config = load_config(args.config)
    logger.info(f"Model type: {config.model_type}")
    if config.model_type == "local":
        logger.info(f"Local model path: {config.local_model_path}")

    # Create results directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    methods_str = "_".join(args.methods)
    results_dir = Path("results") / f"{methods_str}_{args.num_topics}topics_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load topics
    logger.info("Loading topics from FreshWiki...")
    freshwiki_loader = FreshWikiLoader()
    entries = freshwiki_loader.get_evaluation_sample(args.num_topics)

    if not entries:
        logger.error("No FreshWiki entries found")
        return

    logger.info(f"Loaded {len(entries)} topics")

    # Initialize runner
    try:
        runner = BaselinesRunner(config)
    except Exception as e:
        logger.error(f"Failed to initialize runner: {e}")
        return

    # Initialize evaluator
    evaluator = None
    if not args.skip_evaluation:
        evaluator = ArticleEvaluator()
        logger.info("Evaluation enabled")

    # Run baselines
    topics = [entry.topic for entry in entries]
    all_results = runner.run_all_baselines(topics, args.methods)

    # Evaluate results
    final_results = {}
    for topic, baseline_results in all_results.items():
        topic_result = {"baselines": {}}
        
        # Get corresponding entry for evaluation
        entry = next((e for e in entries if e.topic == topic), None)
        
        for method, result in baseline_results.items():
            method_result = {
                "generation_results": {
                    "success": result["success"],
                    "word_count": result["word_count"],
                    "metadata": result["article"].metadata
                },
                "evaluation_results": {}
            }
            
            # Evaluate if possible
            if evaluator and result["success"] and entry:
                try:
                    metrics = evaluator.evaluate_article(result["article"], entry)
                    method_result["evaluation_results"] = metrics
                except Exception as e:
                    logger.warning(f"Evaluation failed for {method} on {topic}: {e}")
            
            topic_result["baselines"][method] = method_result
        
        final_results[topic] = topic_result

    # Save results
    results_file = results_dir / "results.json"
    output_data = {
        "configuration": {
            "model_type": config.model_type,
            "local_model_path": config.local_model_path if config.model_type == "local" else None,
            "methods": args.methods,
            "storm_settings": {
                "max_conv_turn": config.max_conv_turn,
                "max_perspective": config.max_perspective,
                "search_top_k": config.search_top_k,
                "enable_polish": config.enable_polish
            }
        },
        "results": final_results
    }

    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info("SUMMARY")
    logger.info(f"{'='*60}")

    for method in args.methods:
        successes = sum(1 for r in final_results.values() 
                       if r["baselines"].get(method, {}).get("generation_results", {}).get("success", False))
        total_words = sum(r["baselines"].get(method, {}).get("generation_results", {}).get("word_count", 0)
                         for r in final_results.values()
                         if r["baselines"].get(method, {}).get("generation_results", {}).get("success", False))
        avg_words = total_words / max(successes, 1)
        
        logger.info(f"{method}: {successes}/{len(final_results)} successful, {avg_words:.0f} avg words")

    logger.info(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()