"""
SQLite-free DSPy integration - completely avoids SQLite instead of mocking it.
"""
import os
import logging

logger = logging.getLogger(__name__)


def setup_dspy_integration(query_handler, max_tokens=512, temperature=0.7):
    """
    Setup DSPy integration without any SQLite usage.
    
    This is the main function that other modules expect to import.
    """
    return setup_dspy_for_storm(query_handler, max_tokens, temperature)


def setup_dspy_for_storm(query_handler, max_tokens=512, temperature=0.7):
    """
    Setup DSPy for STORM without any SQLite usage.
    
    This approach completely disables SQLite features rather than mocking them.
    """
    logger.info("🔧 Setting up SQLite-free DSPy for STORM...")
    
    # Step 1: Disable all SQLite-dependent features
    os.environ['DSPY_CACHEDIR'] = ''
    os.environ['DSP_CACHE_DISABLED'] = '1'
    os.environ['DSPY_DISABLE_CACHE'] = '1'
    os.environ['DSPY_DISABLE_OPTIMIZERS'] = '1'
    os.environ['DSPY_DISABLE_TELEMETRY'] = '1'
    
    # Step 2: Import DSPy and create minimal wrapper
    try:
        import dspy
        from dspy.dsp import LM
        
        class SQLiteFreeWrapper(LM):
            """LM wrapper that avoids all SQLite-dependent code paths."""
            
            def __init__(self, query_handler, max_tokens=512, temperature=0.7):
                self.query_handler = query_handler
                self.max_tokens = max_tokens
                self.temperature = temperature
                
                # Call parent init with minimal parameters to avoid SQLite code
                super().__init__("qwen-local")
                
                # Explicitly disable caching
                self._cache = None
                
            def basic_request(self, prompt, **kwargs):
                """Direct request without caching or persistence."""
                try:
                    response = self.query_handler.query(
                        prompt=str(prompt),
                        max_new_tokens=kwargs.get('max_tokens', self.max_tokens),
                        temperature=kwargs.get('temperature', self.temperature)
                    )
                    return [response] if response else ["Error: No response generated"]
                except Exception as e:
                    logger.error(f"Query handler failed: {e}")
                    return [f"Error: {str(e)}"]
        
        # Create wrapper
        storm_lm = SQLiteFreeWrapper(query_handler, max_tokens, temperature)
        
        # Configure DSPy with minimal settings to avoid SQLite
        dspy.configure(lm=storm_lm, rm=None)
        
        logger.info("✅ DSPy configured without SQLite (features disabled)")
        return storm_lm
        
    except ImportError:
        logger.error("DSPy not available - install with: pip install dspy-ai==2.4.9")
        return None
    except Exception as e:
        logger.error(f"Failed to configure SQLite-free DSPy: {e}")
        
        # If DSPy fails to import due to SQLite, try importing just the parts we need
        try:
            logger.info("Attempting minimal DSPy import...")
            
            # Create a minimal LM-like object that STORM can use
            class MinimalLM:
                def __init__(self, query_handler, max_tokens=512, temperature=0.7):
                    self.query_handler = query_handler
                    self.max_tokens = max_tokens
                    self.temperature = temperature
                    self.model = "qwen-local"
                
                def basic_request(self, prompt, **kwargs):
                    try:
                        response = self.query_handler.query(
                            prompt=str(prompt),
                            max_new_tokens=kwargs.get('max_tokens', self.max_tokens),
                            temperature=kwargs.get('temperature', self.temperature)
                        )
                        return [response] if response else ["Error: No response generated"]
                    except Exception as e:
                        return [f"Error: {str(e)}"]
                
                def __call__(self, prompt, **kwargs):
                    return self.basic_request(prompt, **kwargs)
            
            storm_lm = MinimalLM(query_handler, max_tokens, temperature)
            logger.info("✅ Created minimal LM wrapper (DSPy-free)")
            return storm_lm
            
        except Exception as fallback_error:
            logger.error(f"Even minimal LM creation failed: {fallback_error}")
            return None